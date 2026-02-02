import numpy as np
import torch, torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Categorical, kl_divergence
from collections import deque
import copy
import time

# ---------------------------------------------------------------
# 2.  Actor–Critic network (shared torso, three heads)
# ---------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, branches=[7,4,4], hidden=128):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
       
        # One head per node
        self.policy0 = nn.Linear(hidden, branches[0])
        self.policy1 = nn.Linear(hidden, branches[1])
        self.policy2 = nn.Linear(hidden, branches[2])
       
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1))


    def forward(self, obs):
        z = self.base(obs)
        logits0 = self.policy0(z)
        logits1 = self.policy1(z)
        logits2 = self.policy2(z)
        value   = self.value(z).squeeze(-1)
        return (logits0, logits1, logits2), value



        # ---------------------------------------------------------------
# 4.  PPO fine-tuner (DEP-only reward)
# ---------------------------------------------------------------
class PPOFineTuner:
    def __init__(self,
                 model: nn.Module,
                 beta_kl: float = 1.0,
                 clip: float = 0.2,
                 lr: float = 3e-4,
                 vf_coef: float = 0.5,
                 ent_coef: float = 0.01,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 update_epochs: int = 1,
                 minibatch_size: int = 256,
                 actor_epochs =1  ,
                 critic_epochs = 1,
                 threshold = 20,
                 policy_coeff = 1,
                 kl_targ: float = 0.1):
        """
          model    : the network to fine-tune (will be updated)
          pi_base    : frozen behavior-cloned policy (no grad)
          beta_kl    : weight on the KL penalty term
        """
        self.pi_base   = copy.deepcopy(model).eval()           # freeze base policy
        for p in self.pi_base.parameters():
            p.requires_grad = False
            
        self.beta_kl   = beta_kl
        self.net = model
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.clip, self.vf_coef, self.ent_coef = clip, vf_coef, ent_coef
        self.gamma, self.lam = gamma, lam
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.actor_epochs = actor_epochs
        self.critic_epochs = critic_epochs
        self.threshold  = threshold
        self.p_coeff = policy_coeff
        self.kl_targ = kl_targ

    @staticmethod
    def _compute_gae(rewards, values, dones, last_value, gamma, lam):
        gae = 0.0
        returns = []
        values = values + [last_value]
        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]
            delta = rewards[step] + gamma * values[step+1] * mask - values[step]
            gae = delta + gamma * lam * mask * gae
            returns.insert(0, gae + values[step])
        advs = [ret - val for ret, val in zip(returns, values[:-1])]
        return returns, advs

    def train(self, env, epochs=50, steps_per_epoch=4096):
        for epoch in range(1, epochs+1):
            t_epoch0 = time.perf_counter()
            t_roll0 = time.perf_counter()
            # -----------------------------
            # 1) Collect one big rollout
            # -----------------------------
            obs_buf, act_buf, logp_buf = [], [], []
            rew_buf, done_buf = [], []
            o = env.reset()

            t_env = 0.0
            t_policy = 0.0
            for _ in range(steps_per_epoch):
                t0 = time.perf_counter()
                ob = torch.tensor(o, dtype=torch.float32)
                (l0,l1,l2), v = self.net(ob.unsqueeze(0))
                d0 = Categorical(logits=l0)
                d1 = Categorical(logits=l1)
                d2 = Categorical(logits=l2)
                a0, a1, a2 = d0.sample(), d1.sample(), d2.sample()
                logp = d0.log_prob(a0) + d1.log_prob(a1) + d2.log_prob(a2)
                a = torch.tensor([a0.item(),a1.item(),a2.item()], dtype=torch.long)
                t_policy += time.perf_counter() - t0
                
                t1 = time.perf_counter()
                t_env += time.perf_counter() - t1

                o2, r, done, _ = env.step(a.numpy())
                obs_buf.append(ob)
                act_buf.append(a)
                logp_buf.append(logp.item())
                rew_buf.append(r)
                done_buf.append(float(done))
                o = o2
                if done:
                    o = env.reset()

            t_roll = time.perf_counter() - t_roll0

            # bootstrap last value for return computation
            t_prep0 = time.perf_counter()

            with torch.no_grad():
                (_, _, _), last_v = self.net(torch.tensor(o,dtype=torch.float32).unsqueeze(0))
                last_v = last_v.item()

            # compute undiscounted return targets ret_t once
            # (we’ll reuse these for both critic and actor)
            returns, _ = self._compute_gae(rew_buf,      # rewards
                                           [0]*len(rew_buf),  # dummy vals
                                           done_buf,
                                           last_v,
                                           self.gamma,
                                           self.lam)
            ret_t = torch.tensor(returns, dtype=torch.float32)

            # stack everything
            obs_t     = torch.stack(obs_buf)
            acts_t    = torch.stack(act_buf)
            old_logp_t= torch.tensor(logp_buf, dtype=torch.float32)

            dataset = TensorDataset(obs_t, acts_t, old_logp_t, ret_t)
            loader  = DataLoader(dataset,
                                 batch_size=self.minibatch_size,
                                 shuffle=True)

            t_prep = time.perf_counter() - t_prep0

            # ---------------------------------
            # 2) Critic‐only phase
            # ---------------------------------
            # freeze policy, unfreeze critic
            t_critic0 = time.perf_counter()

            for p in self.net.policy0.parameters(): p.requires_grad = False
            for p in self.net.policy1.parameters(): p.requires_grad = False
            for p in self.net.policy2.parameters(): p.requires_grad = False
            for p in self.net.value.parameters(): p.requires_grad = True
            for p in self.net.base.parameters():       p.requires_grad = True

            total_vloss = 0.0
            total_samples_v = 0
            for _ in range(self.critic_epochs):
                for mb_obs, mb_acts, mb_oldlp, mb_ret in loader:
                    # forward for value only
                    (_, _, _), vals = self.net(mb_obs)
                    vloss = F.mse_loss(vals, mb_ret)
                    self.opt.zero_grad()
                    vloss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                    self.opt.step()

                    bs = mb_obs.size(0)
                    total_vloss    += vloss.item() * bs
                    total_samples_v+= bs

            avg_vloss = total_vloss / total_samples_v
            t_critic = time.perf_counter() - t_critic0

            # ---------------------------------
            # 3) Recompute advantages
            # ---------------------------------
            # now that critic is trained, get fresh value estimates
            t_adv0 = time.perf_counter()
            with torch.no_grad():
                (_, _, _), all_vals = self.net(obs_t)
            adv_t = ret_t - all_vals
            #adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            # replace dataset with adv included
            dataset = TensorDataset(obs_t, acts_t, old_logp_t, ret_t, adv_t)
            loader  = DataLoader(dataset,
                                 batch_size=self.minibatch_size,
                                 shuffle=True)
            t_adv = time.perf_counter() - t_adv0
            # ---------------------------------
            # 4) Actor‐only phase
            # ---------------------------------
            t_actor0 = time.perf_counter()

            for p in self.net.value.parameters(): p.requires_grad = False
            for p in self.net.policy0.parameters():    p.requires_grad = True
            for p in self.net.policy1.parameters():    p.requires_grad = True
            for p in self.net.policy2.parameters():    p.requires_grad = True

            total_ploss = 0.0
            total_ent   = 0.0
            total_samples_a = 0
            klp_total   = 0.0

            for _ in range(self.actor_epochs):
                for mb_obs, mb_acts, mb_oldlp, mb_ret, mb_adv in loader:
                    (l0_raw,l1_raw,l2_raw), _ = self.net(mb_obs)
                     # 2) clamp them to keep exp(logit) numerically safe
                    l0 = torch.clamp(l0_raw, min=-self.threshold, max=self.threshold)
                    l1 = torch.clamp(l1_raw, min=-self.threshold, max=self.threshold)
                    l2 = torch.clamp(l2_raw, min=-self.threshold, max=self.threshold)
                    d0 = Categorical(logits=l0)
                    d1 = Categorical(logits=l1)
                    d2 = Categorical(logits=l2)
                    logp = d0.log_prob(mb_acts[:,0]) \
                         + d1.log_prob(mb_acts[:,1]) \
                         + d2.log_prob(mb_acts[:,2])
                    ratio = torch.exp(logp - mb_oldlp)
                    s1 = ratio * mb_adv
                    s2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * mb_adv
                    ploss = -torch.min(s1, s2).mean()
                    ent   = (d0.entropy()+d1.entropy()+d2.entropy()).mean()
                    
                   
                    # KL‐penalty to base
                    with torch.no_grad():
                        (b0_raw,b1_raw,b2_raw), _ = self.pi_base(mb_obs)
                    b0 = torch.clamp(b0_raw, min=-self.threshold, max=self.threshold)
                    b1 = torch.clamp(b1_raw, min=-self.threshold, max=self.threshold)
                    b2 = torch.clamp(b2_raw, min=-self.threshold, max=self.threshold)
                    db0,db1,db2 = Categorical(logits=b0),Categorical(logits=b1),Categorical(logits=b2)
                    klp = (kl_divergence(d0,db0)+
                           kl_divergence(d1,db1)+
                           kl_divergence(d2,db2)).mean()

                    loss = self.p_coeff*ploss - self.ent_coef*ent + self.beta_kl*klp

                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                    self.opt.step()

                    bs = mb_obs.size(0)
                    total_ploss   += ploss.item() * bs
                    total_ent     += ent.item()   * bs
                    total_samples_a += bs
                    klp_total += klp.item() * bs

            avg_ploss = total_ploss / total_samples_a
            avg_ent   = total_ent   / total_samples_a
            avg_klp   = klp_total   /total_samples_a

            t_actor = time.perf_counter() - t_actor0

            if avg_klp > self.kl_targ:
                self.beta_kl *= 1.5
            elif avg_klp <0.05:
                self.beta_kl = 0.1

            t_epoch = time.perf_counter() - t_epoch0

            # ---- final epoch log ----
            print(f"[Epoch {epoch}/{epochs}]  "
                  f"CriticLoss={avg_vloss:.4f}  "
                  f"ActorLoss={avg_ploss:.4f}  "
                  f"Entropy={avg_ent:.4f} "
                  f"KLDivg = {avg_klp:.4f}")

            print(
            f"[Epoch {epoch}/{epochs}] "
            f"rollout={t_roll:.2f}s (policy={t_policy:.2f}s, env={t_env:.2f}s) | "
            f"prep={t_prep:.2f}s | critic={t_critic:.2f}s | adv={t_adv:.2f}s | actor={t_actor:.2f}s | "
            f"epoch_total={t_epoch:.2f}s"
        )



    