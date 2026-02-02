# Updated GridRoutingEnv with per-node info logging and evaluation function

import gym
from gym import spaces
import numpy as np
import math
from scipy.special import gamma as gamma_fn, gammainc

class GridRoutingEnv(gym.Env):
    """
     Centralized environment for 2×2 grid (4 nodes), with
     perfect channel-state information (no noise).

       0 --- 1
       |     |
       2 --- 3
       Link order: (0→1),(0→2),(1→3),(2→3)
    """


     

    def __init__(self,
                 arrival_rate=1.0,
                 B=1.0,
                 T_slot=1.0,
                 L_pkt=1.0,
                 lambda_p=0.05,
                 mu_q=0.1,
                 mu_th = 0.0,
                 beta=1.0,
                 N0=1.0):
        super().__init__()
        # Topology
        self.num_nodes = 4
        self.source     = 0
        self.dest       = 3
        self.neighbours = {0:[1,2], 1:[3], 2:[3], 3:[]}
        self.forwarding_nodes = [0,1,2]

        # Power/rate modes
        self.power_levels = [10.0, 13.0, 16.0]
        self.rates        = [1.0, 2.0, 3.5]
        self.node_action_map = {}
        for node in self.forwarding_nodes:
            amap = [(None,0.0,0.0)]
            for nh in self.neighbours[node]:
                for P,R in zip(self.power_levels,self.rates):
                    amap.append((nh,P,R))
            self.node_action_map[node] = amap

        # Action space: one discrete for each forwarding node
        self.action_space = spaces.MultiDiscrete(
            [len(self.node_action_map[n]) for n in self.forwarding_nodes]
        )

        # Observation: 4 link gains + 4 queues
        obs_dim = 4 + self.num_nodes
        self.observation_space = spaces.Box(0.0, np.inf, (obs_dim,), np.float32)

        # Params
        self.arrival_rate = arrival_rate
        self.B            = B
        self.T_slot       = T_slot
        self.L_pkt        = L_pkt
        self.lambda_p     = lambda_p
        self.mu_q         = mu_q
        self.beta         = beta
        self.N0           = N0
        self.mu_th        = mu_th

        self.reset()

    def reset(self):
        self.queues = np.zeros(self.num_nodes, dtype=int)
        self.gammas = {
            (i,j): np.random.exponential()
            for i,nbs in self.neighbours.items() for j in nbs
        }
        return self._get_obs()

    def _get_obs(self):
        link_list = [(0,1),(0,2),(1,3),(2,3)]
        gh = np.array([self.gammas[lk] for lk in link_list], dtype=np.float32)
        q  = self.queues.astype(np.float32)
        return np.concatenate([gh, q])

    @staticmethod
    def _lower_inc_gamma(s, x):
        return gamma_fn(s) * gammainc(s, x)

    def DEP(self, P_t, gamma_val, L=1, Omega=1.0):
        if P_t <= 0 or gamma_val <= 0:
            return 1.0
        X = Omega * self.N0 / (P_t * gamma_val)
        ln_term = np.log1p(1/X)
        arg1 = L*(1+X)*ln_term
        arg2 = L*X*ln_term
        g1 = self._lower_inc_gamma(L, arg1)
        g2 = self._lower_inc_gamma(L, arg2)
        p = - (g1 - g2)/gamma_fn(L) + 1.0
        return float(np.clip(p, 0.0, 1.0))

    def step(self, actions):
        info = {
            "dep_list":        [],
            "throughput_list": [],
            "power_list":      [],
            "queue_list":      [],
            "arrival_list":    [],
             "next_hop_list": []
        }
        total_reward = 0.0

        # arrivals at source
        arr = np.random.poisson(self.arrival_rate)
        self.queues[self.source] += arr
        # record per-node arrival (only source has arrivals)
        for node in self.forwarding_nodes:
            info["arrival_list"].append(arr if node==self.source else 0)

        # apply actions per node
        for idx, node in enumerate(self.forwarding_nodes):
            ai = int(actions[idx])
            nh, P, R = self.node_action_map[node][ai]
            q_before = self.queues[node]

            # record which neighbor (or None) was chosen
            info["next_hop_list"].append(nh)

            if nh is None or q_before==0:
                served = 0
                dep_val = 0.0
                power   = 0.0
                actual = 0
                r_i     = -self.mu_q * self.queues[node]
                #r_i = 0.0
            else:
                gamma = self.gammas[(node,nh)]
                dep_val = self.DEP(P, gamma)
               
                snr     = P*gamma/self.N0
                cap     = math.log2(1+snr)
                mu_max  = math.floor(R*self.B*self.T_slot/self.L_pkt)
                served  = mu_max if cap>=R else 0
                actual  = min(served, q_before)
                self.queues[node]    -= actual
                self.queues[nh]      += actual
                #We always assume destination node is zero for Back-pressure sake!
                self.queues[3] = 0
                
                power = P
                #r_i = self.beta*dep_val - self.lambda_p*P - self.mu_q*q_before

                """ New reward formulation in next lines 
                    (commented old one in the above line) 
                """
                succ = 1 if actual > 0 else 0
                if node == 2 and nh == 3:
                  dep_val = actual*1*succ
                else:
                  dep_val = actual*succ*dep_val
                
                r_i = self.beta*dep_val - self.lambda_p*P + self.mu_th*actual - self.mu_q*self.queues[node]

            total_reward += r_i
            info["dep_list"].append(dep_val)
            info["throughput_list"].append(actual)
            info["power_list"].append(power)
            info["queue_list"].append(q_before)


        # update fading
        for link in self.gammas:
            self.gammas[link] = np.random.exponential()

        obs = self._get_obs()
        return obs, total_reward, False, info







