import numpy as np
import torch, torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
from grid_routing_env import GridRoutingEnv  
# ---------------------------------------------------------------
# 1.  Simple Back-Pressure expert (central, deterministic)
# ---------------------------------------------------------------
class BackPressureExpert:
    def __init__(self, env: GridRoutingEnv, lam_p=0.0):
        self.env = env
        self.lam = lam_p                       # power penalty

    def act(self, obs):
        """Return MultiDiscrete joint action [a0,a1,a2]"""
        queues = obs[-4:]                      # q0,q1,q2,q3
        # gamma ordering (0→1,0→2,1→3,2→3):
        g01,g02,g13,g23 = obs[:4]

        joint_action = []

        # ----- node 0 decisions -----
        scores0 = []
        for nh, P, R in self.env.node_action_map[0]:
            if nh is None:                     # WAIT
                scores0.append(-1e9)
            else:
                qdiff = queues[0] - queues[nh]
                snr   = P * (g01 if nh==1 else g02) / self.env.N0
                cap   = np.log2(1+snr)
                score = qdiff * cap - self.lam * P
                scores0.append(score)
        joint_action.append(int(np.argmax(scores0)))

        # ----- node 1 decisions -----
        scores1=[]
        for nh,P,R in self.env.node_action_map[1]:
            if nh is None: scores1.append(-1e9)
            else:
                qdiff = queues[1]-queues[nh]
                snr   = P*g13/self.env.N0
                cap   = np.log2(1+snr)
                scores1.append(qdiff*cap-self.lam*P)
        joint_action.append(int(np.argmax(scores1)))

        # ----- node 2 decisions -----
        scores2=[]
        for nh,P,R in self.env.node_action_map[2]:
            if nh is None: scores2.append(-1e9)
            else:
                qdiff = queues[2]-queues[nh]
                snr   = P*g23/self.env.N0
                cap   = np.log2(1+snr)
                scores2.append(qdiff*cap-self.lam*P)
        joint_action.append(int(np.argmax(scores2)))

        return np.array(joint_action, dtype=np.int32)