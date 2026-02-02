# Re-import required libraries after state reset
import numpy as np
import matplotlib.pyplot as plt
import torch





def evaluate_and_plot_policy_per_node(policy, env, steps=5000):
    dep_buf = np.zeros((steps, len(env.forwarding_nodes)))
    thr_buf = np.zeros_like(dep_buf)
    q_buf   = np.zeros_like(dep_buf)
    #arival_buf =  np.zeros((steps, 1))
    arival_buf =  np.zeros_like(dep_buf)
    netHop_buf = np.zeros_like(dep_buf)
    power_buf = np.zeros_like(dep_buf)

    obs = env.reset()
    for t in range(steps):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        (l0, l1, l2), _ = policy(obs_t)
        a0 = int(l0.argmax()); a1 = int(l1.argmax()); a2 = int(l2.argmax())
        action = np.array([a0, a1, a2], dtype=int)

        obs, _, _, info = env.step(action)

        dep_buf[t] = info["dep_list"]
        thr_buf[t] = info["throughput_list"]
        q_buf[t]   = info["queue_list"]
        arival_buf[t] = info["arrival_list"]
        netHop_buf[t] = info["next_hop_list"]
        power_buf[t] = info["power_list"]

    node_ids = env.forwarding_nodes  # e.g. [0,1,2]

    # 1) DEP per node (cumulative)
    plt.figure(figsize=(8,4))
    for i, node in enumerate(node_ids):
        plt.plot(np.cumsum(dep_buf[:, i]), label=f"Node {node}")
    plt.title("Cumulative DEP per Node")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative DEP")
    plt.legend()
    plt.grid(True)

    # 2) Throughput per node (cumulative)
    plt.figure(figsize=(8,4))
    for i, node in enumerate(node_ids):
        plt.plot(np.cumsum(thr_buf[:, i]), label=f"Node {node}")
    plt.title("Cumulative Throughput per Node")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Throughput")
    plt.legend()
    plt.grid(True)

    # 3) Queue length per node
    plt.figure(figsize=(8,4))
    for i, node in enumerate(node_ids):
        plt.plot(np.cumsum(q_buf[:, i]), label=f"Node {node}")
    plt.title("Queue Length per Node Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Queue Length")
    plt.legend()
    plt.  grid(True)

    plt.show()
    return dep_buf, thr_buf, q_buf,arival_buf,netHop_buf, power_buf



def plot_mean_std_bar(q_buf, label, ylabel):
  means = q_buf.mean(axis=0)
  stds  = q_buf.std(axis=0)
  N = q_buf.shape[1]
  nodes = np.arange(N)

  plt.figure(figsize=(6,4))
  plt.bar(nodes, means, yerr=stds, capsize=5)
  plt.xticks(nodes, [f"Node {i}" for i in nodes])
  plt.title("Average " + label + " ± 1 σ")
  plt.xlabel("Node")
  plt.ylabel(ylabel)
  plt.grid(axis="y")
  plt.show()



def plot_all_nodes_moving_average(q_buf, label, ylabel, ma_window=1000):
  """
  q_buf: shape [T, N], where T is time steps, N is number of nodes
  ma_window: window size for moving average
  """
  T, N = q_buf.shape
  t = np.arange(T)

  plt.figure(figsize=(8, 4))
  for node in range(N):
      ma = np.convolve(q_buf[:, node], np.ones(ma_window)/ma_window, mode='same')
      plt.plot(t, ma, label=f"Node {node}")

  plt.title("Moving Average " + label +  f" (window={ma_window})")
  plt.xlabel("Time Step")
  plt.ylabel(ylabel)
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()


def plot_all_nodes_moving_std(q_buf, label, ylabel, ma_window=1000):
  """
  q_buf: shape [T, N]  (T steps, N nodes)
  ma_window: window size for moving std
  """
  T, N = q_buf.shape
  t = np.arange(T)

  # pre-compute the box-filter kernel once
  kernel = np.ones(ma_window) / ma_window

  plt.figure(figsize=(8, 4))
  for node in range(N):
      x = q_buf[:, node]

      # moving mean
      mean = np.convolve(x, kernel, mode='same')

      # moving mean of squares
      mean_sq = np.convolve(x**2, kernel, mode='same')

      # moving std = sqrt(E[x²] – (E[x])²); clip for numerical safety
      std = np.sqrt(np.clip(mean_sq - mean**2, a_min=0, a_max=None))

      plt.plot(t, std, label=f"Node {node}")

  plt.title(f"Moving Std (window={ma_window}) — {label}")
  plt.xlabel("Time Step")
  plt.ylabel(ylabel)
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()
