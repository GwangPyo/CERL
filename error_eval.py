import numpy as np


if __name__ == "__main__":
    tail_latency = 5
    latency_list = []
    for i in range(10000):
        latency = np.random.exponential(1)
        err = np.random.normal(scale=0.05)
        latency += err
        if latency > tail_latency:
            latency = tail_latency
        elif latency < 0.10001:
            latency = 0.10001
        latency = int(np.round(latency * 10))
        latency_list.append(latency)
    print(np.mean(latency_list))

