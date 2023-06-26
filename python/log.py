#!/usr/bin/env python3

import sys
import pandas as pd
import matplotlib.pyplot as plt

path = sys.argv[1]
log = pd.read_csv("./res/log")

plt.figure(figsize=(15,15))

plt.subplot(3, 1, 1)
plt.title("State")
plt.plot(log["X"], label="X", color="cadetblue")
plt.plot(log["SPY"], label="SPY", color="steelblue")
plt.plot(log["IEF"], label="IEF", color="dodgerblue")
plt.plot(log["GSG"], label="GSG", color="lightskyblue")
plt.plot(log["EUR=X"], label="EUR=X", color="cornflowerblue")
plt.ylim(0, 1)
plt.legend()

plt.subplot(3, 1, 2)
plt.title("Action")
plt.plot(log["action"], label="0: Short\n1: Idle\n2: Long", color="steelblue")
plt.legend()

plt.subplot(3, 1, 3)
plt.title("Reward")
plt.plot(log["benchmark"], label="Benchmark", color="cadetblue")
plt.plot(log["model"], label="Model", color="steelblue")
plt.legend()

plt.savefig("./res/{}.png" .format(path))
