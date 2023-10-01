#!/usr/bin/env python3

import sys
import pandas as pd
import matplotlib.pyplot as plt

path = sys.argv[1]
log = pd.read_csv("./res/log")

plt.figure(figsize=(30,20))

plt.subplot(4, 2, 1)
plt.title("State: X")
plt.plot(log["X"], label="X", color="cadetblue")

plt.subplot(4, 2, 2)
plt.title("State: SPY")
plt.plot(log["SPY"], label="SPY", color="steelblue")

plt.subplot(4, 2, 4)
plt.title("State: IEF")
plt.plot(log["IEF"], label="IEF", color="dodgerblue")

plt.subplot(4, 2, 6)
plt.title("State: GSG")
plt.plot(log["GSG"], label="GSG", color="lightskyblue")

plt.subplot(4, 2, 8)
plt.title("State: EUR=X")
plt.plot(log["EUR=X"], label="EUR=X", color="cornflowerblue")

#####

plt.subplot(4, 2, 3)
plt.title("Action")
plt.plot(log["action"], label="0: Short\n1: Idle\n2: Long", color="steelblue")

plt.subplot(4, 2, 5)
plt.title("Reward (Historical)")
plt.plot(log["benchmark"], label="Benchmark", color="cadetblue")
plt.plot(log["model"], label="Model", color="steelblue")

#####

benchmark = log["benchmark"][-252:]
model = log["model"][-252:]

benchmark = (benchmark - benchmark.iloc[0]) * 100 / benchmark.iloc[0]
model = (model - model.iloc[0]) * 100 / model.iloc[0]

plt.subplot(4, 2, 7)
plt.title("Reward (Recent)")
plt.plot(benchmark, label="Benchmark", color="cadetblue")
plt.plot(model, label="Model", color="steelblue")

plt.savefig("./res/{}.png" .format(path))
