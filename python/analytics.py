#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ticker = sys.argv[1]
log = pd.read_csv("./res/log")

short_valuations = np.array(log["X"][log["action"] == 0])
idle_valuations = np.array(log["X"][log["action"] == 1])
long_valuations = np.array(log["X"][log["action"] == 2])

plt.figure(figsize=(10,15))

plt.subplot(3, 1, 1)
ax = sns.histplot(data=short_valuations, binwidth=0.1, binrange=(-3,3), stat="probability")
ax.set(xlabel="Short Valuations", ylabel="Frequency")

plt.subplot(3, 1, 2)
ax = sns.histplot(data=idle_valuations, binwidth=0.1, binrange=(-3,3), stat="probability")
ax.set(xlabel="Idle Valuations", ylabel="Frequency")

plt.subplot(3, 1, 3)
ax = sns.histplot(data=long_valuations, binwidth=0.1, binrange=(-3,3), stat="probability")
ax.set(xlabel="Long Valuations", ylabel="Frequency")

plt.savefig("./res/{}-analytics.png" .format(ticker))