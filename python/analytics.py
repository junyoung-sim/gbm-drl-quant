#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ticker = sys.argv[1]
log = pd.read_csv("./res/log")

plt.figure(figsize=(15,8))

k = 1
for col in log.columns[:-3]:
    short_valuations = log[col][log["action"] == 0]
    idle_valuations = log[col][log["action"] == 1]
    long_valuations = log[col][log["action"] == 2]

    plt.subplot(2, 3, k)
    plt.hist(short_valuations, bins=np.arange(-3, 3.1, 0.1), histtype="step", label="Short", color="steelblue")
    plt.hist(idle_valuations, bins=np.arange(-3, 3.1, 0.1), histtype="step", label="Idle", color="lightskyblue")
    plt.hist(long_valuations, bins=np.arange(-3, 3.1, 0.1), histtype="step", label="Long", color="cadetblue")
    plt.ylabel("Count")
    plt.xlabel("{} Valuation Score" .format(col))
    plt.legend()

    k += 1

plt.savefig("./res/{}-analytics.png" .format(ticker))