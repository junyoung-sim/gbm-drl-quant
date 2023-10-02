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
    amap = np.zeros((3, 5))
    action = [0, 1, 2] # short, idle, long
    valuation = [-2, -1, 0, 1, 2] # valuation scores (int)
    for a in action:
        for v in valuation:
            if v == -2:
                pv = np.sum(np.trunc(log[col]) <= v) / log[col].shape[0]
                pav = np.sum([(np.trunc(log[col]) <= v) & (log["action"] == a)]) / log[col].shape[0]
            elif v == 2:
                pv = np.sum(np.trunc(log[col]) >= v) / log[col].shape[0]
                pav = np.sum([(np.trunc(log[col]) >= v) & (log["action"] == a)]) / log[col].shape[0]
            else:
                pv = np.sum(np.trunc(log[col]) == v) / log[col].shape[0]
                pav = np.sum([(np.trunc(log[col]) == v) & (log["action"] == a)]) / log[col].shape[0]
            if pv == 0:
                amap[a][v+2] = 0
            else:
                amap[a][v+2] = pav / pv
    
    ax = plt.subplot(2, 3, k)
    ax.plot(valuation, amap[0], label="Short", color="steelblue")
    ax.plot(valuation, amap[1], label="Idle", color="lightskyblue")
    ax.plot(valuation, amap[2], label="Long", color="cadetblue")    
    ax.vlines(log[col].iloc[-1], 0, 1, color="red")

    plt.ylabel("Probability")
    plt.xlabel("{} Valuation Score" .format(col))
    plt.legend()

    k += 1

plt.savefig("./res/{}-analytics.png" .format(ticker))