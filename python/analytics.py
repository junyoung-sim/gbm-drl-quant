#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ticker = sys.argv[1]
log = pd.read_csv("./res/log")

short_valuations = log["X"][log["action"] == 0]
idle_valuations = log["X"][log["action"] == 1]
long_valuations = log["X"][log["action"] == 2]

plt.hist(short_valuations, bins=np.arange(-3, 3.1, 0.1), histtype="step", label="Short", color="steelblue")
plt.hist(idle_valuations, bins=np.arange(-3, 3.1, 0.1), histtype="step", label="Idle", color="lightskyblue")
plt.hist(long_valuations, bins=np.arange(-3, 3.1, 0.1), histtype="step", label="Long", color="cadetblue")
plt.ylabel("Count")
plt.xlabel("{} Valuation Score" .format(ticker))
plt.legend()

plt.savefig("./res/{}-analytics.png" .format(ticker))