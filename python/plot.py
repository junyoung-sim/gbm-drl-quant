#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

log = [float(val[:-1]) for val in open("./res/log", "r").readlines()[:-1]]

plt.plot(log)
plt.savefig("./res/log.png")