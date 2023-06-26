#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt

def main():
    df = None
    for i in range(1, len(sys.argv)):
        ticker = sys.argv[i]
        dat = pd.read_csv("./data/{}.csv" .format(ticker))
        dat = dat.rename(columns={"gbm_score": ticker})

        if i == 1:
            dat = dat.rename(columns={"adjClose": "raw"})
            df = dat
        else:
            dat = dat.drop(columns=["adjClose"])
            df = df.merge(dat, on="date")
    
    df = df.drop(columns=["date"])
    df.to_csv("./data/cleaned.csv", index=False)

if __name__ == "__main__":
    main()