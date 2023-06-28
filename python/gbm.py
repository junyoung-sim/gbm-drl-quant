#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OBS = 100 # observation period
EXT = 50 # extrapolation period
EPOCH = 1000 # simulation epochs

def main():
    ticker = sys.argv[1]
    raw = pd.read_csv("./data/{}.csv" .format(ticker))

    returns = []
    for t in range(1, raw.shape[0]):
        returns.append((raw["adjClose"][t] - raw["adjClose"][t-1]) / raw["adjClose"][t-1]) # daily returns
    
    returns = np.array(returns)
    raw = raw.iloc[1:]

    scores = []
    for T in range(OBS-1, raw.shape[0]):
        s0 = raw["adjClose"][T] # current value
        mu = returns[T+1-OBS:T+1].mean() # mean daily return
        sigma = np.std(returns[T+1-OBS:T+1]) # variability in daily returns
        drift = mu + 0.5 * sigma**2 # drift of random walk

        score = 0
        for i in range(EPOCH):
            brownian = np.random.normal(0, 1, EXT) # brownian shock values
            for k in range(1, EXT):
                brownian[k] += brownian[k-1]

            t = np.arange(1, EXT+1) # time
            path = s0 * np.exp(drift * t + sigma * brownian) # simulated path

            score += sum(path > s0)
        
        score = float(score) / (EPOCH * EXT) # P(s_t > s0 | t E {1, 2, 3, ... , EXT}) (valuation score)
        scores.append(score)

        print("GBM SIM. T={} @ {} SCORE={}" .format(T, ticker, score))
    
    scores = np.array(scores)
    raw = raw.iloc[-scores.shape[0]:]
    raw["gbm_score"] = scores

    raw.to_csv("./data/{}.csv" .format(ticker), index=False)

if __name__ == "__main__":
    main()