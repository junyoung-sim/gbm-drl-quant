#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OBS = 100 # observation period
EXT = 20 # extrapolation period
EPOCH = 1000 # simulation epochs

def main():
    ticker = sys.argv[1]
    raw = pd.read_csv("./data/{}.csv" .format(ticker))

    # compute daily retuns
    t0 = np.array(raw["adjClose"][0:-1])
    t1 = np.array(raw["adjClose"][1:])
    returns = (t1 - t0) / t0
    
    raw = raw.iloc[1:]
    scores = []
    for T in range(OBS-1, raw.shape[0]):
        s0 = raw["adjClose"][T] # current value
        mu = returns[T+1-OBS:T+1].mean() # mean daily return
        sigma = np.std(returns[T+1-OBS:T+1]) # variability in daily returns
        drift = mu + 0.5 * sigma**2 # drift of random walk

        brownian = np.random.normal(0, 1, (EPOCH, EXT)) # brownian shock values
        brownian = np.cumsum(brownian, axis=1)

        t = np.arange(1, EXT+1) # extrapolation time
        path = s0 * np.exp(drift * t + sigma * brownian) # simulated paths

        # P(s_t > s0 | t E {1, 2, 3, ..., EXT}) (valuation score)
        score = float(sum(path.flatten() > s0)) / (EPOCH * EXT)
        scores.append(score)

        print("GBM SIM. T={} @ {} SCORE={}" .format(T, ticker, score))

        # save recent simulations
        if T == raw.shape[0] - 1:
            path = np.insert(path, 0, s0, axis=1)
            for p in path:
                plt.plot(p)
            plt.savefig("./res/{}-sim.png" .format(ticker))
    
    # standardize scores
    scores = np.array(scores)
    scores = (scores - scores.mean()) / np.std(scores)

    # insert valuation scores to dataframe
    raw = raw.iloc[-scores.shape[0]:]
    raw["gbm_score"] = scores

    raw.to_csv("./data/{}.csv" .format(ticker), index=False)

if __name__ == "__main__":
    main()