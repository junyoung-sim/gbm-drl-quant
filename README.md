# Geometric Brownian Motion and Generalized Deep Reinforcement Learning for Low-Frequency Trading

***IN PROGRESS***

This AI trading model consists of two major components:

    (1) Geometric Brownian Motion for price simulation and short-term valuation cycle estimation

    (2) Generalized Deep Reinforcement Learning for Trading

## Geometric Brownian Motion for Estimating Short-Term Valuation Cycles

Geometric Brownian Motion is a stochastic process that models a randomly varying quantity following a Brownian motion with drift. It is a popular stochastic method for simulating stock prices that follow a trend while experiencing a random walk of up-and-downs characterizing risk. The following resources were used:

http://www.columbia.edu/~ks20/FE-Notes/4700-07-Notes-BM.pdf

http://www.columbia.edu/~ks20/FE-Notes/4700-07-Notes-GBM.pdf

![alt text](https://github.com/junyoung-sim/gbm-drl-quant/blob/main/etc/gbm_sample_path.png)

![alt text](https://github.com/junyoung-sim/gbm-drl-quant/blob/main/etc/gbm_lognormal_prices.png)

Using the simulation shown above, we can estimate the probability that the value of a security would be greater than the current value during the N-day period following the current value. At every time ***t***, observe the security's historical data during the past (100) days and simulate (100) sample paths for the next (50) days from ***t***. This will yield a lognormal distribution of price values as shown in the second figure above. From that distribution of simulated prices, we may estimate the probability that the security's value would be greater than the current value during the (50)-day extrapolation period following ***t***. Due to the principles of GBM, this probability ("valuation score") is impacted by the short-term mean return and variation in return (volatility or risk).

Repeat the same procedure for every ***t*** and we obtain the following output.

![alt text](https://github.com/junyoung-sim/gbm-drl-quant/blob/main/etc/valuation_cycle_example.png)

The graph on the top shows the valuation series of the S&P 500 obtained from the simulation on each day. Notice that the stock index reverses direction once the valuation score reaches a critical value!

## GBM + Generalized Deep Reinforcement Learning = ?!

This trading model combines the aforementioned GBM simulation with Generalized Deep Reinforcement Learning for Trading (https://github.com/junyoung-sim/quant). Instead of observing the PAA-discretized and standardized price series of a stock of interest and major market-indicating securities (SPY, IEF, EUR=X, GSG) as in the original research, the trading model observes a multivariate state space that consists of the standardized valuation series of the stock of interest and the four market-indicating securities obtained through the GBM simulations with an observation period of (100) days, an extrapolation period of (50) days, and (1000) sample paths.

The trading model was trained on the S&P 500 Top 100 holdings. 90% of the historical data was allocated for training. The remaining 10% was used for out-of-sample testing.

The following figure shows an example of the trading model's training performance (BLK).

![alt text](https://github.com/junyoung-sim/gbm-drl-quant/blob/main/res/1/BLK-train.png)

The following figure shows an example of the trading model's out-of-sample performance after training ().

![alt text]()

The following table summarizes and compares the trading model's out-of-sample performance to that of the buy-and-hold benchmark.

| Metric | Benchmark | Model  |
|--------|-----------|--------|
| E(R)   | 0.0000    | 0.0000 |
| SD(R)  | 0.0000    | 0.0000 |
| SR     | 0.0000    | 0.0000 |
| MDD    | 0.0000    | 0.0000 |

E(R) = annualized return, SD(R) = return standard deviation, SR = sharpe ratio, MDD = maximum drawdown

The following figure shows an example of up-to-date model performance ().

![alt text]()

***Refer to ./res for full build and test results along with up-to-date model outputs.***