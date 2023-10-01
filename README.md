# Geometric Brownian Motion and Generalized Deep Reinforcement Learning for Low-Frequency Trading

This AI trading model consists of two major components:

    (1) Geometric Brownian Motion for price simulation and short-term valuation cycle estimation

    (2) Generalized Deep Reinforcement Learning for Trading

## Geometric Brownian Motion for Estimating Short-Term Valuation Cycles

Geometric Brownian Motion is a stochastic process that models a randomly varying quantity following a Brownian motion with drift. It is a popular stochastic method for simulating stock prices that follow a trend while experiencing a random walk of up-and-downs characterizing risk.

http://www.columbia.edu/~ks20/FE-Notes/4700-07-Notes-BM.pdf

http://www.columbia.edu/~ks20/FE-Notes/4700-07-Notes-GBM.pdf

![alt text](https://github.com/junyoung-sim/gbm-drl-quant/blob/main/etc/gbm_sample_path.png)

![alt text](https://github.com/junyoung-sim/gbm-drl-quant/blob/main/etc/gbm_lognormal_prices.png)

Using the simulation shown above, we can estimate the probability that the value of a security would be greater than the current value during the N-day period following the current value. At every time ***t***, observe the security's historical data during the past N days and simulate M sample paths for the next K days from ***t***. This will yield a lognormal distribution of price values as shown in the second figure above. From that distribution of simulated prices, we may estimate the probability that the security's value would be greater than the current value during the K-day extrapolation period following ***t***. Due to the principles of GBM, this probability ("valuation score") is impacted by the short-term mean return and variation in return (volatility or risk).

Repeat the same procedure for every ***t*** and we obtain the following output.

![alt text](https://github.com/junyoung-sim/gbm-drl-quant/blob/main/etc/valuation_cycle_example.png)

The graph on the top shows the "valuation series" of the S&P 500 obtained from the simulation on each day. Notice that the stock index reverses direction once the valuation score reaches a critical value and crosses 0.50!

## GBM + Generalized Deep Reinforcement Learning = ?!

This trading model combines the aforementioned GBM simulation with Generalized Deep Reinforcement Learning for Trading (https://github.com/junyoung-sim/quant). Instead of observing the price series of a stock and major market-indicating securities (SPY, IEF, EUR=X, GSG) as in the original research, the trading model observes a multivariate state space (100-day look-back) that consists of the standardized valuation series of the stock of interest and the four market-indicating securities obtained through the GBM simulations with an observation period of 60 days, an extrapolation period of 20 days, and 1000 sample paths. The model utilizes the discrete reward function and adaptive synchronization demonstrated in the original quant model and research.

## Performance

DIA: Dow Jones Industrial Average (simulated price movements)
![alt text](https://github.com/junyoung-sim/gbm-drl-quant/blob/main/res/DIA-sim.png)

DIA: Dow Jones Industrial Average (up-to-date test performance)
![alt text](https://github.com/junyoung-sim/gbm-drl-quant/blob/main/res/DIA-test.png)

DIA: Dow Jones Industrial Average (distribution of valuation scores by model actions for DIA)
![alt text](https://github.com/junyoung-sim/gbm-drl-quant/blob/main/res/DIA-analytics.png)

QQQ: NASDAQ Composite Index (simulated price movements)
![alt text](https://github.com/junyoung-sim/gbm-drl-quant/blob/main/res/QQQ-sim.png)

QQQ: NASDAQ Composite Index (up-to-date test performance)
![alt text](https://github.com/junyoung-sim/gbm-drl-quant/blob/main/res/QQQ-test.png)

QQQ: NASDAQ Composite Index (distribution of valuation scores by model actions for QQQ)
![alt text](https://github.com/junyoung-sim/gbm-drl-quant/blob/main/res/QQQ-analytics.png)

Benchmark: DIA + QQQ (equal weight)

| Metric | Benchmark | Model  |
|--------|-----------|--------|
| E(R)   | 0.1144    | 0.3774 |
| SD(R)  | 0.1891    | 0.4364 |
| SR     | 0.6051    | 0.8649 |
| MDD    | 0.5263    | 0.2381 |

E(R) = annualized return, SD(R) = return standard deviation, SR = sharpe ratio, MDD = maximum drawdown

## Usage

~~~
make && ./exec build <tickers> ./models/checkpoint # for building a model
./exec test <tickers> ./models/checkpoint # for testing a model
~~~