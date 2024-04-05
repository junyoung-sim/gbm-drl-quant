# Low-Frequency Momentum Trading via Geometric Brownian Motion and Generalized Deep Reinforcement Learning

## Geometric Brownian Motion for Estimating Valuation Cycles

Geometric Brownian Motion is a stochastic process that models a randomly varying quantity following a Brownian Motion with drift. It is a popular stochastic method for simulating stock prices that follow a trend while experiencing a random walk.

http://www.columbia.edu/~ks20/FE-Notes/4700-07-Notes-BM.pdf

http://www.columbia.edu/~ks20/FE-Notes/4700-07-Notes-GBM.pdf

![alt text](https://github.com/junyoung-sim/gbm-drl-quant/blob/main/etc/gbm_sample_path.png)

![alt text](https://github.com/junyoung-sim/gbm-drl-quant/blob/main/etc/gbm_lognormal_prices.png)

Using the simulation shown above, we can estimate the probability that the value of a security would be greater than the current value. At every time ***t***, observe the security's historical data during the past N days and simulate M sample paths for the next K days from ***t***. This will yield a lognormal distribution of price values as shown in the second figure above. From that distribution of simulated prices, we may estimate the probability that the security's value would be greater than the current value during the K-day extrapolation period following ***t***. Let this probability be the "valuation score" of the security.

Repeat the same procedure for every ***t*** and simulation yields the following output.

![alt text](https://github.com/junyoung-sim/gbm-drl-quant/blob/main/etc/vscore.png)

The graph above shows the valuation series of the S&P 500. Notice that the critical values and zero-crossings in the valuation series coincide with the extremas and major turning points in the stock index's price movement.

## GBM + Generalized Deep Reinforcement Learning

This trading model improves the original version of Generalized Deep Reinforcement Learning for Trading (https://github.com/junyoung-sim/quant) by observing the valuation series (obs=60-days, ext=20-days, epoch=1000) of a stock the four market-indicating securities (SPY, IEF, EUR=X, CL=F) instead of their raw price series. All else are kept constant (discrete reward function, standard DQN, adaptive sync) with the exception of some learning hyperparameters (constant learning rate of 10^-6, fixed replay capacity of 25000).

Note: CPU multithreading is implemented to expedite the valuation score simulations.

## Performance

The baseline model was trained on the S&P 500 top 50 holdings. The model is regularly updated to maintain and improve performance.

**Equal-Weight S&P 500 Top 100 (Dec 2023)**

| Metric | Benchmark | Model  |
|--------|-----------|--------|
| E(R)   | 0.1453    | 0.3730 |
| SD(R)  | 0.2773    | 0.5063 |
| SR     | 0.5238    | 0.7369 |
| MDD    | 0.5974    | 0.4554 |

E(R) = annualized return, SD(R) = return standard deviation, SR = sharpe ratio, MDD = maximum drawdown

JPM (performance)
![alt text](https://github.com/junyoung-sim/gbm-drl-quant/blob/main/res/test/v2/JPM-test.png)

JPM (model behavior)
![alt text](https://github.com/junyoung-sim/gbm-drl-quant/blob/main/res/test/v2/JPM-analytics.png)

BLK (performance)
![alt text](https://github.com/junyoung-sim/gbm-drl-quant/blob/main/res/test/v2/BLK-test.png)

BLK (model behavior)
![alt text](https://github.com/junyoung-sim/gbm-drl-quant/blob/main/res/test/v2/BLK-analytics.png)

## Usage

~~~
make && ./exec build <tickers> ./models/checkpoint # for building a model
./exec test <tickers> ./models/checkpoint # for testing a model
~~~
