#include <cstdlib>
#include <vector>
#include <cmath>
#include <fstream>
#include <random>
#include <chrono>

#include "../lib/gbm.hpp"

std::vector<double> geometric_brownian_motion(std::vector<double> &dat, unsigned int N, unsigned int epoch, std::default_random_engine &seed, bool plot) {
    std::vector<double> returns; // daily returns
    for(int t = 1; t < dat.size(); t++)
        returns.push_back((dat[t] - dat[t-1]) / dat[t-1]);
    
    double s0 = dat.back();
    double mu = mean(returns); // mean return
    double sigma = stdev(returns); // variability in returns (volatility or risk)
    double drift = mu + 0.5 * pow(sigma, 2); // drift of random walk

    std::ofstream out("./res/simulation"); // save simulated paths

    std::normal_distribution<double> std_normal(0.0, 1.0);

    unsigned int score = 0;

    for(int e = 0; e < epoch; e++) {
        std::vector<double> brownian(N+1, 0); // brownian motion shocks
        for(int t = 1; t <= N; t++)
            brownian[t] = brownian[t-1] + std_normal(seed);
        
        std::vector<double> path(N+1); path[0] = s0; // simulate path
        for(int t = 1; t <= N; t++) {
            path[t] = s0 * exp(drift * t + sigma * brownian[t]);
            score += path[t] > s0;
        }

        if(plot) {
            for(double &x: path)
                out << x << " ";
            out << "\n";
        }
    }

    out.close();

    // P(s_t > s0 | t E {1, 2, ... , N}) (valuation score) and mu (mean return)
    return std::vector<double>{(double)score / (epoch * N), mu};
}