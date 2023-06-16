#include <cstdlib>
#include <vector>
#include <cmath>
#include <fstream>

#include "../lib/gbm.hpp"

double geometric_brownian_motion(std::vector<double> &dat, unsigned int N, unsigned int epoch, std::default_random_engine &seed, bool plot) {
    std::vector<double> ret;
    for(int t = 1; t < dat.size(); t++)
        ret.push_back((dat[t] - dat[t-1]) / dat[t-1]);
    
    double s0 = dat.back();
    double mu = mean(ret);
    double sigma = stdev(ret);
    double drift = mu + 0.5 * pow(sigma, 2);

    std::normal_distribution<double> std_normal(0.0, 1.0);

    unsigned int up = 0;

    for(int e = 0; e < epoch; e++) {
        std::vector<double> brownian(N+1, 0);
        for(int t = 1; t <= N; t++)
            brownian[t] = brownian[t-1] + std_normal(seed);
        
        std::vector<double> path(N+1); path[0] = s0;
        for(int t = 1; t <= N; t++) {
            path[t] = s0 * exp(drift * t + sigma * brownian[t]);
            up += path[t] > s0;
        }

        if(plot) {
            std::ofstream gbm("./res/gbm", std::ios_base::app);
            for(double &x: path)
                gbm << x << " ";
            gbm << "\n";
            gbm.close();            
        }
    }

    return (double)up / (epoch * N);
}