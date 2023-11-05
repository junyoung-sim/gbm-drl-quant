#include <cstdlib>
#include <vector>
#include <cmath>

#include "../lib/gbm.hpp"

std::vector<double> returns(std::vector<double> &raw) {
    std::vector<double> r;
    for(int i = 1; i < raw.size(); i++)
        r.push_back((raw[i] - raw[i-1]) / raw[i-1]);
    return r;
}

void vscore(std::vector<double> raw, std::vector<double> *v, std::default_random_engine seed) {
    for(int t = OBS-1; t < raw.size(); t++) {
        std::vector<double> temp = {raw.begin() + t+1-OBS, raw.begin() + t+1};
        std::vector<double> ret = returns(temp);
        double s0 = temp.back();
        double mu = mean(ret);
        double sigma = stdev(ret);
        double drift = mu + 0.5 * pow(sigma, 2);

        std::vector<std::vector<double>> path(EPOCH, std::vector<double>(EXT));
        normal(path, seed); cumsum(path);

        unsigned int sum = 0;
        for(int i = 0; i < EPOCH; i++) {
            for(int j = 0; j < EXT; j++) {
                path[i][j] *= sigma;
                path[i][j] += drift * (j+1);
                path[i][j] = s0 * exp(path[i][j]);
                sum += (path[i][j] > s0);
            }
        }
        v->push_back((double)sum / (EPOCH * EXT));
    }

    standardize(*v);
}