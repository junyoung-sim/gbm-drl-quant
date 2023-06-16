#ifndef __QUANT_HPP_
#define __QUANT_HPP_

#include <cstdlib>
#include <vector>
#include <string>
#include <random>
#include <chrono>

#include "../lib/gbm.hpp"
#include "../lib/net.hpp"

class Quant
{
private:
    NeuralNetwork agent; // Deep Q-Network (DQN)
    NeuralNetwork target;

    unsigned int obs; // GBM simulation observation period (days)
    unsigned int ext; // GBM sample path extrapolation period (days)
    unsigned int epoch; // GBM simulation epoch

    std::vector<double> action_space;

    std::string checkpoint;
    std::default_random_engine seed;

public:
    Quant() {}
    Quant(std::string path): checkpoint(path) {
        obs = 120;
        ext = 60;
        epoch = 100;

        action_space = {1.0, 0.0, -1.0}; // long, idle, short

        init({{20,20},{20,20},{20,18},{18,16},{16,14},{14,12},{12,3}});
        load();
    }

    void init(std::vector<std::vector<unsigned int>> shape);
    void sync();

    std::vector<double> sample_state(std::vector<std::vector<double>> &dat, unsigned int t);

    unsigned int greedy(std::vector<double> &state);
    unsigned int epsilon_greedy(std::vector<double> &state, double eps);

    void save();
    void load();
};

#endif