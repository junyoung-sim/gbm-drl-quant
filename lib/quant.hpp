#ifndef __QUANT_HPP_
#define __QUANT_HPP_

#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <map>

#include "../lib/gbm.hpp"
#include "../lib/net.hpp"

#define TICKER 0

struct Memory {
    std::vector<double> state;
    unsigned int action;
    double optimal;

    Memory(std::vector<double> &s, unsigned int a, double opt) {
        state.swap(s);
        action = a;
        optimal = opt;
    }
};

typedef std::map<std::string, std::vector<std::vector<double>>> Environment;

class Quant
{
private:
    // Deep Q-Network (DQN)
    NeuralNetwork agent;
    NeuralNetwork target;

    const std::vector<double> action_space = {-1.0, 0.0, 1.0}; // short, idle, long

    const unsigned int paa_window = 5; // discretization window (5 days)
    const unsigned int obs = 100; // GBM simulation observation period (100 days)
    const unsigned int ext = 50; // GBM sample path extrapolation period (50 days)
    const unsigned int epoch = 100; // GBM simulation epoch

    std::string checkpoint;
    std::default_random_engine seed;

public:
    Quant() {}
    Quant(std::string path): checkpoint(path) {
        init({{505,505},{505,505},{505,505},{505,3}});
        load();
    }

    void init(std::vector<std::vector<unsigned int>> shape);
    void sync();

    std::vector<double> sample_state(std::vector<std::vector<double>> &env, unsigned int t);

    unsigned int greedy(std::vector<double> &state);
    unsigned int epsilon_greedy(std::vector<double> &state, double eps);

    void build(std::vector<std::string> &tickers, Environment &env, double train);
    void sgd(Memory &memory, double alpha, double lambda);

    void save();
    void load();
};

#endif