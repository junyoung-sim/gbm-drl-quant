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

    Memory(std::vector<double> &s, unsigned int a, double opt): action(a), optimal(opt) { state.swap(s); }
};

class Quant
{
private:
    std::vector<std::string> tickers;
    std::vector<std::string> indicators;

    Net agent;
    Net target;
    std::string checkpoint;
    std::default_random_engine *seed;

    std::vector<double> action_space = {-1.0, 0.0, 1.0}; // short, idle, long

public:
    Quant() {}
    Quant(std::vector<std::string> &t, std::vector<std::string> &i, std::default_random_engine &s, std::string path) {
        seed = &s; tickers.swap(t); indicators.swap(i);
        init({{500,500},{500,500},{500,500},{500,500},{500,500},{500,3}}); load();
    }

    void init(std::vector<std::vector<unsigned int>> shape);
    void sync();

    std::vector<std::vector<double>> generate_environment(std::string &ticker);
    std::vector<double> sample_state(std::vector<std::vector<double>> &env, unsigned int t);

    unsigned int greedy(std::vector<double> &state);
    unsigned int epsilon_greedy(std::vector<double> &state, double eps);

    void build();
    void sgd(Memory &memory, double alpha, double lambda);

    void test();

    void save();
    void load();
};

#endif