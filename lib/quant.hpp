#ifndef __QUANT_HPP_
#define __QUANT_HPP_

#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <map>

#include "../lib/data.hpp"
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

    const unsigned int obs = 100; // observation period

    std::string checkpoint;
    std::default_random_engine seed;

public:
    Quant() {}
    Quant(std::string path): checkpoint(path) {
        init({{500,500},{500,500},{500,500},{500,500},{500,500},{500,3}});
        load();
    }

    void init(std::vector<std::vector<unsigned int>> shape);
    void sync();

    std::vector<double> sample_state(std::vector<std::vector<double>> &env, unsigned int t);

    unsigned int greedy(std::vector<double> &state);
    unsigned int epsilon_greedy(std::vector<double> &state, double eps);

    void build(std::vector<std::string> &tickers, Environment &env);
    void sgd(Memory &memory, double alpha, double lambda);

    void test(std::vector<std::string> &tickers, Environment &env);

    void save();
    void load();
};

#endif