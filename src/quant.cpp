#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <fstream>

#include "../lib/quant.hpp"

void Quant::init(std::vector<std::vector<unsigned int>> shape) {
    // agent and target networks have idential shapes
    for(unsigned int l = 0; l < shape.size(); l++) {
        unsigned int in = shape[l][0], out = shape[l][1];
        agent.add_layer(in, out);
        target.add_layer(in, out);
    }

    srand(time(NULL));
    seed.seed(std::chrono::system_clock::now().time_since_epoch().count());

    agent.init(seed);
    sync();
}

void Quant::sync() {
    // synchronize target network to agent network
    for(unsigned int l = 0; l < agent.num_of_layers(); l++) {
        for(unsigned int n = 0; n < agent.layer(l)->out_features(); n++) {
            for(unsigned int i = 0; i < agent.layer(l)->in_features(); i++) {
                double weight = agent.layer(l)->node(n)->weight(i);
                target.layer(l)->node(n)->set_weight(i, weight);
            }

            double bias = agent.layer(l)->node(n)->bias();
            target.layer(l)->node(n)->set_bias(bias);
        }
    }
}

std::vector<double> Quant::sample_state(std::vector<std::vector<double>> &env, unsigned int t) {
    std::vector<double> state;
    return state;
}

unsigned int Quant::greedy(std::vector<double> &state) {
    std::vector<double> q = agent.predict(state);
    return std::max_element(q.begin(), q.end()) - q.begin(); // exploit greedy action
}

unsigned int Quant::epsilon_greedy(std::vector<double> &state, double eps) {
    unsigned int action = greedy(state);
    double explore = (double)rand() / RAND_MAX;
    if(explore < eps) {
        action = rand() % action_space.size(); // explore random action
        std::cout << "(E) ";
    }
    else
        std::cout << "(P) ";
    return action;
}

void Quant::build(std::vector<std::string> &tickers, Environment &env, double train) {
    unsigned int env_size = 0;
    for(std::string &ticker: tickers)
        env_size += env[ticker][TICKER].size() * train - obs;
    
    // learning hyperparameters

    double eps_init = 1.00; // initial exploration rate
    double eps_min = 0.10; // final exploration rate
    double gamma = 0.80; // long-term reward discount factor

    std::vector<Memory> replay_memory; // store past experiences
    unsigned int capacity = (unsigned int)(env_size * 0.10); // replay memory capacity
    unsigned int batch_size = 10; // batch size for SGD on agent network

    double alpha_init = 0.00001; // initial learning rate for SGD
    double alpha_min = 0.00000001; // final learning rate for SGD
    double alpha_decay = log(alpha_min) - log(alpha_init); // learning rate exponential decay rate
    double lambda = 0.10; // L2 regularization for SGD

    double eps = eps_init; // exploration rate
    double alpha = alpha_init; // learning for SGD

    unsigned int experiences = 0; // total number of states observed during training
    double rss = 0.00, mse = 0.00; // residual squared sum, mean squared error

    // train
}

void Quant::sgd(Memory &memory, double alpha, double lambda) { // stochastic gradient descent (mse)
    std::vector<double> q = agent.predict(memory.state);
    for(int l = agent.num_of_layers() - 1; l >= 0; l--) {
        double partial_gradient = 0.00, gradient = 0.00;
        for(unsigned int n = 0; n < agent.layer(l)->out_features(); n++) {
            if(l == agent.num_of_layers() - 1 && n != memory.action) continue; // ignore non-selected actions
            else {
                if(l == agent.num_of_layers() - 1)
                    partial_gradient = -2.00 * (memory.optimal - q[n]);
                else
                    partial_gradient = agent.layer(l)->node(n)->err() * relu_prime(agent.layer(l)->node(n)->sum());

                double updated_bias = agent.layer(l)->node(n)->bias() - alpha * partial_gradient;
                agent.layer(l)->node(n)->set_bias(updated_bias);

                for(unsigned int i = 0; i < agent.layer(l)->in_features(); i++) {
                    if(l == 0)
                        gradient = partial_gradient * memory.state[i];
                    else {
                        gradient = partial_gradient * agent.layer(l-1)->node(i)->act();
                        agent.layer(l-1)->node(i)->add_err(partial_gradient * agent.layer(l)->node(n)->weight(i));
                    }

                    gradient += lambda * agent.layer(l)->node(n)->weight(i);

                    double updated_weight = agent.layer(l)->node(n)->weight(i) - alpha * gradient;
                    agent.layer(l)->node(n)->set_weight(i, updated_weight);
                }
            }
        }
    }
}

void Quant::save() {
    std::ofstream out(checkpoint);
    if(out.is_open()) {
        for(unsigned int l = 0; l < agent.num_of_layers(); l++) {
            for(unsigned int n = 0; n < agent.layer(l)->out_features(); n++) {
                for(unsigned int i = 0; i < agent.layer(l)->in_features(); i++)
                    out << agent.layer(l)->node(n)->weight(i) << " ";
                out << agent.layer(l)->node(n)->bias() << "\n";
            }
        }
        out.close();
    }
}

void Quant::load() {
    std::ifstream out(checkpoint);
    if(out.is_open()) {
        for(unsigned int l = 0; l < agent.num_of_layers(); l++) {
            for(unsigned int n = 0; n < agent.layer(l)->out_features(); n++) {
                std::string line;
                std::getline(out, line);
                for(unsigned int i = 0; i < agent.layer(l)->in_features(); i++) {
                    double weight = std::stod(line.substr(0, line.find(" ")));
                    agent.layer(l)->node(n)->set_weight(i, weight);
                    line = line.substr(line.find(" ") + 1);
                }
                double bias = std::stod(line);
                agent.layer(l)->node(n)->set_bias(bias);
            }
        }
        sync();
        out.close();
    }
}