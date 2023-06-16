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
    for(unsigned int i = 0; i < env.size(); i++) {
        std::vector<double> dat = {env[i].begin() + t + 1 - obs, env[i].begin() + t + 1}; // observation window
        double score = geometric_brownian_motion(dat, ext, epoch, seed, false); // valuation score
        state.push_back(score);
    }
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