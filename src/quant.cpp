#include <cstdlib>
#include <vector>
#include <string>
#include <random>
#include <thread>
#include <algorithm>
#include <iostream>
#include <fstream>

#include "../lib/quant.hpp"

void Quant::init(std::vector<std::vector<unsigned int>> shape) {
    for(unsigned int l = 0; l < shape.size(); l++) {
        unsigned int in = shape[l][0], out = shape[l][1];
        agent.add_layer(in, out);
        target.add_layer(in, out);
    }
    agent.init(*seed); sync();
}

void Quant::sync() {
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

std::vector<std::vector<double>> Quant::generate_environment(std::string &ticker) {
    std::string cmd = "./python/download.py " + ticker;
    for(std::string &indicator: indicators)
        cmd += indicator + " ";
    fix_dsp(cmd); std::system(cmd.c_str());

    std::string merge = "./data/merge.csv"; fix_dsp(merge);
    std::vector<std::vector<double>> raw = read_csv(merge);
    std::vector<std::vector<double>> env(raw.size()+1, std::vector<double>());

    std::vector<std::thread> threads;
    for(int i = 0; i < raw.size(); i++) {
        std::thread th(vscore, raw[i], &env[i+1], *seed);
        threads.push_back(std::move(th));
    }
    for(int i = 0; i < raw.size(); i++) threads[i].join();

    env[TICKER] = {raw[TICKER].begin() + OBS - 1, raw[TICKER].end()};
    return env;
}

std::vector<double> Quant::sample_state(std::vector<std::vector<double>> &env, unsigned int t) {
    std::vector<double> state;
    for(unsigned int i = 1; i < env.size(); i++) {
        std::vector<double> dat = {env[i].begin() + t + 1 - OBS, env[i].begin() + t + 1};
        state.insert(state.end(), dat.begin(), dat.end());
    }
    return state;
}

unsigned int Quant::greedy(std::vector<double> &state) {
    std::vector<double> q = agent.predict(state);
    return std::max_element(q.begin(), q.end()) - q.begin();
}

unsigned int Quant::epsilon_greedy(std::vector<double> &state, double eps) {
    unsigned int action = greedy(state);
    double explore = (double)rand() / RAND_MAX;
    if(explore < eps) {
        action = rand() % action_space.size();
        std::cout << "(E) ";
    }
    else
        std::cout << "(P) ";
    return action;
}

void Quant::build() {
    const double EPS = 0.25;
    const double GAMMA = 0.80;
    const double ALPHA = 0.00000001;
    const double LAMBDA = 0.10;
    const unsigned int CAPACITY = 1000;
    const unsigned int BATCH_SIZE = 10;
    
    std::vector<Memory> replay;
    unsigned int experiences = 0;

    std::shuffle(tickers.begin(), tickers.end(), *seed);
    for(std::string &ticker: tickers) {
        std::ofstream out("./res/log");
        out << "X,SPY,IEF,GSG,EUR=X,action,benchmark,model\n";
        double benchmark = 1.00, model = 1.00;

        std::vector<std::vector<double>> env = generate_environment(ticker);
        const unsigned int START = OBS - 1, END = env[TICKER].size() - 2;
        for(unsigned int t = START; t <= END; t++) {
            std::vector<double> state = sample_state(env, t);
            unsigned int action = epsilon_greedy(state, EPS);
            
            std::vector<double> next_state = sample_state(env, t+1);
            std::vector<double> next_q = target.predict(next_state);

            double diff = (env[TICKER][t+1] - env[TICKER][t]) / env[TICKER][t];
            double observed_reward = (diff >= 0.00 ? action_space[action] : -action_space[action]);
            double optimal = observed_reward + GAMMA * *std::max_element(next_q.begin(), next_q.end());

            Memory memory(state, action, optimal);
            replay.push_back(memory);

            benchmark *= 1.00 + diff;
            model *= 1.00 + diff * action_space[action];

            for(unsigned int i = 1; i < env[TICKER].size(); i++)
                out << state[OBS*i-1] << ",";
            out << action << "," << benchmark << "," << model << "\n";
            std::cout << "T=" << t << " @ " << ticker << " ACTION=" << action << " ";
            std::cout << "-> OBS=" << observed_reward << " OPT=" << optimal << " ";
            std::cout << "BENCH=" << benchmark << " " << "MODEL=" << model << "\n";

            if(replay.size() == CAPACITY) {
                std::vector<unsigned int> index(CAPACITY, 0);
                std::iota(index.begin(), index.end(), 0);
                std::shuffle(index.begin(), index.end(), *seed);
                index.erase(index.begin() + BATCH_SIZE, index.end());

                for(unsigned int k: index)
                    sgd(replay[k], ALPHA, LAMBDA);
                
                replay.erase(replay.begin());
            }
        } sync();

        out.close();
        std::system(("./python/log.py " + ticker + "-train").c_str());
    }
    save();
}

void Quant::sgd(Memory &memory, double alpha, double LAMBDA) {
    std::vector<double> q = agent.predict(memory.state);
    for(int l = agent.num_of_layers() - 1; l >= 0; l--) {
        double partial_gradient = 0.00, gradient = 0.00;
        for(unsigned int n = 0; n < agent.layer(l)->out_features(); n++) {
            if(l == agent.num_of_layers() - 1 && n != memory.action) continue;
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

                    gradient += LAMBDA * agent.layer(l)->node(n)->weight(i);

                    double updated_weight = agent.layer(l)->node(n)->weight(i) - alpha * gradient;
                    agent.layer(l)->node(n)->set_weight(i, updated_weight);
                }
            }
        }
    }
}

void Quant::test() {
    std::vector<int> action_count = {0, 0, 0};
    for(std::string &ticker: tickers) {
        std::ofstream out("./res/log");
        out << "X,SPY,IEF,GSG,EUR=X,action,benchmark,model\n";
        double benchmark = 1.00, model = 1.00;

        std::vector<std::vector<double>> env = generate_environment(ticker);
        unsigned int START = OBS - 1, END = env[TICKER].size() - 1;
        for(unsigned int t = START; t <= END; t++) {
            std::vector<double> state = sample_state(env, t);
            unsigned int action = greedy(state);

            if(t == END) { action_count[action]++; continue; }

            double diff = (env[TICKER][t+1] - env[TICKER][t]) / env[TICKER][t];
            benchmark *= 1.00 + diff;
            model *= 1.00 + diff * action_space[action];

            for(unsigned int i = 1; i < env[TICKER].size(); i++)
                out << state[OBS*i-1] << ",";
            out << action << "," << benchmark << "," << model << "\n";
            std::cout << "T=" << t << " @ " << ticker << " ACTION=" << action << " ";
            std::cout << "DIFF=" << diff << " BENCH=" << benchmark << " MODEL=" << model << "\n";
        }

        out.close();
        std::system(("./python/log.py " + ticker + "-test").c_str());
        std::system(("./python/stats.py push " + ticker).c_str());
        std::system(("./python/analytics.py " + ticker).c_str());
    }

    std::system("./python/stats.py summary");

    std::cout << "\n";
    std::cout << "ACTION (0) - SHORT: " << action_count[0] << "\n";
    std::cout << "ACTION (1) - IDLE : " << action_count[1] << "\n";
    std::cout << "ACTION (2) - LONG : " << action_count[2] << "\n";
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