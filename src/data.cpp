#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>

#include "../lib/data.hpp"

std::vector<std::vector<double>> read_csv(std::string path) {
    std::vector<std::vector<double>> dat;
    std::ifstream file(path);

    if(!file.is_open())
        return dat;
    
    std::string line, val;
    file >> line;
    unsigned int columns = 1;
    for(char &ch: line)
        columns += (ch == ',');
    
    dat.resize(columns, std::vector<double>());
    while(file >> line) {
        for(unsigned int col = 0; col < columns; col++) {
            double val = std::stod(line.substr(0, line.find(",")));
            dat[col].push_back(val);

            line = line.substr(line.find(",") + 1);
        }
    }

    file.close();
    return dat;
}

double mean(std::vector<double> &dat) {
    double sum = 0.00;
    for(double &x: dat)
        sum += x;
    return sum / dat.size();
}

double stdev(std::vector<double> &dat) {
    double mu = mean(dat);
    double rss = 0.00;
    for(double &x: dat)
        rss += pow(x - mu, 2);
    return sqrt(rss / dat.size());
}

void standardize(std::vector<double> &dat) {
    double mu = mean(dat);
    double sigma = stdev(dat);
    for(double &x: dat)
        x = (x - mu) / sigma;
}

void normal(std::vector<std::vector<double>> &mat, std::default_random_engine &seed) {
    std::normal_distribution<double> std_normal(0.0, 1.0);
    for(int i = 0; i < mat.size(); i++)
        for(int j = 0; j < mat[i].size(); j++)
            mat[i][j] = std_normal(seed);
}

void cumsum(std::vector<std::vector<double>> &mat) {
    for(int i = 0; i < mat.size(); i++)
        for(int j = 1; j < mat[i].size(); j++)
            mat[i][j] += mat[i][j-1];
}

void fix_dsp(std::string &str) {
    #ifdef _WIN32
        std::replace(str.begin(), str.end(), '/', '\\');
    #endif
}