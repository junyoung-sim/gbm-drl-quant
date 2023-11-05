#ifndef __DATA_HPP_
#define __DATA_HPP_

#include <cstdlib>
#include <vector>
#include <string>

std::vector<std::vector<double>> read_csv(std::string path);

double mean(std::vector<double> &dat);
double stdev(std::vector<double> &dat);
void standardize(std::vector<double> &dat);

void normal(std::vector<std::vector<double>> &mat, std::default_random_engine &seed);
void cumsum(std::vector<std::vector<double>> &mat);

void fix_dsp(std::string &str);

#endif