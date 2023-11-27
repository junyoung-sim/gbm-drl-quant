#ifndef __GBM_HPP_
#define __GBM_HPP_

#include <cstdlib>
#include <vector>
#include <random>

#include "../lib/data.hpp"

#define OBS 60
#define EXT 20
#define EPOCH 1000

std::vector<double> returns(std::vector<double> &raw);

void vscore(std::vector<double> raw, std::vector<double> *v, std::default_random_engine seed);

#endif