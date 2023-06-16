#ifndef __GBM_HPP_
#define __GBM_HPP_

#include <cstdlib>
#include <vector>

#include "../lib/data.hpp"

double geometric_brownian_motion(std::vector<double> &dat, unsigned int N, unsigned int epoch, std::default_random_engine &seed, bool plot);

#endif