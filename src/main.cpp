#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <random>

#include "../lib/gbm.hpp"

/*
    ./exec MODE TICKER CHECKPOINT (TRAIN) (TEST)
*/

std::string mode, ticker, checkpoint; // basic command line arguments

double train, test; // data partition for build

std::vector<std::string> indicators = {"SPY", "IEF"}; // stock, bond

std::vector<std::vector<double>> env; // ticker environment (dataset)

std::default_random_engine seed; // random seed

void boot(int argc, char *argv[]) {
    // read command line arguments
    mode       = argv[1];
    ticker     = argv[2];
    checkpoint = argv[3];

    if(mode == "build") {
        train = std::stod(argv[4]);
        test  = std::stod(argv[5]);
    }

    // download historical data of ticker and indicators
    std::cout << "\nDownloading...\n\n";
    download(ticker);
    for(std::string &ind: indicators)
        download(ind);

    env = historical_data(ticker, indicators); // create ticker environment (dataset)
    std::cout << "\n";
}

int main(int argc, char *argv[])
{
    boot(argc, argv);

    std::cout << geometric_brownian_motion(env[1], 60, 100, seed, false);

    return 0;
}