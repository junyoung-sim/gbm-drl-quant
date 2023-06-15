#include <iostream>
#include <cstdlib>
#include <string>

#include "../lib/data.hpp"

std::string mode, ticker, checkpoint; // basic command line arguments

std::vector<std::string> indicators = {"SPY", "IEF"}; // stock, bond

double train, test; // data partition

void boot(int argc, char *argv[]) { // ./exec MODE TICKER CHECKPOINT (TRAIN) (TEST)
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
    
    
}

int main(int argc, char *argv[])
{
    boot(argc, argv);

    return 0;
}