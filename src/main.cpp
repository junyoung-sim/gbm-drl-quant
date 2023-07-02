#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>

#include "../lib/quant.hpp"

/*
<STATE>                              
    X: Ticker-of-interest
    SPY: S&P 500
    IEF: US Treasury 7-10Y
    GSG: Commodities
    EUR=X: Europe/USD

    ./exec MODE (TRAIN) (TEST) <TICKERS> ./models/CHECKPOINT
*/

std::vector<std::string> tickers;
std::vector<std::string> indicators = {"SPY", "IEF", "GSG", "EUR=X"};

std::string mode;
std::string checkpoint;

Environment env; // (ticker, dataset)

void boot(int argc, char *argv[]) {
    mode = argv[1];
    for(int i = 2; i < argc - 1; i++)
        tickers.push_back(argv[i]);
    checkpoint = argv[argc-1];

    std::cout << "\nDownloading... (this may take a while)\n\n";
    for(std::string &ind: indicators) {
        download(ind);
        std::system(("./python/gbm.py " + ind).c_str()); // GBM simulation
    }
    for(std::string &ticker: tickers) {
        download(ticker);
        std::system(("./python/gbm.py " + ticker).c_str()); // GBM simulation
        env[ticker] = historical_data(ticker, indicators);
    }

    std::cout << "\n";
    std::cout << std::fixed;
    std::cout.precision(15);
}

int main(int argc, char *argv[])
{
    boot(argc, argv);

    Quant quant(checkpoint);

    if(mode == "build")
        quant.build(tickers, env);
    
    if(mode == "test")
        quant.test(tickers, env);

    return 0;
}