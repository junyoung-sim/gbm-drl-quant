#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <map>
#include <fstream>

#include "../lib/quant.hpp"

/*
<STATE>                              <ACTION>
    X: Ticker-of-interest                0: Long
    SPY: S&P 500                         1: Idle
    DIA: Dow 30                          2: Short
    QQQ: NASDAQ 100
    VEA: Non-US Developed 50         <REWARD>
    FEZ: Euro 50                         Discrete (+/-1)
    AIA: Asia 50
    SPEM: Emerging Markets           <SPECS>
    SHY: US Treasury 1-3Y                X E S&P 500 (TOP 100)
    IEF: US Treasury 7-10Y               
    TLT: US Treasury 15-20Y
    VNQ: Real Estate
    XLY: Consumer Discretionary
    XLP: Consumer Staples                
    XLE: Energy
    XLF: Financial
    XLV: Health Care
    XLI: Industrial
    XLB: Materials
    XLU: Utilities

    ./exec MODE (TRAIN) (TEST) <TICKERS> CHECKPOINT (./models/*)
*/

std::vector<std::string> tickers;
std::vector<std::string> indicators = {"SPY", "DIA", "QQQ", "VEA", "FEZ",
                                       "AIA", "SPEM", "SHY", "IEF", "TLT",
                                       "VNQ", "XLY", "XLP", "XLE", "XLF",
                                       "XLV", "XLI", "XLB", "XLU"};
std::string mode;
std::string checkpoint;

double train, test;

std::map<std::string, std::vector<std::vector<double>>> env; // (ticker, dataframe)

void boot(int argc, char *argv[]) {
    mode = argv[1];
    if(mode == "build") {
        train = std::stod(argv[2]);
        test = std::stod(argv[3]);
    }
    for(int i = 4; i < argc - 1; i++)
        tickers.push_back(argv[i]);
    checkpoint = argv[argc-1];

    std::cout << "\nDownloading... (this may take a while)\n\n";
    //for(std::string &ind: indicators)
    //    download(ind);
    for(std::string &ticker: tickers) {
        //download(ticker);
        env[ticker] = historical_data(ticker, indicators);
    }

    std::cout << "\n";
}

int main(int argc, char *argv[])
{
    boot(argc, argv);

    Quant quant(checkpoint);

    return 0;
}