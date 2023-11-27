#include <cstdlib>
#include <string>
#include <vector>

#include "../lib/quant.hpp"

std::vector<std::string> tickers;
std::vector<std::string> indicators = {"SPY", "IEF", "EUR=X", "CL=F"};

std::string mode;
std::string checkpoint;
std::default_random_engine seed(std::chrono::system_clock::now().time_since_epoch().count());

void boot(int argc, char *argv[]) {
    mode = argv[1];
    for(int i = 2; i < argc - 1; i++)
        tickers.push_back(argv[i]);
    checkpoint = argv[argc-1];

    std::string cmd = "rm ./data/* ./res/*";
    fix_dsp(cmd); std::system(cmd.c_str());

    std::cout << std::fixed;
    std::cout.precision(15);
}

int main(int argc, char *argv[])
{
    boot(argc, argv);

    Quant quant(tickers, indicators, seed, checkpoint);

    if(mode == "build") quant.build();
    if(mode == "test") quant.test();
    if(mode == "run") quant.run();

    return 0;
}