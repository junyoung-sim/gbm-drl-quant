#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>

void download(std::string &ticker) {
    std::system(("./python/download.py " + ticker).c_str());
}

std::vector<std::vector<double>> read_csv(std::string path) {
    std::vector<std::vector<double>> dat;
    std::ifstream file(path);

    if(file.is_open()) {
        std::string line, val;
        std::getline(file, line);

        unsigned int columns = 1;
        for(char &ch: line)
            columns += (ch == ',');

        dat.resize(columns, std::vector<double>());
        while(std::getline(file, line)) {
            for(unsigned int col = 0; col < columns; col++) {
                double val = std::stod(line.substr(0, line.find(",")));
                dat[col].push_back(val);

                line = line.substr(line.find(",") + 1);
            }
        }

        file.close();
    }

    return dat;
}

std::vector<std::vector<double>> historical_data(std::string ticker, std::vector<std::string> &indicators) {
    // columns={ticker, indicator_1, indicator_2, ... , indicator_n}
    std::string clean = "./python/clean.py " + ticker + " ";
    for(unsigned int i = 0; i < indicators.size(); i++) {
        clean += indicators[i];
        if(i < indicators.size() - 1)
            clean += " ";
    }
    std::system(clean.c_str());

    // each row has the historical data of the ticker and each indicator
    return read_csv("./data/cleaned.csv");
}

double mean(std::vector<double> &dat) {
    double sum = 0.00;
    for(double &x: dat)
        sum += x;
    return sum / dat.size();
}

double stdev(std::vector<double> &dat) {
    double s = 0.00;
    double m = mean(dat);
    for(double &x: dat)
        s += pow(x - m, 2);
    s /= dat.size() - 1;
    s = sqrt(s);
    return s;
}