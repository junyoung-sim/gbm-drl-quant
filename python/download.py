#!/usr/bin/env python3

import sys
import certifi
import warnings
import json, csv
import pandas as pd
from urllib.request import urlopen
import platform

warnings.filterwarnings("ignore")

dsp = "/"
if platform.system() == "Windows":
    dsp = "\\"

def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

def main():
    df = pd.DataFrame()
    tickers = sys.argv[1:]
    apikey = open(f".{dsp}apikey", "r").readline()
    for ticker in tickers:
        # download historical data of each ticker
        url = "https://financialmodelingprep.com/api/v3/historical-price-full"
        if ticker.endswith("=X"):
            url += "/{}?apikey={}&from=2000-01-01" .format("USD" + ticker[:-2], apikey)
        else:
            url += "/{}?apikey={}&from=2000-01-01" .format(ticker, apikey)
        
        json_data = get_jsonparsed_data(url)
        out = open(f".{dsp}data{dsp}{ticker}.csv", "w")
        csv_writer = csv.writer(out)

        header = True
        for data in json_data["historical"]:
            if header:
                csv_writer.writerow(data.keys())
                header = False
            csv_writer.writerow(data.values())
        out.close()

        # merge historical data of all tickers into one csv file
        hist = pd.read_csv(f".{dsp}data{dsp}{ticker}.csv").loc[::-1][["date", "adjClose"]]
        hist = hist.rename(columns={"adjClose": ticker})
        if df.empty:
            df = hist
        else:
            df = df.merge(hist, on="date")
        
    df = df.drop(columns=["date"])
    df.to_csv(f".{dsp}data{dsp}merge.csv", index=False)

if __name__ == "__main__":
    # ./python/download.py <tickers>
    main()