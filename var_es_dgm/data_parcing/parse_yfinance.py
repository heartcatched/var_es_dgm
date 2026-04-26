import os
import yfinance as yf
import pandas as pd


def download_stock_data(ticker, start_date, end_date):
    """
    Download historical stock data for a given ticker within a specified date range.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    start_date : str
        Start date in the format 'YYYY-MM-DD'.
    end_date : str
        End date in the format 'YYYY-MM-DD'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the stock data.
    """
    return yf.download(ticker, start=start_date, end=end_date)


def parce_data(tickers, start_date, end_date, output_path=None):
    """
    Download and save stock data for multiple tickers.

    Parameters
    ----------
    tickers : list
        List of stock ticker symbols.
    start_date : str
        Start date in the format 'YYYY-MM-DD'.
    end_date : str
        End date in the format 'YYYY-MM-DD'.
    output_path : str, optional
        Directory path to save the CSV files. Defaults to None.

    Returns
    -------
    None
    """
    stock_data = {}
    for ticker in tickers:
        try:
            # Download stock data and reset index
            stock_data[ticker] = download_stock_data(
                ticker, start_date, end_date
            ).reset_index()
            # Save data to CSV if output_path is provided
            if output_path:
                stock_data[ticker].to_csv(
                    output_path + ticker.replace(".", "_") + ".csv", index=False
                )
            print(ticker, stock_data[ticker]["Date"].min())
        except (yf.DownloadError, KeyError):
            # Handle download errors
            print(f"Error downloading data for {ticker}")


def take_trading_from_2005():
    """
    Filter stocks with trading data starting from 2005 and save the combined data to a CSV file.

    Returns
    -------
    None
    """
    PATH_TO_DATA = "data/stocks/"
    data_files = list(filter(lambda x: x[-4:] == ".csv", os.listdir(PATH_TO_DATA)))
    df = pd.DataFrame()
    for file_name in data_files:
        # Read CSV files and add Ticker column
        temp = pd.read_csv(PATH_TO_DATA + file_name, parse_dates=["Date"]).assign(
            Ticker=file_name[:-4]
        )
        try:
            # Concatenate data if trading started on or before 2005
            if temp["Date"].min().year <= 2005:
                df = pd.concat([df, temp])
        except AttributeError:
            continue

    # Print number of suitable stocks and list of companies
    print("Number of suitable stocks: {0}".format(df["Ticker"].nunique()))
    print("\nList of availiable companies:\n", df["Ticker"].unique())
    # Save the combined data to a CSV file
    df.to_csv("data/complete_stocks.csv", index=False)


if __name__ == "__main__":
    DATA_FOLDER = "data/"
    # Get the list of DJIA tickers (replace with your preferred source)
    tickers = (
        pd.read_excel(DATA_FOLDER + "SP-500-Companies-List.xlsx")
        .sort_values(by="Weight", ascending=False)
        .iloc[:100]  # take only 100 largest
        .loc[:, "Symbol"]
    )

    # Define start and end date (change as needed)
    start_date = "2005-01-01"
    end_date = "2026-03-12"
    # Download and save stock data
    parce_data(tickers, start_date, end_date, output_path=DATA_FOLDER + "stocks/")
    # Filter and save trading data from 2005
    take_trading_from_2005()
