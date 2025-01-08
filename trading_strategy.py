# trading_strategy.py

import sys
import argparse
import pandas as pd
import numpy as np
import yfinance as yf # type: ignore
from scipy.optimize import minimize # type: ignore  
from datetime import datetime
import time

def fetch_crypto_data(tickers, start, end, interval="1d"):
    """
    Download data from Yahoo for the exact date range and interval provided.
    Includes retry logic and better error handling.
    """
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Add a small buffer to the date range to ensure we get data
            data = yf.download(
                tickers,
                start=start,
                end=end,
                interval=interval,
                progress=False,  # Disable progress bar to reduce noise
                ignore_tz=True  # Ignore timezone issues
            )
            
            if data.empty:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise ValueError(f"No data returned from {start} to {end} after {max_retries} attempts.")
            
            # If multiple tickers, data.columns is a MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                close_data = data['Close']
            else:
                # Single ticker case
                close_data = data['Adj Close'] if 'Adj Close' in data else data['Close']
            
            # Verify we have data for all tickers
            if isinstance(close_data, pd.DataFrame):
                missing_tickers = [ticker for ticker in tickers if ticker not in close_data.columns]
            else:
                missing_tickers = [] if not close_data.empty else tickers
                
            if missing_tickers:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise ValueError(f"Missing data for tickers: {missing_tickers}")
                
            return close_data
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise ValueError(f"Error fetching data: {str(e)}")
    
    raise ValueError("Failed to fetch data after all retries")

def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns from price data."""
    return data.pct_change().dropna()

def portfolio_volatility(weights: np.ndarray, returns: pd.DataFrame) -> float:
    """
    Objective function for Markowitz optimization:
    Minimize annualized portfolio volatility given the daily returns.
    """
    annual_cov = returns.cov() * 252  # Annualize daily covariance (252 trading days)
    return float(np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights))))

def optimize_portfolio(returns: pd.DataFrame) -> np.ndarray:
    """Optimize portfolio weights using Markowitz optimization."""
    num_assets = len(returns.columns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    try:
        result = minimize(
            portfolio_volatility,
            x0=np.array([1.0/num_assets] * num_assets),
            args=(returns,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return result.x
    except Exception as e:
        print(f"Optimization failed: {e}")
        return np.array([1.0/num_assets] * num_assets)

def get_real_time_data(tickers):
    """
    Optionally adjust weights using 'real-time' info from yfinance.
    This is a simplistic example that fetches volume and market cap.
    """
    real_time_data = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            real_time_data[ticker] = {
                'price': info.get('regularMarketPrice', 0.0),
                'volume': info.get('volume24Hr', 0.0),
                'market_cap': info.get('marketCap', 0.0)
            }
        except:
            real_time_data[ticker] = {
                'price': 0.0,
                'volume': 0.0,
                'market_cap': 0.0
            }
    return real_time_data

def adjust_weights(optimal_weights, real_time_data, tickers):
    """
    Combines three factors:
    1. Markowitz optimal weights (60% influence)
    2. Trading volume (20% influence)
    3. Market cap (20% influence)
    """
    # Get volume-based weighting
    volume_factor = np.array([
        real_time_data[t]['volume'] if real_time_data[t]['volume'] else 1.0
        for t in tickers
    ], dtype=np.float64)

    # Get market cap weighting
    market_cap_factor = np.array([
        real_time_data[t]['market_cap'] if real_time_data[t]['market_cap'] else 1.0
        for t in tickers
    ], dtype=np.float64)

    # Combine all factors
    adjusted = (optimal_weights * 0.6 +          # Markowitz optimization
               volume_factor * 0.2 +             # Trading volume
               market_cap_factor * 0.2)          # Market capitalization
    adjusted /= np.sum(adjusted)
    return adjusted

if __name__ == "__main__":
    print('starting trading_strategy.py')
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument('--end', required=True, help="End date in YYYY-MM-DD format")
    args = parser.parse_args()

    tickers = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'ADA-USD']
    start_date = args.start
    end_date   = args.end

    print(f"Fetching data from {start_date} to {end_date}")
    
    # 1) Pull data for this exact range
    data = fetch_crypto_data(tickers, start=start_date, end=end_date, interval="1d")
    returns = calculate_returns(data)

    print("Optimizing portfolio...")
    
    # 2) Run Markowitz optimization to minimize volatility
    optimal_weights = optimize_portfolio(returns)

    print("Getting real-time data...")
    
    # 3) Optionally adjust with 'real-time' factors (volume, market cap)
    real_time_info  = get_real_time_data(tickers)
    final_weights   = adjust_weights(optimal_weights, real_time_info, tickers)

    print("Final allocations:")
    # 4) Print allocation lines so trade_executor.py can parse them
    for ticker, weight in zip(tickers, final_weights):
        # Multiply by 100 to get percentage
        allocation_line = f"{ticker}:{weight * 100:.2f}"
        print(allocation_line)
        sys.stdout.flush()  # Ensure output is flushed
