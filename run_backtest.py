from trade_executor import backtest_strategy
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Any

def main():
    """Run backtest simulation."""
    start_date = datetime(2024, 12, 1)
    end_date = datetime(2025, 1, 7)
    initial_capital = 100000.0
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 'ADA-USD']
    
    print(f"\nRunning backtest from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Initial capital: ${initial_capital:,.2f}")
    print(f"Tickers: {', '.join(tickers)}\n")
    
    results = backtest_strategy(start_date, end_date, initial_capital, tickers)
    
    if 'error' in results:
        print(f"Error running backtest: {results['error']}")
        return
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    
    print(f"\nFinal Portfolio Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {results['max_drawdown']*100:.2f}%")
    
    # Calculate annualized metrics
    days = (end_date - start_date).days
    annualized_return = ((1 + results['total_return']) ** (365/days) - 1) * 100
    annualized_volatility = np.std([t['value']/results['portfolio_history'][i-1]['value'] - 1 
                                  for i, t in enumerate(results['portfolio_history'][1:])]) * np.sqrt(252) * 100
    
    print(f"Annualized Return: {annualized_return:.2f}%")
    print(f"Annualized Volatility: {annualized_volatility:.2f}%")
    
    # Trade statistics
    buy_trades = [t for t in results['trade_history'] if t['type'] == 'buy']
    sell_trades = [t for t in results['trade_history'] if t['type'] == 'sell']
    total_fees = sum(t['fees'] for t in results['trade_history'])
    
    print(f"\nTotal Trades: {len(results['trade_history'])}")
    print(f"Buy Trades: {len(buy_trades)}")
    print(f"Sell Trades: {len(sell_trades)}")
    print(f"Total Fees Paid: ${total_fees:,.2f}")
    
    # Plot results
    plot_results(results['portfolio_history'])

def plot_results(portfolio_history: List[Dict[str, Any]]):
    """Plot portfolio value over time."""
    df = pd.DataFrame(portfolio_history)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Calculate daily returns
    df['daily_return'] = df['value'].pct_change()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
    
    # Plot portfolio value
    ax1.plot(df.index, df['value'], color='blue', linewidth=2)
    ax1.set_title('Portfolio Value Over Time', fontsize=12, pad=10)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot daily returns
    ax2.plot(df.index, df['daily_return'] * 100, color='green', alpha=0.6)
    ax2.set_title('Daily Returns', fontsize=12, pad=10)
    ax2.set_ylabel('Daily Return (%)', fontsize=10)
    ax2.set_xlabel('Date', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Format dates on x-axis
    plt.gcf().autofmt_xdate()
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
