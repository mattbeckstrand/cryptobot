from trade_executor import backtest_strategy
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def run_backtest():
    # Test parameters
    tickers = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'ADA-USD']
    start_date = datetime(2024, 8, 15)  # Just December 2023
    end_date = datetime(2024, 12, 31)
    initial_capital = 100000.0

    print("\n" + "="*50)
    print("Starting Crypto Trading Backtest")
    print("="*50)
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print("Assets:", ", ".join(tickers))
    print("="*50 + "\n")

    # Run backtest
    results = backtest_strategy(start_date, end_date, initial_capital, tickers)
    
    if 'error' in results:
        print(f"\nBacktest failed: {results['error']}")
        return

    # Convert portfolio history to DataFrame once for all calculations
    df = pd.DataFrame(results['portfolio_history'])
    trades = pd.DataFrame(results['trade_history'])
    
    # Calculate all metrics
    daily_returns = df['value'].pct_change().dropna()
    total_return = (df['value'].iloc[-1] / initial_capital) - 1
    annualized_return = ((1 + total_return) ** (252 / len(df)) - 1) * 100
    volatility = daily_returns.std() * np.sqrt(252) * 100
    sharpe_ratio = (np.mean(daily_returns) * 252) / (np.std(daily_returns) * np.sqrt(252))
    max_drawdown = ((df['value'].cummax() - df['value']) / df['value'].cummax()).max() * 100
    
    # Print final summary
    print("\n" + "="*50)
    print("BACKTEST SUMMARY")
    print("="*50)
    
    # Performance metrics
    print("\nPERFORMANCE METRICS:")
    print(f"{'Initial Capital:':<25} ${initial_capital:,.2f}")
    print(f"{'Final Portfolio Value:':<25} ${df['value'].iloc[-1]:,.2f}")
    print(f"{'Total Return:':<25} {total_return*100:,.2f}%")
    print(f"{'Annualized Return:':<25} {annualized_return:,.2f}%")
    print(f"{'Annualized Volatility:':<25} {volatility:,.2f}%")
    print(f"{'Sharpe Ratio:':<25} {sharpe_ratio:.2f}")
    print(f"{'Maximum Drawdown:':<25} {max_drawdown:.2f}%")
    
    if not trades.empty:
        # Trading statistics
        print("\nTRADING STATISTICS:")
        total_trades = len(trades)
        total_fees = trades['fees'].sum()
        buy_trades = (trades['type'] == 'buy').sum()
        sell_trades = total_trades - buy_trades
        avg_trade_size = trades['amount'].mean()
        
        print(f"{'Total Trades:':<25} {total_trades:,}")
        print(f"{'Buy Trades:':<25} {buy_trades:,}")
        print(f"{'Sell Trades:':<25} {sell_trades:,}")
        print(f"{'Average Trade Size:':<25} ${avg_trade_size:,.2f}")
        print(f"{'Total Fees Paid:':<25} ${total_fees:,.2f}")
        print(f"{'Fees % of Capital:':<25} {(total_fees/initial_capital)*100:.2f}%")
    
    print("\n" + "="*50)

    # Plot results
    plot_results(df)

def plot_results(df: pd.DataFrame):
    """Plot backtest results using vectorized operations."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot portfolio value and cash
    ax1.plot(df['date'], df['value'], label='Portfolio Value', color='blue')
    ax1.plot(df['date'], df['cash'], label='Cash', linestyle='--', color='green')
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value ($)')
    ax1.grid(True)
    ax1.legend()
    
    # Calculate and plot daily returns
    daily_returns = df['value'].pct_change() * 100
    ax2.plot(df['date'][1:], daily_returns[1:], label='Daily Returns', color='gray')
    ax2.set_title('Daily Returns')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Return (%)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_backtest()
