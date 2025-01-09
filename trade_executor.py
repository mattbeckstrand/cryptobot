# trade_executor.py

import json
import openai
from typing import Any, Dict, List, Union, TypedDict, Optional, cast
from coinbase.wallet.client import Client  # type: ignore
import yfinance as yf  # type: ignore
import time
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from datetime import datetime, timedelta
from config import COINBASE_API_KEY, COINBASE_API_SECRET, OPENAI_API_KEY
from trading_strategy import calculate_returns, optimize_portfolio, fetch_crypto_data

# Risk management parameters
MAX_DAILY_LOSS = 0.02  # 2% maximum daily loss
STOP_LOSS_PERCENTAGE = 0.01  # 1% per trade
MAX_POSITION_SIZE = 0.20  # No single crypto > 20% of portfolio

# Initialize tracking variables
trade_history: List[Dict[str, Any]] = []

# Initialize clients
openai.api_key = OPENAI_API_KEY

try:
    client = Client(COINBASE_API_KEY, COINBASE_API_SECRET)
except Exception as e:
    print(f"Error initializing Coinbase client: {e}")
    client = None

def calculate_fees(trade_amount: float) -> float:
    """Estimate Coinbase fees."""
    return trade_amount * 0.006  # 0.6% fee

class PortfolioEntry(TypedDict):
    date: datetime
    value: float
    cash: float

def backtest_strategy(start_date: datetime, end_date: datetime, initial_capital: float, tickers: List[str]) -> Dict[str, Any]:
    """Run backtest simulation for the given period."""
    portfolio_history: List[PortfolioEntry] = []
    trade_history: List[Dict[str, Any]] = []
    current_cash = initial_capital
    current_holdings = {ticker: 0.0 for ticker in tickers}
    daily_returns = []
    max_value = initial_capital
    
    try:
        # Fetch all data at once
        data = fetch_crypto_data(
            tickers,
            start=start_date - timedelta(days=30),
            end=end_date,
            interval='1d'
        )
        
        if isinstance(data, pd.Series):
            data = pd.DataFrame({tickers[0]: data})
            
        dates = data.index[data.index >= start_date]
        
    except Exception as e:
        return {'error': str(e)}

    total_days = len(dates)
    window = 30
    allocations = {}
    
    # Pre-calculate allocations
    for i in range(window, len(data)):
        progress = (i / len(data)) * 40  # First 40% for allocations
        print(f"Backtesting: {progress:.1f}%", end="\r")
        
        window_data = data.iloc[i-window:i]
        returns = calculate_returns(window_data)
        optimal_weights = optimize_portfolio(returns)
        allocations[data.index[i]] = dict(zip(tickers, optimal_weights * 100))

    # Run simulation
    for day_index, current_date in enumerate(dates):
        progress = 40 + ((day_index + 1) / total_days * 60)  # Remaining 60% for simulation
        print(f"Backtesting: {progress:.1f}%", end="\r")
            
        try:
            allocation = allocations.get(current_date)
            if not allocation:
                continue
                
            current_prices = data.loc[current_date]
            holdings_value = sum(current_holdings[ticker] * current_prices[ticker] for ticker in tickers)
            current_value = holdings_value + current_cash
            max_value = max(max_value, current_value)
            
            target_amounts = {ticker: (alloc/100.0) * current_value for ticker, alloc in allocation.items()}
            
            # Execute trades
            for ticker in tickers:
                current_price = current_prices[ticker]
                current_value = current_holdings[ticker] * current_price
                target_value = target_amounts.get(ticker, 0.0)
                trade_amount = target_value - current_value
                
                if abs(trade_amount) > 1.0:
                    fees = calculate_fees(abs(trade_amount))
                    if trade_amount > 0 and current_cash >= (trade_amount + fees):
                        shares = trade_amount / current_price
                        current_holdings[ticker] += shares
                        current_cash -= (trade_amount + fees)
                        trade_history.append({
                            'timestamp': current_date,
                            'ticker': ticker,
                            'type': 'buy',
                            'amount': trade_amount,
                            'price': current_price,
                            'fees': fees
                        })
                    elif trade_amount < 0 and current_holdings[ticker] >= abs(trade_amount) / current_price:
                        shares = abs(trade_amount) / current_price
                        current_holdings[ticker] -= shares
                        current_cash += abs(trade_amount) - fees
                        trade_history.append({
                            'timestamp': current_date,
                            'ticker': ticker,
                            'type': 'sell',
                            'amount': abs(trade_amount),
                            'price': current_price,
                            'fees': fees
                        })
            
            portfolio_value = sum(current_holdings[ticker] * current_prices[ticker] for ticker in tickers) + current_cash
            portfolio_history.append({
                'date': current_date,
                'value': portfolio_value,
                'cash': current_cash
            })
            
            if len(portfolio_history) > 1:
                daily_return = (portfolio_value / portfolio_history[-2]['value']) - 1
                daily_returns.append(daily_return)
                
        except Exception as e:
            continue
    
    if portfolio_history:
        final_value = portfolio_history[-1]['value']
        total_return = (final_value / initial_capital) - 1
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if daily_returns else 0
        max_drawdown = (max_value - min(h['value'] for h in portfolio_history)) / max_value
    else:
        final_value = initial_capital
        total_return = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'trade_history': trade_history,
        'portfolio_history': portfolio_history
    }

def calculate_price_changes(historical_data: pd.DataFrame, current_date: datetime, ticker: str) -> tuple[float, float]:
    """
    Calculate 24h and 7d price changes using proper date handling.
    
    Args:
        historical_data: DataFrame with DateTimeIndex and price data
        current_date: The date for which to calculate changes
        ticker: The ticker symbol to analyze
    
    Returns:
        tuple of (24h_change_pct, 7d_change_pct)
    """
    try:
        # Get current price
        current_price = float(historical_data.loc[current_date][ticker])
        
        # Calculate 24h change
        one_day_ago = current_date - timedelta(days=1)
        prev_day_data = historical_data[historical_data.index <= one_day_ago].iloc[-1]
        daily_change = ((current_price - float(prev_day_data[ticker])) / float(prev_day_data[ticker])) * 100
        
        # Calculate 7d change
        seven_days_ago = current_date - timedelta(days=7)
        week_ago_data = historical_data[historical_data.index <= seven_days_ago].iloc[-1]
        week_change = ((current_price - float(week_ago_data[ticker])) / float(week_ago_data[ticker])) * 100
        
        return daily_change, week_change
        
    except Exception as e:
        print(f"Error calculating changes for {ticker}: {e}")
        return 0.0, 0.0

def analyze_with_gpt(current_prices: Dict[str, float], historical_data: pd.DataFrame, proposed_trades: List[Dict[str, Any]], current_date: datetime) -> Dict[str, Any]:
    """Use GPT to analyze market conditions and validate trades."""
    print("\n" + "="*50)
    print("CALLING GPT FOR ANALYSIS")
    print("="*50)
    
    try:
        # Prepare market context with more accurate data
        market_summary = []
        for ticker, price in current_prices.items():
            daily_change, week_change = calculate_price_changes(historical_data, current_date, ticker)
            market_summary.append(
                f"{ticker}: ${price:.2f} (24h: {daily_change:+.2f}%, 7d: {week_change:+.2f}%)"
            )
        
        print("\nMarket Summary:")
        for summary in market_summary:
            print(summary)

        # Prepare trade context
        trade_summary = []
        for trade in proposed_trades:
            trade_summary.append(
                f"Proposed {trade['type']} {trade['ticker']} "
                f"Amount: ${trade['amount']:.2f} at ${trade['price']:.2f}"
            )
        print("\nProposed Trades:")
        for trade_str in trade_summary:
            print(trade_str)

        print("\nRequesting GPT analysis...")

        # Enhanced prompt for more detailed analysis
        prompt = f"""As a crypto trading AI, analyze these market conditions for the specified date and time period and proposed trades:

Market Summary:
{chr(10).join(market_summary)}

Proposed Trades:
{chr(10).join(trade_summary)}

Analyze the market conditions and proposed trades. Consider:
1. Market trends and momentum (short and medium term, based on the date and time period, and the proposed trades)
2. Risk factors (volatility, market sentiment)
3. Trade timing and size optimization
4. Portfolio balance and diversification
5. Recent price movements and volume

Please also consider that the proposed trades are for the specified date and time period and come from a trading strategy that has been optimized through the markowitz optimization process.

Any analysis should be strictly based on where analysis is for the date and time of the market data passed 

Provide your response in JSON format with the following structure:
{{
    "recommendations": "approve" | "modify" | "reject",
    "risk_level": "low" | "medium" | "high",
    "insights": "your strategic insights here",
    "trade_adjustments": [
        {{
            "ticker": "ticker symbol",
            "action": "increase" | "decrease" | "hold",
            "adjustment_factor": 0.0 to 2.0  // 1.0 means no change, 0.5 means halve the amount, etc.
        }}
    ],
    "confidence_score": 0.0 to 1.0,
    "market_outlook": "bullish" | "neutral" | "bearish"
}}"""

        # Get GPT's analysis
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": "You are an expert crypto trading AI analyst. Focus on risk management and provide specific, actionable recommendations. Please also don't be so bearish and risk averse that there is no opportunity to make money. I want you to partly gamble on opportunitites hwere you might see strong potential for profit."
            }, {
                "role": "user",
                "content": prompt
            }],
            temperature=0.3
        )
        
        print("\nGPT Response Received!")
        
        # Parse GPT's response
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Empty response from GPT")
            
        try:
            # Clean the response: remove markdown code blocks if present
            cleaned_content = content
            if "```json" in cleaned_content:
                cleaned_content = cleaned_content.split("```json")[1]
            if "```" in cleaned_content:
                cleaned_content = cleaned_content.split("```")[0]
            
            analysis = json.loads(cleaned_content.strip())
            
            print("\nGPT Analysis:")
            print(f"Recommendation: {analysis['recommendations']}")
            print(f"Risk Level: {analysis['risk_level']}")
            print(f"Market Outlook: {analysis.get('market_outlook', 'unknown')}")
            print(f"Confidence Score: {analysis.get('confidence_score', 0.0):.2f}")
            print(f"Insights: {analysis['insights']}")
            
            if 'trade_adjustments' in analysis:
                print("\nTrade Adjustments:")
                for adj in analysis['trade_adjustments']:
                    print(f"{adj['ticker']}: {adj['action']} (factor: {adj['adjustment_factor']:.2f})")
            
            print("="*50 + "\n")
            
            return analysis
        except json.JSONDecodeError as e:
            print(f"Error parsing GPT response: {e}")
            print("Raw response:", content)
            raise

    except Exception as e:
        print(f"\nError in GPT analysis: {e}")
        print("Proceeding with quantitative signals only")
        print("="*50 + "\n")
        return {
            "recommendations": "proceed",
            "risk_level": "unknown",
            "insights": "GPT analysis failed, proceeding with quantitative signals only",
            "trade_adjustments": [],
            "confidence_score": 0.0,
            "market_outlook": "neutral"
        }

def execute_trades(
    current_portfolio: Dict[str, float],
    target_amounts: Dict[str, float],
    simulation: bool = False,
    current_prices: Optional[Dict[str, float]] = None,
    historical_data: Optional[pd.DataFrame] = None
) -> None:
    """Execute trades on Coinbase or in simulation."""
    if not simulation and not client:
        print("Coinbase client not initialized. Cannot execute live trades.")
        return

    # Prepare proposed trades
    proposed_trades = []
    for crypto, target in target_amounts.items():
        current_value = current_portfolio.get(crypto, 0.0)
        trade_amount = target - current_value
        
        if abs(trade_amount) <= 1.0:  # Skip small trades
            continue

        price = current_prices.get(crypto, 0.0) if current_prices else 0.0
        proposed_trades.append({
            'ticker': crypto,
            'type': 'buy' if trade_amount > 0 else 'sell',
            'amount': abs(trade_amount),
            'price': price
        })

    # Get GPT analysis if we have market data
    if current_prices and historical_data and proposed_trades:
        analysis = analyze_with_gpt(current_prices, historical_data, proposed_trades, datetime.now())
        print("\nGPT Analysis:")
        print(f"Risk Level: {analysis['risk_level']}")
        print(f"Insights: {analysis['insights']}")
        
        if analysis['recommendations'] == 'reject':
            print("GPT recommends against these trades. Skipping execution.")
            return
    
    # Execute approved trades
    for trade in proposed_trades:
        try:
            if simulation:
                print(f"Simulation: {'Buy' if trade['type'] == 'buy' else 'Sell'} ${trade['amount']:.2f} of {trade['ticker']}")
                continue

            if trade['type'] == 'buy':
                client.buy(currency_pair=trade['ticker'], amount=str(trade['amount']), currency='USD')
                print(f"Bought ${trade['amount']:.2f} of {trade['ticker']}")
            else:
                client.sell(currency_pair=trade['ticker'], amount=str(trade['amount']), currency='USD')
                print(f"Sold ${trade['amount']:.2f} of {trade['ticker']}")

            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"Error executing trade for {trade['ticker']}: {e}")

def get_current_portfolio() -> Dict[str, float]:
    """Get current holdings from Coinbase."""
    if not client:
        return {}
        
    try:
        accounts = client.get_accounts()['data']
        return {
            f"{account['currency']}-USD": float(account['balance']['amount'])
            for account in accounts
            if float(account['balance']['amount']) > 0.0
        }
    except Exception as e:
        print(f"Error retrieving portfolio: {e}")
        return {}

def main() -> None:
    """Main function for live trading."""
    tickers = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'ADA-USD']
    
    # Get current portfolio
    current_portfolio = get_current_portfolio()
    if not current_portfolio:
        print("Error: Cannot retrieve current portfolio.")
        return

    # Get historical data and calculate allocation
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    data = fetch_crypto_data(tickers, start=start_date, end=end_date)
    returns = calculate_returns(data)
    weights = optimize_portfolio(returns)
    allocation = dict(zip(tickers, weights * 100))

    # Get current prices
    current_prices = {ticker: data[ticker].iloc[-1] for ticker in tickers}

    # Calculate target amounts
    total_value = sum(current_portfolio.values())
    target_amounts = {ticker: (weight/100.0) * total_value for ticker, weight in allocation.items()}

    # Execute trades with GPT analysis
    execute_trades(
        current_portfolio=current_portfolio,
        target_amounts=target_amounts,
        current_prices=current_prices,
        historical_data=data
    )
    print("Portfolio rebalancing complete.")

if __name__ == "__main__":
    main()
