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
    
    print("Fetching historical price data...")
    try:
        # Fetch all data at once instead of chunks
        data = fetch_crypto_data(
            tickers,
            start=start_date - timedelta(days=30),  # Extra 30 days for initial calculation
            end=end_date,
            interval='1d'
        )
        
        # Convert to DataFrame if single ticker
        if isinstance(data, pd.Series):
            data = pd.DataFrame({tickers[0]: data})
            
        dates = data.index[data.index >= start_date]
        print(f"Successfully loaded {len(dates)} days of historical data")
    except Exception as e:
        print(f"Error fetching historical data: {str(e)}")
        return {'error': str(e)}

    print("\nStarting simulation...")
    total_days = len(dates)
    
    # Pre-calculate all allocations using vectorized operations
    print("Pre-calculating allocations...")
    window = 30  # 30-day lookback
    allocations = {}
    
    # Create a sliding window of returns
    for i in range(window, len(data)):
        if i % 20 == 0:
            print(f"Calculating allocations: {(i/len(data))*100:.1f}%")
        
        window_data = data.iloc[i-window:i]
        returns = calculate_returns(window_data)
        optimal_weights = optimize_portfolio(returns)
        allocations[data.index[i]] = dict(zip(tickers, optimal_weights * 100))  # Convert to percentages

    # Vectorize portfolio calculations
    print("\nRunning trading simulation...")
    for day_index, current_date in enumerate(dates):
        if day_index % 20 == 0:
            progress = (day_index + 1) / total_days * 100
            print(f"Progress: {progress:.1f}% ({day_index + 1}/{total_days} days)")
            
        try:
            # Get allocation for this date
            allocation = allocations.get(current_date)
            if not allocation:
                continue
                
            # Calculate current portfolio value using vectorized operations
            current_prices = data.loc[current_date]
            holdings_value = sum(current_holdings[ticker] * current_prices[ticker] for ticker in tickers)
            current_value = holdings_value + current_cash
            max_value = max(max_value, current_value)
            
            # Calculate target amounts
            target_amounts = {ticker: (alloc/100.0) * current_value for ticker, alloc in allocation.items()}
            
            # Prepare proposed trades for GPT analysis
            proposed_trades = []
            for ticker in tickers:
                current_price = current_prices[ticker]
                current_value = current_holdings[ticker] * current_price
                target_value = target_amounts.get(ticker, 0.0)
                trade_amount = target_value - current_value
                
                if abs(trade_amount) > 1.0:  # Minimum trade size
                    proposed_trades.append({
                        'ticker': ticker,
                        'type': 'buy' if trade_amount > 0 else 'sell',
                        'amount': abs(trade_amount),
                        'price': current_price
                    })
            
            # Get GPT analysis for trades
            if proposed_trades:
                analysis = analyze_with_gpt(current_prices, data, proposed_trades)
                if analysis['recommendations'] == 'reject':
                    print("GPT recommends against these trades. Skipping day.")
                    continue
            
            # Execute approved trades
            for trade in proposed_trades:
                fees = calculate_fees(trade['amount'])
                if trade['type'] == 'buy' and current_cash >= (trade['amount'] + fees):
                    shares = trade['amount'] / trade['price']
                    current_holdings[trade['ticker']] += shares
                    current_cash -= (trade['amount'] + fees)
                    trade_history.append({
                        'timestamp': current_date,
                        'ticker': trade['ticker'],
                        'type': 'buy',
                        'amount': trade['amount'],
                        'price': trade['price'],
                        'fees': fees
                    })
                elif trade['type'] == 'sell' and current_holdings[trade['ticker']] >= trade['amount'] / trade['price']:
                    shares = trade['amount'] / trade['price']
                    current_holdings[trade['ticker']] -= shares
                    current_cash += trade['amount'] - fees
                    trade_history.append({
                        'timestamp': current_date,
                        'ticker': trade['ticker'],
                        'type': 'sell',
                        'amount': trade['amount'],
                        'price': trade['price'],
                        'fees': fees
                    })
            
            # Record daily portfolio value
            portfolio_value = sum(current_holdings[ticker] * current_prices[ticker] for ticker in tickers) + current_cash
            portfolio_history.append({
                'date': current_date,
                'value': portfolio_value,
                'cash': current_cash
            })
            
            # Calculate daily return
            if len(portfolio_history) > 1:
                daily_return = (portfolio_value / portfolio_history[-2]['value']) - 1
                daily_returns.append(daily_return)
                
        except Exception as e:
            print(f"Warning: Error processing day {current_date.strftime('%Y-%m-%d')}: {str(e)}")
            continue
    
    print("\nSimulation completed. Calculating performance metrics...")
    
    # Calculate performance metrics
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

def analyze_with_gpt(current_prices: Dict[str, float], historical_data: pd.DataFrame, proposed_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Use GPT to analyze market conditions and validate trades."""
    print("\n" + "="*50)
    print("CALLING GPT FOR ANALYSIS")
    print("="*50)
    
    try:
        # Prepare market context with more data
        market_summary = []
        for ticker, price in current_prices.items():
            prev_price = historical_data[ticker].iloc[-2]
            print(f"prev_price: {prev_price}    price: {price}")
            daily_change = ((price - prev_price) / prev_price) * 100
            print(f"daily_change: {daily_change}")
            week_change = ((price - historical_data[ticker].iloc[-7]) / historical_data[ticker].iloc[-7]) * 100
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
        analysis = analyze_with_gpt(current_prices, historical_data, proposed_trades)
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
