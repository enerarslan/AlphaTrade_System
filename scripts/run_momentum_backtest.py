#!/usr/bin/env python3
"""Momentum-based backtest using trend following."""
import pandas as pd
import numpy as np
import time
import warnings
from pathlib import Path
import json

warnings.filterwarnings('ignore')

def compute_features(df):
    """Compute momentum features."""
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_60'] = df['close'].rolling(60).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # ADX for trend strength
    high = df['high']
    low = df['low']
    close = df['close']
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    df['atr'] = tr.rolling(14).mean()

    plus_dm = (high - high.shift(1)).where((high - high.shift(1)) > (low.shift(1) - low), 0).clip(lower=0)
    minus_dm = (low.shift(1) - low).where((low.shift(1) - low) > (high - high.shift(1)), 0).clip(lower=0)

    plus_di = 100 * (plus_dm.rolling(14).mean() / df['atr'])
    minus_di = 100 * (minus_dm.rolling(14).mean() / df['atr'])
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['adx'] = dx.rolling(14).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    # Volatility
    df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252 * 26)

    return df


def main():
    print('='*80)
    print('BACKTESTING - Trend Following Momentum Strategy')
    print('='*80)

    # Load data
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META']  # High momentum stocks
    data = {}
    print('Loading data...')
    for sym in symbols:
        df = pd.read_csv(f'data/raw/{sym}_15min.csv', parse_dates=['timestamp'])
        df = df.set_index('timestamp').sort_index()
        df = df['2023-01-01':'2024-12-31']  # Recent bullish period
        data[sym] = compute_features(df)
        print(f'  {sym}: {len(df)} bars')

    # Backtest parameters
    initial_capital = 100000
    commission_rate = 0.0005  # 5 bps
    slippage_rate = 0.0002  # 2 bps
    max_positions = 3
    position_size_pct = 0.25

    print()
    print('Running backtest...')
    start_time = time.time()

    capital = initial_capital
    positions = {}
    equity_curve = []
    trades = []
    max_equity = initial_capital
    max_drawdown = 0

    # Get common dates
    common_dates = None
    for sym in data:
        if common_dates is None:
            common_dates = set(data[sym].index)
        else:
            common_dates = common_dates.intersection(set(data[sym].index))
    common_dates = sorted(list(common_dates))[100:]

    for date in common_dates:
        current_equity = capital
        for sym, pos in positions.items():
            price = data[sym].loc[date]['close']
            current_equity += pos['shares'] * price

        equity_curve.append(current_equity)

        if current_equity > max_equity:
            max_equity = current_equity
        dd = (max_equity - current_equity) / max_equity
        if dd > max_drawdown:
            max_drawdown = dd

        # Score symbols for entry
        scores = {}
        for sym in symbols:
            if sym in positions:
                continue

            row = data[sym].loc[date]
            if pd.isna(row['adx']) or pd.isna(row['macd']):
                continue

            adx = row['adx']
            macd = row['macd']
            macd_signal = row['macd_signal']
            plus_di = row['plus_di']
            minus_di = row['minus_di']
            sma_10 = row['sma_10']
            sma_30 = row['sma_30']
            sma_60 = row['sma_60']
            close = row['close']
            volatility = row['volatility']

            # Strong trend with momentum
            if (adx > 25 and
                plus_di > minus_di and
                macd > macd_signal and
                close > sma_10 > sma_30 and
                volatility < 0.6):  # Volatility filter
                score = adx + (plus_di - minus_di)
                scores[sym] = score

        # Open positions
        open_slots = max_positions - len(positions)
        if open_slots > 0 and scores:
            sorted_symbols = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            for sym in sorted_symbols[:open_slots]:
                price = data[sym].loc[date]['close']
                shares = int((capital * position_size_pct) / price)
                if shares > 0 and capital >= shares * price * 1.01:
                    cost = shares * price * (1 + commission_rate + slippage_rate)
                    capital -= cost
                    positions[sym] = {
                        'shares': shares,
                        'entry_price': price,
                        'stop_loss': price * 0.97,  # 3% stop
                        'bars_held': 0
                    }
                    trades.append({'type': 'BUY', 'pnl': 0})

        # Check exits
        to_close = []
        for sym, pos in positions.items():
            row = data[sym].loc[date]
            price = row['close']
            adx = row['adx']
            macd = row['macd']
            macd_signal = row['macd_signal']
            plus_di = row['plus_di']
            minus_di = row['minus_di']

            pos['bars_held'] += 1

            # Trail stop loss
            new_stop = price * 0.97
            if new_stop > pos['stop_loss']:
                pos['stop_loss'] = new_stop

            # Exit conditions
            exit_signal = False

            # Stop loss hit
            if price < pos['stop_loss']:
                exit_signal = True
            # Trend weakening
            elif adx < 20 or plus_di < minus_di:
                exit_signal = True
            # MACD crossover
            elif macd < macd_signal:
                exit_signal = True
            # Take profit at 15%
            elif (price - pos['entry_price']) / pos['entry_price'] > 0.15:
                exit_signal = True
            # Max holding period (5 days = 130 bars)
            elif pos['bars_held'] > 130:
                exit_signal = True

            if exit_signal:
                to_close.append(sym)

        for sym in to_close:
            pos = positions[sym]
            price = data[sym].loc[date]['close']
            proceeds = pos['shares'] * price * (1 - commission_rate - slippage_rate)
            capital += proceeds
            pnl = (price - pos['entry_price']) * pos['shares']
            trades.append({'type': 'SELL', 'pnl': pnl})
            del positions[sym]

    # Close remaining
    for sym, pos in list(positions.items()):
        price = data[sym].iloc[-1]['close']
        proceeds = pos['shares'] * price * (1 - commission_rate - slippage_rate)
        capital += proceeds
        pnl = (price - pos['entry_price']) * pos['shares']
        trades.append({'type': 'SELL', 'pnl': pnl})

    elapsed = time.time() - start_time
    print(f'Backtest completed in {elapsed:.2f}s')

    # Calculate metrics
    final_equity = capital
    total_return = (final_equity - initial_capital) / initial_capital

    equity_arr = np.array(equity_curve)
    returns = np.diff(equity_arr) / equity_arr[:-1]
    returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 26) if len(returns) > 0 and np.std(returns) > 0 else 0

    sell_trades = [t for t in trades if t['type'] == 'SELL']
    wins = len([t for t in sell_trades if t.get('pnl', 0) > 0])
    losses = len([t for t in sell_trades if t.get('pnl', 0) <= 0])
    win_rate = wins / len(sell_trades) if sell_trades else 0

    total_profit = sum([t.get('pnl', 0) for t in sell_trades if t.get('pnl', 0) > 0])
    total_loss = abs(sum([t.get('pnl', 0) for t in sell_trades if t.get('pnl', 0) < 0]))
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

    print()
    print('BACKTEST RESULTS')
    print('='*60)
    print(f'Initial Capital: ${initial_capital:,.2f}')
    print(f'Final Equity:    ${final_equity:,.2f}')
    print(f'Total Return:    {total_return*100:.2f}%')
    print(f'Sharpe Ratio:    {sharpe:.2f}')
    print(f'Max Drawdown:    {max_drawdown*100:.2f}%')
    print(f'Total Trades:    {len(sell_trades)}')
    print(f'Win Rate:        {win_rate*100:.1f}%')
    print(f'Profit Factor:   {profit_factor:.2f}')
    print()
    print('TARGET CHECK:')
    print(f'  Sharpe > 1.0:     {"PASS" if sharpe > 1.0 else "FAIL"} ({sharpe:.2f})')
    print(f'  Max DD < 15%:     {"PASS" if max_drawdown < 0.15 else "FAIL"} ({max_drawdown*100:.1f}%)')
    print(f'  Win Rate > 52%:   {"PASS" if win_rate > 0.52 else "FAIL"} ({win_rate*100:.1f}%)')
    print(f'  Profit Factor > 1.3: {"PASS" if profit_factor > 1.3 else "FAIL"} ({profit_factor:.2f})')

    # Save results
    results = {
        'strategy': 'Trend Following Momentum',
        'initial_capital': initial_capital,
        'final_equity': final_equity,
        'total_return_pct': total_return * 100,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_drawdown * 100,
        'total_trades': len(sell_trades),
        'win_rate_pct': win_rate * 100,
        'profit_factor': profit_factor,
        'wins': wins,
        'losses': losses,
    }

    Path('backtest_results').mkdir(exist_ok=True)
    with open('backtest_results/backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print('BACKTESTING: COMPLETED')
    return results


if __name__ == '__main__':
    main()
