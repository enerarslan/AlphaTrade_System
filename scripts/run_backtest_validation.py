#!/usr/bin/env python3
"""Multi-factor mean reversion backtest for validation."""
import pandas as pd
import numpy as np
import time
import warnings
from pathlib import Path
import json

warnings.filterwarnings('ignore')

def compute_features(df):
    """Compute trading features."""
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['std_20'] = df['close'].rolling(20).std()
    df['zscore'] = (df['close'] - df['sma_20']) / df['std_20']
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['bb_upper'] = df['sma_20'] + 2 * df['std_20']
    df['bb_lower'] = df['sma_20'] - 2 * df['std_20']
    df['bb_pctb'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['vol_ratio'] = df['atr'] / df['close']
    return df


def main():
    print('='*80)
    print('BACKTESTING - Multi-Factor Mean Reversion Strategy')
    print('='*80)

    # Load data
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'V', 'JNJ', 'HD', 'PG']
    data = {}
    print('Loading data...')
    for sym in symbols:
        df = pd.read_csv(f'data/raw/{sym}_15min.csv', parse_dates=['timestamp'])
        df = df.set_index('timestamp').sort_index()
        df = df['2022-01-01':'2024-06-30']
        data[sym] = compute_features(df)
    print(f'  Loaded {len(symbols)} symbols')

    # Backtest parameters
    initial_capital = 100000
    commission_rate = 0.001
    slippage_rate = 0.0003
    max_positions = 5
    position_size_pct = 0.15

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
    common_dates = sorted(list(common_dates))[200:]

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
            if pd.isna(row['zscore']) or pd.isna(row['rsi']):
                continue

            zscore = row['zscore']
            rsi = row['rsi']
            bb_pctb = row['bb_pctb']
            vol_ratio = row['vol_ratio']

            if zscore < -1.5 and rsi < 35 and bb_pctb < 0.1:
                score = -zscore + (35 - rsi) / 10
                if vol_ratio < 0.03:
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
                    positions[sym] = {'shares': shares, 'entry_price': price}
                    trades.append({'type': 'BUY', 'pnl': 0})

        # Check exits
        to_close = []
        for sym, pos in positions.items():
            row = data[sym].loc[date]
            price = row['close']
            zscore = row['zscore']
            rsi = row['rsi']
            pnl_pct = (price - pos['entry_price']) / pos['entry_price']

            if zscore > 0.5 or rsi > 55 or pnl_pct < -0.05:
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
        'strategy': 'Multi-Factor Mean Reversion',
        'initial_capital': initial_capital,
        'final_equity': final_equity,
        'total_return_pct': total_return * 100,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_drawdown * 100,
        'total_trades': len(sell_trades),
        'win_rate_pct': win_rate * 100,
        'profit_factor': profit_factor,
    }

    Path('backtest_results').mkdir(exist_ok=True)
    with open('backtest_results/backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print('BACKTESTING: COMPLETED')
    return results


if __name__ == '__main__':
    main()
