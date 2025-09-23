import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ValueStrategy import ValueStrategy
from MomentumStrategy import MomentumStrategy
from prepare_ticker_data import get_all_us_tickers, get_spy500_tickers, download_data, slice_data_package


class Backtester:
    def __init__(self, full_data_package, strategy_class, start_date, end_date,
                 initial_capital=100000, rebalance_freq='W', params={}, verbose=False):
        self.full_data_package = full_data_package
        self.strategy_class = strategy_class
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.params = params # Store the entire params dictionary

        self.cash = initial_capital
        self.portfolio = {}
        self.transactions = []
        self.portfolio_history = []

        self.sp500_set = set(get_spy500_tickers())
        
        self.all_trade_days = self.full_data_package['prices'].index
        self.decision_dates = self.all_trade_days[
            (self.all_trade_days >= self.start_date) &
            (self.all_trade_days <= self.end_date)
        ].to_series().resample(self.rebalance_freq).first()
        self.decision_dates.dropna(inplace=True)

        self.verbose = verbose

    def run(self):
        """执行回测"""
        print(f"\n🚀 开始执行回测...")
        print(f"时间范围: {self.start_date.date()} to {self.end_date.date()}")
        print(f"初始资金: ${self.initial_capital:,.2f}")
        print(f"调仓频率: 每月第一个交易日")

        for decision_date in tqdm(self.decision_dates, desc="回测进度"):
            sliced_pkg = slice_data_package(self.full_data_package, decision_date)

            strategy = self.strategy_class(
                sliced_pkg,
                sp500_tickers_set=self.sp500_set,
                params=self.params,
                verbose = self.verbose
            )
            strategy.set_portfolio(self.portfolio)
            
            buy_signals, sell_signals = strategy.run_full_strategy()
            
            self._execute_trades(decision_date, sell_signals, buy_signals)

            print(f"\n---------- Portfolio Status [Date: {decision_date.date()}] ----------")
            
            holdings_value = 0.0
            total_value = self.cash
            current_prices = self.full_data_package['prices'].loc[decision_date]

            if self.portfolio:
                for ticker, data in self.portfolio.items():
                    current_price = current_prices.get(ticker)
                    if pd.notna(current_price) and data.get('shares', 0) > 0:
                        holdings_value += data['shares'] * current_price
                total_value = holdings_value + self.cash
            
            total_return = (total_value / self.initial_capital) - 1

            print(f"Cash on Hand: ${self.cash:,.2f}")
            print(f"Holdings Value: ${holdings_value:,.2f}")

            if not self.portfolio or holdings_value == 0:
                print("Portfolio is currently empty.")
            else:
                print(f"Holdings ({len(self.portfolio)} stocks):")
                holdings_list = []
                for ticker, data in self.portfolio.items():
                    if data.get('shares', 0) > 0:
                        current_price = current_prices.get(ticker, 0)
                        market_value = data.get('shares', 0) * current_price
                        holdings_list.append({
                            'Ticker': ticker,
                            'Shares': data.get('shares'),
                            'Buy Price': data.get('buy_price'),
                            'Current Price': f"${current_price:,.2f}",
                            'Market Value': f"${market_value:,.2f}"
                        })
                if holdings_list:
                    holdings_df = pd.DataFrame(holdings_list)
                    print(holdings_df.to_string(index=False))

            print("-" * 25)
            print(f"Total Portfolio Value: ${total_value:,.2f}")
            print(f"Total Return: {total_return:.2%}")
            print("------------------------------------------------------------")

            self._record_portfolio_value(decision_date)

        print("\n✅ 回测执行完毕!")
        self.results = self.analyze_performance()
        self.plot_performance()

        if self.transactions:
            tx_df = pd.DataFrame(self.transactions, columns=['Date','Action','Ticker','Shares','Price','Reason'])
            print("\n--- 📑 交易记录 ---")
            print(tx_df.to_string(index=False))
        else:
            print("无交易记录。")
        return self.results

    def _execute_trades(self, trade_date, sell_signals, buy_signals):
        trade_prices = self.full_data_package['prices'].loc[trade_date]

        # --- Execute Sells ---
        for trade in sell_signals:
            ticker = trade['ticker']
            shares_to_sell = trade['shares']
            reason = trade['reason']  # Extract the reason

            if ticker in self.portfolio and ticker in trade_prices.index:
                sell_price = trade_prices[ticker]
                if pd.notna(sell_price) and sell_price > 0:
                    proceeds = shares_to_sell * sell_price
                    self.cash += proceeds
                    if shares_to_sell >= self.portfolio[ticker]['shares']:
                        del self.portfolio[ticker]
                    else:
                        self.portfolio[ticker]['shares'] -= shares_to_sell
                        self.portfolio[ticker]['profit_taken'] = True
                    # Append the reason to the transaction record
                    self.transactions.append((trade_date, 'SELL', ticker, shares_to_sell, sell_price, reason))

        # --- Execute Buys ---
        if buy_signals:
            holdings_value = 0.0
            for ticker, data in self.portfolio.items():
                current_price = trade_prices.get(ticker)
                if pd.notna(current_price) and data.get('shares', 0) > 0:
                    holdings_value += data['shares'] * current_price
            total_portfolio_value = self.cash + holdings_value
            max_positions = self.params.get('max_positions', 10)
            target_investment_per_stock = total_portfolio_value / max_positions

            for trade in buy_signals:
                ticker = trade['ticker']
                reason = trade['reason']  # Extract the reason

                if ticker in trade_prices.index:
                    buy_price = trade_prices[ticker]
                    if pd.notna(buy_price) and buy_price > 0:
                        investment_amount = min(target_investment_per_stock, self.cash)
                        if investment_amount > buy_price:
                            shares_to_buy = int(investment_amount / buy_price)
                            if shares_to_buy > 0:
                                cost = shares_to_buy * buy_price
                                self.cash -= cost
                                self.portfolio[ticker] = {
                                    'shares': shares_to_buy,
                                    'buy_price': buy_price,
                                    'profit_taken': False
                                }
                                # Append the reason to the transaction record
                                self.transactions.append((trade_date, 'BUY', ticker, shares_to_buy, buy_price, reason))

    def _record_portfolio_value(self, date):
        """记录当日收盘后的总资产"""
        close_prices = self.full_data_package['prices'].loc[date]
        holdings_value = 0

        for ticker, data in self.portfolio.items():
            if ticker in close_prices.index and pd.notna(close_prices[ticker]):
                holdings_value += data['shares'] * close_prices[ticker]

        total_value = self.cash + holdings_value
        self.portfolio_history.append((date, total_value))

    def analyze_performance(self, verbose=True):
        """计算关键性能指标"""
        history_df = pd.DataFrame(self.portfolio_history, columns=['Date', 'PortfolioValue'])
        if history_df.empty:
            print("警告: 没有生成有效的投资组合历史记录，无法进行分析。")
            return {}
        history_df.set_index('Date', inplace=True)
        
        
        benchmark_prices = self.full_data_package['prices']['SPY'].reindex(history_df.index)
        benchmark = benchmark_prices.pct_change().dropna()
        history_df['BenchmarkReturn'] = (benchmark + 1).cumprod()

        history_df['DailyReturn'] = history_df['PortfolioValue'].pct_change()
        
        total_return = (history_df['PortfolioValue'].iloc[-1] / self.initial_capital) - 1
        days = (history_df.index[-1] - history_df.index[0]).days
        annualized_return = (1 + total_return) ** (365.25 / days) - 1 if days > 0 else 0
        annualized_volatility = history_df['DailyReturn'].std() * np.sqrt(252)

        risk_free_rate = self.params.get('risk_free_rate', 0.02)
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
        
        roll_max = history_df['PortfolioValue'].cummax()
        daily_drawdown = history_df['PortfolioValue'] / roll_max - 1.0
        max_drawdown = daily_drawdown.min()

        results = {
            "params": self.params,
            "N (持股数)": self.params.get('max_positions', 10),
            "总回报 (Total Return)": total_return,
            "年化回报 (Annualized Return)": annualized_return,
            "夏普比率 (Sharpe Ratio)": sharpe_ratio,
            "最大回撤 (Max Drawdown)": max_drawdown,
            "performance_df": history_df
        }

        if verbose:
            print("\n--- 📊 回测性能分析 ---")
            for key, value in results.items():
                if key != 'performance_df':
                    if isinstance(value, (float, np.floating)):
                        if '回报' in key or '回撤' in key:
                            val_str = f"{value:.2%}"
                        else:
                            val_str = f"{value:.2f}"
                    else:
                        val_str = str(value)
                    print(f"{key:<30}: {val_str}")
        return results

    def plot_performance(self):
        """绘制资产曲线图"""
        if 'performance_df' not in self.results or self.results['performance_df'].empty:
            print("无法绘制性能图表，因为没有生成有效的回测结果。")
            return
            
        df = self.results['performance_df']
        
        normalized_value = df['PortfolioValue'] / self.initial_capital
        # [修改] 使用 'prices' 键
        benchmark_initial = self.full_data_package['prices']['SPY'].reindex(df.index).iloc[0]
        normalized_benchmark = self.full_data_package['prices']['SPY'].reindex(df.index) / benchmark_initial
        
        plt.figure(figsize=(15, 7))
        plt.plot(normalized_value.index, normalized_value, label='Strategy Equity Curve', color='royalblue', linewidth=2)
        plt.plot(normalized_benchmark.index, normalized_benchmark, label='SPY Benchmark', color='grey', linestyle='--')
        
        plt.title(f'Strategy Performance vs. Benchmark (SPY) (N={self.params["max_positions"]})', fontsize=16)
        plt.xlabel('Date')
        plt.ylabel('Normalized Value (Initial Capital = 1.0)')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # --- 1. 定义参数 ---
    BACKTEST_START_DATE = "2024-09-01"
    # 强烈建议将结束日期提前几天，以确保数据可用性
    BACKTEST_END_DATE = "2025-09-01" 
    DATA_DOWNLOAD_START = "2021-05-01"
    
    value_params = {
        'initial_capital': 1000000,
        'risk_free_rate': 0.02,
        'max_positions': 5,
        'filter_method': {'method': 'dynamic'},
        'recent_momentum_days': 21,
        'volatility_lookback_days': 126,
        'min_positive_days_pct': 0.5,
        'sp500_weight': 0.06,
        'rsi_length': 14,
        'potential_score_weights': {'price': 0.4, 'technical': 0.3, 'fundamental': 0.33},
        'final_score_weights': {'potential': 0.4, 'risk': 0.6},
        'sell_pct_on_high': 0.5,
        'stop_loss_pct': 0.4,
        'long_term_ma_days': 100
    }

    momentum_params = {
        # Backtester Parameters
        'initial_capital': 1000000,
        'risk_free_rate': 0.02,

        # Strategy General Parameters
        'max_positions': 10, # Let's test for N=10
        'stop_loss_pct': 0.15,

        # Momentum-Specific Parameters
        'high_proximity_pct': 0.10, 
        'trend_filter_ma_days': 50, 
        'exit_ma_days': 50, 
        'momentum_periods': [21, 63, 126], 
        'final_score_weights': {'potential': 0.7, 'risk': 0.3}
    }

    # --- 2. 获取数据 ---
    print("获取全美股票列表...")
    all_tickers = get_all_us_tickers()
    
    print("下载行情数据...")
    data_pkg = download_data(
        all_tickers,
        DATA_DOWNLOAD_START,
        BACKTEST_END_DATE,
        cache_file="ticker_cache/strategy_v3_backtest_v1_202105_2025_09.pkl", 
        screen_tickers=True, 
        overwrite_cache=False,
        max_workers=20   
    )
    
    # --- 3. 运行优化循环 ---
    if data_pkg['prices'].empty or data_pkg['fundamentals'].empty:
         print("\n数据下载失败或数据不完整，无法运行回测。")
    else:
        # 定义要测试的N值范围，例如测试 5, 7, 9
        max_positions_range = range(5, 6, 2)
        
        optimization_results = []
        best_sharpe = -np.inf
        best_result = None

        print(f"\n🚀 开始执行参数优化, 测试 N (最大持股数) 从 {min(max_positions_range)} 到 {max(max_positions_range)}...")
        
        for n in max_positions_range:
            print(f"\n--- 正在测试 N = {n} ---")
            current_params = momentum_params.copy()
            current_params['max_positions'] = n
            
            backtester = Backtester(
                full_data_package=data_pkg,
                strategy_class=ValueStrategy,
                start_date=BACKTEST_START_DATE,
                end_date=BACKTEST_END_DATE,
                initial_capital=current_params['initial_capital'],
                params=current_params,
                verbose=False
            )
            
            results = backtester.run()
            
            # 确保results非空再进行后续操作
            if results:
                optimization_results.append(results)
                if results.get('夏普比率 (Sharpe Ratio)', -np.inf) > best_sharpe:
                    best_sharpe = results['夏普比率 (Sharpe Ratio)']
                    best_result = results

    # --- 4. 打印优化结果汇总 ---
    if optimization_results:
        # 【修正1】创建DataFrame时，同时丢弃 'performance_df' 和 'params' 列
        summary_df = pd.DataFrame(optimization_results).drop(columns=['performance_df', 'params'])
        
        # 格式化输出
        summary_df['总回报 (Total Return)'] = summary_df['总回报 (Total Return)'].map('{:.2%}'.format)
        summary_df['年化回报 (Annualized Return)'] = summary_df['年化回报 (Annualized Return)'].map('{:.2%}'.format)
        summary_df['最大回撤 (Max Drawdown)'] = summary_df['最大回撤 (Max Drawdown)'].map('{:.2%}'.format)
        summary_df['夏普比率 (Sharpe Ratio)'] = summary_df['夏普比率 (Sharpe Ratio)'].map('{:.2f}'.format)
        
        print("\n\n--- 🏆 优化结果汇总 ---")
        print(summary_df.to_string(index=False))
    
    # --- 5. 绘制最佳结果的资产曲线图 ---
    if best_result:
        # 【修正2】不再重新调用 analyze_performance，而是直接打印已有的结果
        print(f"\n--- 📈 最佳结果 (N={best_result['N (持股数)']}) 表现 ---")
        print(f"{'总回报 (Total Return)':<30}: {best_result['总回报 (Total Return)']:.2%}")
        print(f"{'年化回报 (Annualized Return)':<30}: {best_result['年化回报 (Annualized Return)']:.2%}")
        print(f"{'夏普比率 (Sharpe Ratio)':<30}: {best_result['夏普比率 (Sharpe Ratio)']:.2f}")
        print(f"{'最大回撤 (Max Drawdown)':<30}: {best_result['最大回撤 (Max Drawdown)']:.2%}")

        # 重新初始化Backtester只是为了调用plot_performance方法
        best_params = best_result['params']
        best_backtester = Backtester(
            full_data_package=data_pkg,
            strategy_class=ValueStrategy,
            start_date=BACKTEST_START_DATE,
            end_date=BACKTEST_END_DATE,
            initial_capital=best_params['initial_capital'],
            params=best_params
        )
        best_backtester.results = best_result
        best_backtester.plot_performance()