import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas_ta as ta
import pandas as pd
from prepare_ticker_data import get_all_us_tickers, get_spy500_tickers, download_data, slice_data_package

class ValueStrategy:
    def __init__(self, data_pkg, sp500_tickers_set=None, params={}, verbose=True):
        self.prices = data_pkg['prices']
        self.volumes = data_pkg['volumes']
        self.fundamentals = data_pkg['fundamentals']
        self.sp500_tickers_set = sp500_tickers_set
        self.portfolio = {}
        self.candidates = pd.DataFrame()
        self.long_term_ma_days = params.get('long_term_ma_days', 200)

        self.max_positions = params.get('max_positions', 10)
        self.filter_method = params.get('filter_method', {'method': 'dynamic'})


        self.recent_momentum_days = params.get('recent_momentum_days', 14)
        self.volatility_lookback_days = params.get('volatility_lookback_days', 126)
        self.min_positive_days_pct = params.get('min_positive_days_pct', 0.30)
        
        self.sp500_weight = params.get('sp500_weight', 0.03)
        self.rsi_length = params.get('rsi_length', 14)
        self.potential_score_weights = params.get('potential_score_weights', {'price': 0.2, 'technical': 0.4, 'fundamental': 0.4})
        self.final_score_weights = params.get('final_score_weights', {'potential': 0.4, 'risk': 0.6})

        self.sell_pct_on_high = params.get('sell_pct_on_high', 0.7)
        self.stop_loss_pct = params.get('stop_loss_pct', 0.10)

        self.verbose = verbose
        # =================== END: Load all params ===================

    def set_portfolio(self, portfolio):
        self.portfolio = portfolio if portfolio is not None else {}

    def run_full_strategy(self):
        if self.filter_method['method'] == 'dynamic':
            initial_candidates = self.find_candidates_with_dynamic_threshold()
        elif self.filter_method['method'] == 'fixed':
            initial_candidates = self.find_candidates_near_52_week_low(self.filter_method['percentage_threshold'])
        else:
            print("Incorrect filter_method specified. Use 'dynamic' or 'fixed'.")
            return [], []

        momentum_candidates = self.filter_by_turnaround_signal(initial_candidates)
        self.score_candidates(momentum_candidates)
        
        buy_signals = self.generate_buy_signals()
        sell_signals = self.generate_sell_signals()

        return buy_signals, sell_signals
    
    def find_candidates_with_dynamic_threshold(self):
        if self.verbose:
            print(f"\n步骤 a): 使用基于当时市值的动态阈值筛选股票...")
        if len(self.prices) < 252: return []

        latest_prices = self.prices.iloc[-1]
        low_52_week = self.prices.rolling(window=252).min().iloc[-1]
        
        valid_tickers = latest_prices.dropna().index.intersection(self.fundamentals.index)
        
        # 动态计算当时的市值
        shares = self.fundamentals.loc[valid_tickers, 'sharesOutstanding'].fillna(0)
        market_caps_today = latest_prices[valid_tickers] * shares
        
        large_cap_min, mid_cap_min = 10e9, 2e9
        thresholds = pd.Series(1.00, index=valid_tickers)
        thresholds[market_caps_today >= mid_cap_min] = 0.50
        thresholds[market_caps_today >= large_cap_min] = 0.30

        buy_threshold_price = low_52_week * (1 + thresholds)
        condition = latest_prices <= buy_threshold_price
        
        initial_candidates = latest_prices[condition].index.tolist()
        if self.verbose:
            print(f"找到 {len(initial_candidates)} 个符合动态阈值条件的股票。")
        return initial_candidates

    def find_candidates_near_52_week_low(self):
        if self.verbose:
            print(f"\n步骤 a): 寻找价格在52周低点 {self.filter_method['percentage_threshold']*100}% 内的股票...")
        
        if len(self.prices) < 252:
            if self.verbose:
                print("数据不足252天，无法计算52周高低点。")
            return []

        low_52_week = self.prices.rolling(window=252).min().iloc[-1]
        latest_prices = self.prices.iloc[-1]
        condition = latest_prices <= low_52_week * (1 + self.filter_method['percentage_threshold'])
        initial_candidates = latest_prices[condition].index.tolist()
        if self.verbose:
            print(f"找到 {len(initial_candidates)} 个符合条件的初始候选股票。")
        return initial_candidates

    def filter_by_positive_momentum(self, tickers):
        if self.verbose:
            print("\n步骤 b): 过滤掉过去一年总回报为负的股票...")
        
        if len(self.prices) < 252:
            return []

        price_1y_ago = self.prices.iloc[-252]
        latest_prices = self.prices.iloc[-1]

        returns_1y = (latest_prices[tickers] / price_1y_ago[tickers]) - 1
        
        positive_momentum_stocks = returns_1y[returns_1y > 0].index.tolist()
        
        print(f"在候选池中，有 {len(positive_momentum_stocks)} 只股票过去一年回报为正。")
        return positive_momentum_stocks
    
    def filter_by_turnaround_signal(self, tickers):
        if self.verbose:
            print("\n步骤 b): 筛选具备困境反转信号的股票...")

        if len(self.prices) < self.volatility_lookback_days:
            print(f"数据不足 {self.volatility_lookback_days} 天，无法执行困境反转筛选。")
            return []

        if self.verbose:
            print(f"  - 条件1: 检查过去 {self.recent_momentum_days} 天回报率是否为正...")
        price_lookback_ago = self.prices.iloc[-self.recent_momentum_days]
        latest_prices = self.prices.iloc[-1]
        
        returns_recent = (latest_prices[tickers] / price_lookback_ago[tickers]) - 1
        month_up_stocks = returns_recent[returns_recent > 0].index.tolist()
        
        if not month_up_stocks:
            if self.verbose:
                print("  - 结果: 没有候选股票在最近一个月上涨，筛选结束。")
            return []
        if self.verbose:
            print(f"  - 结果: {len(month_up_stocks)} 只股票满足条件1。")

        if self.verbose:
            print(f"  - 条件2: 检查过去 {self.volatility_lookback_days} 天上涨天数占比是否 > {self.min_positive_days_pct*100}%...")
        
        prices_filled = self.prices.ffill()
        daily_returns = prices_filled.pct_change(fill_method=None)
        
        returns_period = daily_returns.iloc[-self.volatility_lookback_days:][month_up_stocks]
        
        positive_days_pct = (returns_period > 0).mean()
        
        final_candidates = positive_days_pct[positive_days_pct >= self.min_positive_days_pct].index.tolist()
        if self.verbose:
            print(f"  - 结果: {len(final_candidates)} 只股票满足条件2。")
            print(f"最终有 {len(final_candidates)} 只股票通过了困境反转信号筛选。")
        
        return final_candidates  
        
    def score_candidates(self, tickers):
        if self.verbose:
            print(f"\n步骤 c): 对剩余候选股票进行风险和潜力打分...")
        if not tickers: self.candidates = pd.DataFrame(); return
        valid_tickers = [t for t in tickers if t in self.fundamentals.index and t in self.prices.columns]
        if not valid_tickers: self.candidates = pd.DataFrame(); return
        
        clean_factors_df = self.fundamentals.loc[valid_tickers].copy()
        latest_prices = self.prices[valid_tickers].iloc[-1]
        avg_volume_3m = self.volumes[valid_tickers].rolling(window=63).mean().iloc[-1]

        shares = clean_factors_df['sharesOutstanding'].fillna(0)
        market_caps_today = latest_prices * shares
        clean_factors_df['marketCap_today'] = market_caps_today

        clean_factors_df['avg_volume'] = avg_volume_3m + 1
        
        for col in clean_factors_df.columns:
            clean_factors_df[col] = pd.to_numeric(clean_factors_df[col], errors='coerce')
        clean_factors_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        clean_factors_df.fillna(clean_factors_df.median(), inplace=True)

        low_52_week = self.prices[valid_tickers].rolling(252).min().iloc[-1]
        high_52_week = self.prices[valid_tickers].rolling(252).max().iloc[-1]
        score_price_value = (1 - (latest_prices - low_52_week) / (high_52_week - low_52_week)) ** 2
        rsi = self.prices[valid_tickers].apply(lambda col: ta.rsi(col, length=self.rsi_length)).iloc[-1]
        score_technical_value = (1 - (rsi - 30) / 40).clip(0, 1)
        pe_rank = 1 - clean_factors_df['trailingPE'].rank(pct=True, ascending=True)
        ps_rank = 1 - clean_factors_df['priceToSalesTrailing12Months'].rank(pct=True, ascending=True)
        score_fundamental_value = ((pe_rank + ps_rank) / 2).fillna(0.5)

        weights = self.potential_score_weights
        scoring_df = pd.DataFrame(index=valid_tickers)
        scoring_df['potential_score'] = (weights['price'] * score_price_value + weights['technical'] * score_technical_value + weights['fundamental'] * score_fundamental_value).fillna(0)

        mkt_cap_log = np.log(clean_factors_df['marketCap_today'] + 1)
        volume_log = np.log(clean_factors_df['avg_volume'] + 1)
        mkt_cap_score = self._apply_sigmoid_scaling(mkt_cap_log, higher_is_better=True)
        volume_score = self._apply_sigmoid_scaling(volume_log, higher_is_better=True)
        scoring_df['risk_score'] = (mkt_cap_score + volume_score) / 2
        
        is_sp500 = scoring_df.index.isin(self.sp500_tickers_set)
        bonus_score = is_sp500.astype(float) * self.sp500_weight
        
        final_weights = self.final_score_weights
        scoring_df['final_score'] = final_weights['potential'] * scoring_df['potential_score'] + final_weights['risk'] * scoring_df['risk_score']
        scoring_df['final_score'] += bonus_score
        
        final_columns = ['potential_score', 'risk_score', 'final_score']
        self.candidates = scoring_df[final_columns].dropna().sort_values(by='final_score', ascending=False)
        if self.verbose:
            print(f"打分完成，得到 {len(self.candidates)} 个最终候选股票。")
            print(self.candidates.head(10))

    def generate_buy_signals(self):
        if self.candidates.empty:
            return []

        active_positions_count = sum(1 for stock_data in self.portfolio.values() if not stock_data.get('profit_taken'))
        positions_to_open = self.max_positions - active_positions_count

        if positions_to_open <= 0:
            print(f"持仓已满 ({active_positions_count}/{self.max_positions} 个核心持仓)，本周不买入。")
            return []
        
        num_to_buy = 1 
        
        potential_buys = self.candidates[~self.candidates.index.isin(self.portfolio.keys())]
        
        if potential_buys.empty:
            if self.verbose:
                print("没有符合买入条件的股票，本周不买入。")
            return []

        top_for_buy = potential_buys.head(num_to_buy)

        buy_signals = []
        for ticker in top_for_buy.index:
            rank = self.candidates.index.get_loc(ticker) + 1
            row = self.candidates.loc[ticker]
            if self.verbose:
                print(f"买入: {ticker} | 总排名 {rank} | "
                    f"potential={row['potential_score']:.2f}, "
                    f"risk={row['risk_score']:.2f}, "
                    f"final={row['final_score']:.2f}")

            reason = f"Top Rank #{rank}"
            buy_signals.append({'ticker': ticker, 'reason': reason})

        return buy_signals
    
    def generate_sell_signals(self):
        if not self.portfolio:
            return []

        sell_list = []
        tickers_in_portfolio = list(self.portfolio.keys())

        if len(self.prices) < self.long_term_ma_days:
            return []

        high_52_week = self.prices[tickers_in_portfolio].rolling(252).max().iloc[-1]
        long_term_ma = self.prices[tickers_in_portfolio].rolling(self.long_term_ma_days).mean().iloc[-1]
        latest_prices = self.prices[tickers_in_portfolio].iloc[-1]

        for ticker in tickers_in_portfolio:
            current_price = latest_prices.get(ticker)
            if pd.isna(current_price):
                continue

            is_profit_taken = self.portfolio[ticker].get('profit_taken')

            # Logic for REMNANT (30%) Holdings
            if is_profit_taken:
                ma_value = long_term_ma.get(ticker)
                if pd.notna(ma_value) and current_price < ma_value:
                    shares_to_sell = self.portfolio[ticker]['shares']
                    if shares_to_sell > 0:
                        reason = f'LT MA-{self.long_term_ma_days} Exit'
                        print(f"SELL ({reason}): {ticker} at ${current_price:.2f}")
                        sell_list.append({'ticker': ticker, 'shares': shares_to_sell, 'reason': reason})

            # Logic for ACTIVE (Core) Holdings
            else:
                buy_price = self.portfolio[ticker].get('buy_price')
                if pd.isna(buy_price): continue

                # 1. Stop-Loss Logic
                stop_loss_price = buy_price * (1 - self.stop_loss_pct)
                if current_price <= stop_loss_price:
                    shares_to_sell = self.portfolio[ticker]['shares']
                    if shares_to_sell > 0:
                        reason = 'Stop-Loss'
                        print(f"SELL ({reason}): {ticker} at ${current_price:.2f}")
                        sell_list.append({'ticker': ticker, 'shares': shares_to_sell, 'reason': reason})
                    continue

                # 2. Profit-Taking Logic
                target_price = high_52_week.get(ticker)
                if pd.notna(target_price) and current_price >= target_price:
                    total_shares = self.portfolio[ticker]['shares']
                    shares_to_sell = int(total_shares * self.sell_pct_on_high)
                    if shares_to_sell > 0:
                        reason = 'Profit-Taking'
                        print(f"SELL ({reason}): {ticker} at ${current_price:.2f}")
                        sell_list.append({'ticker': ticker, 'shares': shares_to_sell, 'reason': reason})

        return sell_list
        
    def _apply_sigmoid_scaling(self, series, higher_is_better=True):
        values = series.dropna().values.reshape(-1, 1)
        if len(values) == 0:
            return pd.Series(index=series.index)

        scaler = StandardScaler()
        z_scores = scaler.fit_transform(values)
        if not higher_is_better:
            z_scores = -z_scores
            
        scores = 1 / (1 + np.exp(-z_scores))
        return pd.Series(scores.flatten(), index=series.dropna().index)
    
    def debug_stock(self, ticker):
        """[完整版] 追踪单只股票以分析其未被选中的原因，并适配策略选择。"""
        print(f"\n--- 🕵️ 调试股票: {ticker} ---")
        
        if ticker not in self.prices.columns or ticker not in self.fundamentals.index:
            print(f"❌ 原因: 未找到 {ticker} 的数据。")
            return
        
        latest_price = self.prices[ticker].iloc[-1]
        if pd.isna(latest_price):
            print(f"❌ 原因: {ticker} 的最新价格无效。")
            return
        print(f"✅ 数据找到。最新价格: ${latest_price:.2f}")

        filter_method = self.filter_method.get('method', 'dynamic')
        print(f"\n--- 步骤 A: 52周低点筛选 (使用 '{filter_method}' 策略) ---")
        
        if len(self.prices[ticker].dropna()) < 252:
            print(f"❌ 淘汰: {ticker} 的历史价格数据不足252天，无法计算52周低点。")
            return

        low_52_week = self.prices[ticker].rolling(252).min().iloc[-1]

        passed_step_a = False
        if filter_method == 'dynamic':
            shares = self.fundamentals.loc[ticker, 'sharesOutstanding']
            market_cap_today = latest_price * shares
            
            large_cap_min, mid_cap_min = 10e9, 2e9
            if market_cap_today >= large_cap_min:
                threshold, cap_tier = 0.30, "大盘股"
            elif market_cap_today >= mid_cap_min:
                threshold, cap_tier = 0.50, "中盘股"
            else:
                threshold, cap_tier = 1.00, "小盘股"
            
            buy_threshold_price = low_52_week * (1 + threshold)
            print(f"  - 当时市值等级: {cap_tier} (${market_cap_today:,.0f})")
            print(f"  - 52周最低点: ${low_52_week:.2f}")
            print(f"  - 动态买入阈值 ({threshold:.0%}): ${buy_threshold_price:.2f}")
            if latest_price <= buy_threshold_price:
                passed_step_a = True
            else:
                print(f"  - 详情: 当前价格 ${latest_price:.2f} > 阈值价格 ${buy_threshold_price:.2f}")

        else:
            threshold = self.filter_method.get('threshold', 0.15)
            buy_threshold_price = low_52_week * (1 + threshold)
            print(f"  - 52周最低点: ${low_52_week:.2f}")
            print(f"  - 固定买入阈值 ({threshold*100}%): ${buy_threshold_price:.2f}")
            if latest_price <= buy_threshold_price:
                passed_step_a = True
            else:
                print(f"  - 详情: 当前价格 ${latest_price:.2f} > 阈值价格 ${buy_threshold_price:.2f}")

        if passed_step_a:
            print(f"  - ✅ 通过: 最新价格低于阈值。")
        else:
            print(f"  - ❌ 淘汰: 最新价格高于阈值。")
            print(f"\n👉 结论: {ticker} 在步骤A被淘汰。")
            return
        
        print("\n--- 步骤 B: 困境反转信号筛选 ---")
        recent_momentum_days=21
        volatility_lookback_days=126
        min_positive_days_pct=0.30

        if len(self.prices[ticker].dropna()) < recent_momentum_days:
            print(f"  - ❌ 淘汰: {ticker} 的历史价格数据不足 {recent_momentum_days} 天，无法计算近期动量。")
            return

        price_then = self.prices[ticker].iloc[-recent_momentum_days]
        recent_return = (latest_price / price_then) - 1
        print(f"  - 部分1: 近期动量 ({recent_momentum_days} 天)")
        print(f"    - {recent_momentum_days}天前价格: ${price_then:.2f}")
        print(f"    - 近期回报率: {recent_return:.2%}")
        if recent_return > 0:
            print("    - ✅ 通过: 近期回报为正。")
        else:
            print("    - ❌ 淘汰: 近期回报为负。")
            print(f"    - 详情: 回报率 {recent_return:.2%}，不大于 0%。")
            print(f"\n👉 结论: {ticker} 因最近 {recent_momentum_days} 天内没有上涨而被淘汰。")
            return
        
        if len(self.prices[ticker].dropna()) < volatility_lookback_days:
            print(f"  - ❌ 淘汰: {ticker} 的历史价格数据不足 {volatility_lookback_days} 天，无法计算波动率。")
            return

        prices_filled = self.prices[ticker].ffill()
        daily_returns = prices_filled.pct_change(fill_method=None)
        
        returns_period = daily_returns.iloc[-volatility_lookback_days:]
        positive_days_pct = (returns_period > 0).mean()
        print(f"  - 部分2: 回弹波动 ({volatility_lookback_days} 天)")
        print(f"    - 上涨天数占比: {positive_days_pct:.2%}")
        print(f"    - 要求占比 > {min_positive_days_pct:.2%}")
        if positive_days_pct >= min_positive_days_pct:
            print("    - ✅ 通过: 上涨天数充足。")
        else:
            print("    - ❌ 淘汰: 在过去半年中上涨天数不足。")
            print(f"    - 详情: 实际占比 {positive_days_pct:.2%}，未达到要求 > {min_positive_days_pct:.2%}")
            print(f"\n👉 结论: {ticker} 因缺乏足够的回弹波动而被淘汰。")
            return

        print("\n--- 步骤 C: 最终得分分析 ---")
        if ticker in self.candidates.index:
            ticker_scores = self.candidates.loc[ticker]
            rank = list(self.candidates.index).index(ticker) + 1
            print(f"  - ✅ 通过所有筛选并已打分。")
            print(f"  - 潜力分 (Potential Score): {ticker_scores['potential_score']:.3f}")
            print(f"  - 风险分 (Risk Score): {ticker_scores['risk_score']:.3f}")
            print(f"  - 最终分 (Final Score): {ticker_scores['final_score']:.3f}")
            print(f"  - 最终排名: 第 {rank} 名，共 {len(self.candidates)} 名")
            top_n = self.max_positions # 假设 top_n 为10，可以根据实际情况调整
            if rank > top_n:
                print(f"\n 结论: {ticker} 通过了所有筛选，但最终得分不够高，未能进入前{top_n}。")
            else:
                print(f"\n🎉 结论: {ticker} 成功入选Top {top_n}！")
        else:
            print(f"  - ❌ 淘汰: {ticker} 通过了所有筛选，但由于数据问题未出现在最终候选名单中。")



if __name__ == '__main__':
    print("获取全美股票列表...")
    all_tickers = get_all_us_tickers() # 
    spy_tickers = get_spy500_tickers()

    START_DATE = "2025-01-01"
    END_DATE = "2025-09-20"
    DECISION_DATE = "2025-05-01"  
    print("下载行情数据...")
    data_pkg = download_data(
        all_tickers,
        START_DATE,
        END_DATE,
        cache_file="ticker_cache/strategy_v3.pkl", 
        screen_tickers=True, 
        overwrite_cache=False,
        max_workers=20  
    )
    print("步骤 2.1: 截取数据包到决策日...")
    data_pkg = slice_data_package(data_pkg, DECISION_DATE)
    params = {
        # Backtester Parameters
        'initial_capital': 100000,
        'risk_free_rate': 0.02,

        # Strategy General Parameters
        'max_positions': 20, # This will be changed by the optimization loop
        'filter_method': {'method': 'dynamic'},

        # filter_by_turnaround_signal Parameters
        'recent_momentum_days': 14,
        'volatility_lookback_days': 126,
        'min_positive_days_pct': 0.30,
        
        # score_candidates Parameters
        'sp500_weight': 0.03,
        'rsi_length': 14,
        'potential_score_weights': {'price': 0.2, 'technical': 0.4, 'fundamental': 0.4},
        'final_score_weights': {'potential': 0.6, 'risk': 0.4},

        # generate_sell_signals Parameters
        'sell_pct_on_high': 0.7,
        'stop_loss_pct': 0.20
    }
    if data_pkg['prices'].empty or data_pkg['fundamentals'].empty:
         print("\n数据下载失败或数据不完整，无法运行策略。")
    else:
        strategy = ValueStrategy(data_pkg, sp500_tickers_set=set(spy_tickers), params=params)
        strategy.run_full_strategy()
        strategy.debug_stock('UNH')
        strategy.debug_stock('CRDO')
        strategy.debug_stock('GOOG')
        strategy.debug_stock('AAPL')