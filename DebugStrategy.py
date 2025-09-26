import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from prepare_ticker_data import download_data, get_all_us_tickers, get_spy500_tickers
import pandas_ta as ta
import random
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class StockEmbeddingNet(nn.Module):
    def __init__(self, n_unique_tickers, embedding_dim, n_numerical_features, hidden_size, output_size=1, dropout_rate=0.3):
        super(StockEmbeddingNet, self).__init__()
        self.ticker_embedding = nn.Embedding(n_unique_tickers, embedding_dim)
        
        self.feature_processor = nn.Sequential(
            nn.Linear(n_numerical_features, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout_rate)
        )
        
        self.final_layers = nn.Sequential(
            nn.Linear(embedding_dim + hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size)
        )

    def forward(self, ticker_indices, numerical_features):
        embedding_out = self.ticker_embedding(ticker_indices)
        features_out = self.feature_processor(numerical_features)
        combined = torch.cat([embedding_out, features_out], dim=1)
        output = self.final_layers(combined)
        return output


class MLStrategy:
    def __init__(self, params={}):
        self.verbose = params.get('verbose', True)
        self.top_n_stocks = params.get('top_n_stocks', 20)
        
        self.embedding_dim = params.get('embedding_dim', 256)
        self.hidden_size = params.get('hidden_size', 256)
        self.learning_rate = params.get('learning_rate', 0.004)
        self.epochs = params.get('epochs', 30)
        self.batch_size = params.get('batch_size', 1024)
        self.num_workers = params.get('num_workers', 8)
        self.model = None
        self.scaler = StandardScaler()
        self.ticker_to_idx = {}
        self.prediction_results = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_features(self, prices, volumes):
        if self.verbose: print("步骤 1: 创建特征...")
        
        prices_s = prices.stack(dropna=False).rename('price').reset_index()
        prices_s.columns = ['Date', 'ticker', 'price']
        volumes_s = volumes.stack(dropna=False).rename('volume').reset_index()
        volumes_s.columns = ['Date', 'ticker', 'volume']
        
        df = pd.merge(prices_s, volumes_s, on=['Date', 'ticker'])
        df = df.sort_values(by=['ticker', 'Date']).set_index('Date')
        
        feature_list = []
        
        for days in [21, 63, 126, 252]:
            df[f'return_{days}d'] = df.groupby('ticker')['price'].pct_change(days)
            feature_list.append(f'return_{days}d')

        df['rsi_14'] = df.groupby('ticker')['price'].transform(lambda x: ta.rsi(x, length=14))
        feature_list.append('rsi_14')

        def safe_calculate_macd(series, fast=12, slow=26, signal=9):
            try:
                if len(series) < slow + signal: return pd.Series(np.nan, index=series.index)
                macd_df = ta.macd(series, fast=fast, slow=slow, signal=signal)
                if macd_df is None or f'MACDh_{fast}_{slow}_{signal}' not in macd_df.columns: return pd.Series(np.nan, index=series.index)
                return pd.to_numeric(macd_df[f'MACDh_{fast}_{slow}_{signal}'], errors='coerce')
            except Exception:
                return pd.Series(np.nan, index=series.index)
        
        df['macd_histogram'] = df.groupby('ticker')['price'].transform(safe_calculate_macd)
        feature_list.append('macd_histogram')

        df['return_63d_rank'] = df.groupby('Date')['return_63d'].rank(pct=True, na_option='bottom')
        feature_list.append('return_63d_rank')
        
        df['avg_volume_21d'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(21).mean())
        df['daily_return'] = df.groupby('ticker')['price'].pct_change()
        df['volatility_63d'] = df.groupby('ticker')['daily_return'].transform(lambda x: x.rolling(63).std())
        df['volume_surge_ratio'] = df['volume'] / df['avg_volume_21d']
        feature_list.extend(['avg_volume_21d', 'volatility_63d', 'volume_surge_ratio'])
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df[['ticker'] + feature_list].reset_index()

    def create_labels(self, prices):
        if self.verbose: print("步骤 2: 创建分类标签...")
        
        future_returns = (prices.shift(-5) / prices) - 1

        top_quartile_threshold = future_returns.quantile(0.75, axis=1)
        labels = future_returns.apply(lambda x: (x > top_quartile_threshold).astype(int), axis=0)

        labels_s = labels.stack(dropna=True).rename('label').reset_index()
        labels_s.columns = ['Date', 'ticker', 'label']
        return labels_s

    def align_and_prepare_data(self, train_features, train_labels, predict_features):
        if self.verbose: print("步骤 3: 对齐数据并为神经网络准备...")
        
        train_data = pd.merge(train_features, train_labels, on=['Date', 'ticker'], how='inner').dropna()
        unique_tickers = train_data['ticker'].unique()
        self.ticker_to_idx = {ticker: i for i, ticker in enumerate(unique_tickers)}
        self.ticker_to_idx['UNK'] = len(unique_tickers)
        
        train_data['ticker_idx'] = train_data['ticker'].map(self.ticker_to_idx)
        numerical_feature_cols = train_data.columns.drop(['Date', 'ticker', 'label', 'ticker_idx'])
        
        X_num_train = train_data[numerical_feature_cols]
        X_cat_train = train_data['ticker_idx']
        y_train = train_data['label']

        unk_idx = self.ticker_to_idx.get('UNK', -1)
        predict_features['ticker_idx'] = predict_features['ticker'].map(self.ticker_to_idx).fillna(unk_idx).astype(int)
        X_num_predict = predict_features[numerical_feature_cols]
        X_cat_predict = predict_features['ticker_idx']
        
        return X_num_train, X_cat_train, y_train, X_num_predict, X_cat_predict, predict_features[['Date', 'ticker']]

    def train_model(self, X_num_train, X_cat_train, y_train):
        if self.verbose: print("步骤 4: 训练神经网络...")
        
        X_num_train_scaled = self.scaler.fit_transform(X_num_train)
        train_dataset = TensorDataset(
            torch.LongTensor(X_cat_train.values),
            torch.FloatTensor(X_num_train_scaled),
            torch.FloatTensor(y_train.values)
        )

        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,  
            pin_memory=True,           
        )
        
        n_unique_tickers = len(self.ticker_to_idx)
        n_numerical_features = X_num_train.shape[1]
        self.model = StockEmbeddingNet(n_unique_tickers, self.embedding_dim, n_numerical_features, self.hidden_size)
        self.model.to(self.device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for cat_batch, num_batch, y_batch in train_loader:
                cat_batch, num_batch, y_batch = cat_batch.to(self.device), num_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(cat_batch, num_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"    - Epoch {epoch+1}/{self.epochs}, 损失: {total_loss/len(train_loader):.6f}")

    def generate_signals(self, X_num_predict, X_cat_predict, identifiers):
        if self.verbose: print("步骤 5: 生成信号...")
        X_pred_num_scaled = self.scaler.transform(X_num_predict)
        
        X_pred_cat_tensor = torch.LongTensor(X_cat_predict.values).to(self.device)
        X_pred_num_tensor = torch.FloatTensor(X_pred_num_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions_tensor = self.model(X_pred_cat_tensor, X_pred_num_tensor).squeeze()
            probabilities = torch.sigmoid(predictions_tensor)
        
        predictions = probabilities.cpu().numpy()
        
        results = identifiers.copy()
        results['predicted_probability'] = predictions
        
        ranked_stocks = results.sort_values(by='predicted_probability', ascending=False)
        return ranked_stocks.head(self.top_n_stocks)


def run_walk_forward_backtest(data_pkg, backtest_params, ml_params):
    """
    执行步进式回测的核心函数。
    """
    print("--- 开始步进式回测 ---")
    prices, volumes = data_pkg['prices'], data_pkg['volumes']
    
    print("\n[全局准备] 正在为整个数据集预先计算特征和标签...")
    temp_strategy = MLStrategy({'verbose': False})
    all_features = temp_strategy.create_features(prices, volumes)
    all_labels = temp_strategy.create_labels(prices)
    print("[全局准备] 完成.\n")

    all_dates = sorted(all_features['Date'].unique())
    backtest_start_date = pd.to_datetime(backtest_params['backtest_start_date'])
    train_window = pd.Timedelta(days=backtest_params['train_window_days'])

    decision_dates = [d for d in all_dates if d >= backtest_start_date]
    decision_dates = pd.to_datetime(decision_dates)
    decision_dates = decision_dates[::int(backtest_params['rebalance_period_days']/7 * 5)] 

    all_predictions = []

    for decision_date in tqdm(decision_dates, desc="步进式回测进度"):
        train_start_date = decision_date - train_window
        
        train_mask = (all_features['Date'] >= train_start_date) & (all_features['Date'] < decision_date)
        predict_mask = all_features['Date'] == decision_date
        
        train_features_fold = all_features[train_mask]
        predict_features_fold = all_features[predict_mask]

        if predict_features_fold.empty or train_features_fold.empty:
            print(f"跳过 {decision_date.date()}: 训练或预测数据不足。")
            continue

        strategy = MLStrategy(ml_params)
        
        X_num_train, X_cat_train, y_train, X_num_predict, X_cat_predict, identifiers = \
            strategy.align_and_prepare_data(train_features_fold, all_labels, predict_features_fold)

        strategy.train_model(X_num_train, X_cat_train, y_train)
        
        top_stocks_fold = strategy.generate_signals(X_num_predict, X_cat_predict, identifiers)
        
        all_predictions.append(top_stocks_fold)

    print("\n--- 回测完成 ---")
    if not all_predictions:
        print("错误：回测期间未能生成任何预测。")
        return None
        
    final_results_df = pd.concat(all_predictions)
    final_results_with_labels = pd.merge(final_results_df, all_labels, on=['Date', 'ticker'], how='left')
    
    return final_results_with_labels


def analyze_backtest_results(results_df):
    """
    计算并打印回测结果的关键性能指标。
    """
    if results_df is None or results_df.empty:
        print("结果为空，无法进行分析。")
        return

    print("\n--- 回测结果分析 ---")
    
    hit_rate = results_df['label'].mean()
    
    print(f"回测总天数: {results_df['Date'].nunique()}")
    print(f"平均每日选股数: {results_df.groupby('Date').size().mean():.2f}")
    print(f"总选股次数 (天数 * 选股数): {len(results_df)}")
    print(f"\n核心指标 - 选股命中率 (Precision@K): {hit_rate:.2%}")
    print("  - (解释: 在所有周期选出的'优等股'中，真正成为优等股的比例)")

    results_df['year'] = results_df['Date'].dt.year
    yearly_hit_rate = results_df.groupby('year')['label'].mean()
    print("\n按年份分析命中率:")
    print(yearly_hit_rate.to_string(float_format='{:.2%}'.format))
    
    return hit_rate


if __name__ == '__main__':
    print("获取 S&P 500 股票列表...")
    tickers_to_download = get_spy500_tickers()
    
    START_DATE = "2018-01-01" 
    END_DATE = "2025-09-23"
    
    print("下载历史数据...")
    data_pkg = download_data(
        tickers_to_download,
        START_DATE,
        END_DATE,
        cache_file="ticker_cache/spy500_2018_2025.pkl",
        screen_tickers=True,
        overwrite_cache=False,
        max_workers=20
    )
    
    if data_pkg['prices'].empty:
        print("\n数据下载失败或不完整。")
    else:
        backtest_params = {
            'backtest_start_date': '2023-01-01', 
            'train_window_days': 365 * 2,      
            'rebalance_period_days': 5         
        }

        ml_params = {
            'epochs': 2,
            'top_n_stocks': 20,
            'verbose': True, 
            'batch_size': 16384*2,
            'num_workers': 4 
        }

        backtest_results = run_walk_forward_backtest(data_pkg, backtest_params, ml_params)
        analyze_backtest_results(backtest_results)
