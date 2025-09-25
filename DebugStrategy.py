import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from prepare_ticker_data import download_data, get_all_us_tickers, get_spy500_tickers
import pandas_ta as ta
import random

# --- Optional: reproducibility ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --- 1. PyTorch Neural Network Definition for Classification ---
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
        # Output raw logits for BCEWithLogitsLoss
        output = self.final_layers(combined)
        return output


class MLStrategy:
    def __init__(self, params={}):
        self.verbose = params.get('verbose', True)
        self.top_n_stocks = params.get('top_n_stocks', 20)
        self.label_period_start_days = params.get('label_period_start_days', 20)
        self.label_avg_window_days = params.get('label_avg_window_days', 5)
        
        self.embedding_dim = params.get('embedding_dim', 256)
        self.hidden_size = params.get('hidden_size', 256)
        self.learning_rate = params.get('learning_rate', 0.004)
        self.epochs = params.get('epochs', 30)
        self.batch_size = params.get('batch_size', 1024)

        self.model = None
        self.scaler = StandardScaler()
        self.ticker_to_idx = {}
        self.prediction_results = None
        self.device = None

    def create_features(self, prices, volumes, fundamentals):
        """
        Generates a feature set for each stock on each day.
        Fundamentals merge removed to avoid look-ahead leakage.
        """
        if self.verbose: print("步骤 1: 创建特征...")
        
        prices_s = prices.stack(dropna=False).rename('price').reset_index()
        prices_s.columns = ['Date', 'ticker', 'price']
        volumes_s = volumes.stack(dropna=False).rename('volume').reset_index()
        volumes_s.columns = ['Date', 'ticker', 'volume']
        
        df = pd.merge(prices_s, volumes_s, on=['Date', 'ticker'])
        df = df.sort_values(by=['ticker', 'Date']).set_index('Date')
        
        feature_list = []
        
        # 1. 动量与技术指标
        for days in [21, 63, 126, 252]:
            df[f'return_{days}d'] = df.groupby('ticker')['price'].pct_change(days)
            feature_list.append(f'return_{days}d')

        df['rsi_14'] = df.groupby('ticker')['price'].transform(lambda x: ta.rsi(x, length=14))
        feature_list.append('rsi_14')

        def safe_calculate_macd(series, fast=12, slow=26, signal=9):
            try:
                if len(series) < slow + signal:
                    return pd.Series(np.nan, index=series.index)
                macd_df = ta.macd(series, fast=fast, slow=slow, signal=signal)
                if macd_df is None or f'MACDh_{fast}_{slow}_{signal}' not in macd_df.columns:
                    return pd.Series(np.nan, index=series.index)
                return pd.to_numeric(macd_df[f'MACDh_{fast}_{slow}_{signal}'], errors='coerce')
            except Exception:
                return pd.Series(np.nan, index=series.index)
        
        df['macd_histogram'] = df.groupby('ticker')['price'].transform(safe_calculate_macd)
        feature_list.append('macd_histogram')

        # 2. 截面（相对）排名
        df['return_63d_rank'] = df.groupby('Date')['return_63d'].rank(pct=True, na_option='bottom')
        feature_list.append('return_63d_rank')
        
        # 3. 流动性与波动率
        df['avg_volume_21d'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(21).mean())
        df['daily_return'] = df.groupby('ticker')['price'].pct_change()
        df['volatility_63d'] = df.groupby('ticker')['daily_return'].transform(lambda x: x.rolling(63).std())
        df['volume_surge_ratio'] = df['volume'] / df['avg_volume_21d']
        feature_list.extend(['avg_volume_21d', 'volatility_63d', 'volume_surge_ratio'])
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        features_df = df[feature_list + ['ticker']].copy().dropna(how='any')
        if self.verbose: print(f"  - 特征创建完成. 维度: {features_df.shape}")
        return features_df

    def create_labels(self, prices):
        """
        Creates a binary classification label.
        Label is 1 if the stock's future return (next 5 days average) 
        is in the top quartile for that day, 0 otherwise.
        """
        if self.verbose: print("步骤 2: 创建分类标签...")

        # 未来 5 日均价（向前 shift，再 rolling）
        future_avg_price = prices.shift(-1).rolling(window=5).mean()
        
        # 计算未来 5 日的平均收益率
        returns = (future_avg_price / prices) - 1

        # 横截面分位数：每天前 25% 的股票标记为 1
        top_quartile_threshold = returns.quantile(0.75, axis=1)
        labels = returns.apply(lambda x: (x > top_quartile_threshold).astype(int), axis=0)

        labels_s = labels.stack(dropna=True).rename('label').reset_index()
        labels_s.columns = ['Date', 'ticker', 'label']
        if self.verbose: print(f"  - 标签创建完成. 维度: {labels_s.shape}")
        return labels_s


    def align_and_prepare_data(self, features, labels):
        if self.verbose: print("步骤 3: 对齐数据并为神经网络准备...")
        features = features.reset_index()
        data = pd.merge(features, labels, on=['Date', 'ticker'], how='inner').dropna()
        unique_tickers = data['ticker'].unique()
        self.ticker_to_idx = {ticker: i for i, ticker in enumerate(unique_tickers)}
        self.ticker_to_idx['UNK'] = len(unique_tickers)
        data['ticker_idx'] = data['ticker'].map(self.ticker_to_idx)
        numerical_feature_cols = data.columns.drop(['Date', 'ticker', 'label', 'ticker_idx'])
        X_numerical = data[numerical_feature_cols]
        X_categorical = data['ticker_idx']
        y = data['label']
        if self.verbose: print(f"  - 最终数据集准备就绪. 维度: {X_numerical.shape}")
        return X_numerical, X_categorical, y, data[['Date', 'ticker']]

    def train_model(self, X_num_train, X_cat_train, y_train):
        if self.verbose: print("步骤 4: 训练神经网络用于分类...")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.verbose: print(f"  - 在设备上训练: {self.device}")

        X_num_train_scaled = self.scaler.fit_transform(X_num_train)
        train_dataset = TensorDataset(
            torch.LongTensor(X_cat_train.values),
            torch.FloatTensor(X_num_train_scaled),
            torch.FloatTensor(y_train.values)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        n_unique_tickers = len(self.ticker_to_idx)
        n_numerical_features = X_num_train.shape[1]
        self.model = StockEmbeddingNet(n_unique_tickers, self.embedding_dim, n_numerical_features, self.hidden_size)
        print(f"  - 模型架构: {self.model}\n")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params}")
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
            if self.verbose:
                print(f"  - Epoch {epoch+1}/{self.epochs}, 损失: {total_loss/len(train_loader):.6f}")
        if self.verbose: print("  - 模型训练完成.")

    def generate_signals(self, latest_features_num, latest_features_cat_mapped, latest_features_cat_actual, latest_labels):
        if self.verbose: print("步骤 5: 生成信号 (概率)...")
        X_pred_num_scaled = self.scaler.transform(latest_features_num)
        
        X_pred_cat_tensor = torch.LongTensor(latest_features_cat_mapped.values).to(self.device)
        X_pred_num_tensor = torch.FloatTensor(X_pred_num_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions_tensor = self.model(X_pred_cat_tensor, X_pred_num_tensor).squeeze()
            probabilities = torch.sigmoid(predictions_tensor)
        
        predictions = probabilities.cpu().numpy()
        
        results = latest_features_cat_actual.to_frame(name='ticker').copy()
        results['predicted_probability'] = predictions
        results['is_top_performer'] = latest_labels.values
        ranked_stocks = results.sort_values(by='predicted_probability', ascending=False)
        self.prediction_results = ranked_stocks.set_index('ticker')
        if self.verbose:
            print(f"\n--- 前 {self.top_n_stocks} 名股票 (按成为优等股的概率排名) ---")
            print(self.prediction_results.head(self.top_n_stocks))
            self.prediction_results.to_csv("nn_stock_predictions.csv")
        return ranked_stocks.head(self.top_n_stocks)

    def get_ticker_prediction(self, ticker):
        if self.prediction_results is None: return None, None
        try:
            result_row = self.prediction_results.loc[ticker]
            return result_row['predicted_probability'], result_row['is_top_performer']
        except KeyError:
            print(f"警告: 在给定日期的预测集中未找到股票 '{ticker}'。")
            return None, None

    def run_full_strategy(self, data_pkg, decision_date):
        prices, volumes, fundamentals = data_pkg['prices'], data_pkg['volumes'], data_pkg['fundamentals']
        
        features = self.create_features(prices, volumes, fundamentals)
        labels = self.create_labels(prices)
        
        X_numerical, X_categorical, y, identifiers = self.align_and_prepare_data(features, labels)
        decision_date_dt = pd.to_datetime(decision_date)
        train_mask = identifiers['Date'] < decision_date_dt
        predict_mask = identifiers['Date'] == decision_date_dt
        X_num_train, X_num_predict = X_numerical[train_mask], X_numerical[predict_mask]
        X_cat_train, _ = X_categorical[train_mask], X_categorical[predict_mask]
        y_train, y_predict = y[train_mask], y[predict_mask]
        predict_tickers_actual = identifiers[predict_mask]['ticker']
        unk_idx = self.ticker_to_idx.get('UNK', -1)
        X_cat_predict_mapped = predict_tickers_actual.map(self.ticker_to_idx).fillna(unk_idx).astype(int)
        
        if X_num_train.empty: print("错误: 没有可用的训练数据。"); return
        if X_num_predict.empty: print(f"错误: 决策日期 {decision_date} 没有数据。"); return
        
        self.train_model(X_num_train, X_cat_train, y_train)
        return self.generate_signals(X_num_predict, X_cat_predict_mapped, predict_tickers_actual, y_predict)


if __name__ == '__main__':
    print("Getting S&P 500 ticker list...")
    tickers_to_download = get_spy500_tickers()
    
    START_DATE = "2020-01-01"
    END_DATE = "2025-09-23"
    DECISION_DATE = "2025-09-10"
    
    print("Downloading historical data...")
    data_pkg = download_data(
        tickers_to_download,
        START_DATE,
        END_DATE,
        cache_file="ticker_cache/temp.pkl",
        screen_tickers=True,
        overwrite_cache=False,
        max_workers=20
    )
    
    if data_pkg['prices'].empty:
        print("\nData download failed or is incomplete.")
    else:
        ml_params = {
            'epochs': 30,
            'top_n_stocks': 20,
            'verbose': True,
            'batch_size': 2048,
        }
        strategy = MLStrategy(params=ml_params)
        top_stocks = strategy.run_full_strategy(data_pkg, DECISION_DATE)
        
        if top_stocks is not None:
            print("\n--- Individual Ticker Prediction ---")
            for ticker_to_check in ['MSFT', 'GOOG', 'CRDO', 'NVDA']:
                prediction, true_value = strategy.get_ticker_prediction(ticker_to_check)
                if prediction is not None:
                    print(f"Ticker: {ticker_to_check: <5} | Predicted Prob: {prediction: >7.4f} | Is Top Performer: {true_value: >7.0f}")
