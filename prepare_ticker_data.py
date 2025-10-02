import time
import os
import logging
import pickle
import requests
import pandas as pd
import yfinance as yf
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from fredapi import Fred 

logger = logging.getLogger(__name__)

FRED_API_KEY = "838d43a27bb06ef4f1495b2f0f92f434"
def get_macro_data(start_date, end_date, api_key=None):
    logger.info("正在下载宏观经济数据...")
    try:
        fred = Fred(api_key=api_key or os.getenv('FRED_API_KEY'))
        # 指标代码: VIXCLS(VIX指数), DGS10(10年期国债收益率), T10Y2Y(利差), CPIAUCSL(CPI)
        series_ids = {
            'VIX': 'VIXCLS',
            'DGS10': 'DGS10',
            'T10Y2Y': 'T10Y2Y',
            'CPI': 'CPIAUCSL',
        }
        
        df_list = []
        for name, series_id in series_ids.items():
            s = fred.get_series(series_id, start_date, end_date)
            s.name = name
            df_list.append(s)
            
        macro_df = pd.concat(df_list, axis=1)
        macro_df = macro_df.ffill().dropna() 
        logger.info("宏观经济数据下载完成。")
        return macro_df
    except Exception as e:
        logger.info(f"下载宏观数据失败: {e}. 请检查你的FRED API Key。")
        return pd.DataFrame()
    
def get_sector_data(tickers):
    logger.info(f"正在为 {len(tickers)} 只股票获取行业数据...")
    sectors = {}
    ticker_objects = yf.Tickers(tickers)
    
    for ticker_str, ticker_obj in tqdm(ticker_objects.tickers.items(), desc="获取行业信息"):
        try:
            info = ticker_obj.info
            if info and 'sector' in info:
                sectors[ticker_str] = info['sector']
            else:
                sectors[ticker_str] = 'Unknown'
        except Exception:
            sectors[ticker_str] = 'Unknown'
            
    logger.info("行业数据获取完成。")
    return pd.Series(sectors, name="sector")

def get_spy500_tickers():
    url = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    spy_tickers = df["Symbol"].tolist()
    return spy_tickers

def get_all_us_tickers(cache_file='raw_tickers.pkl', max_age_hours=24):
    if os.path.exists(cache_file):
        file_mod_time = os.path.getmtime(cache_file)
        if (time.time() - file_mod_time) / 3600 < max_age_hours:
            logger.info(f"Loading raw tickers from recent cache ('{cache_file}')...")
            df = pd.read_pickle(cache_file)
            logger.info(f"Successfully loaded {len(df)} raw tickers from cache.")
            return df['ticker'].tolist()

    logger.info("Fetching raw ticker list from GitHub...")
    raw_tickers = []
    try:
        url = 'https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt'
        df_raw = pd.read_csv(url, header=0)
        raw_tickers = df_raw.iloc[:, 0].dropna().tolist()
        
        pd.DataFrame(raw_tickers, columns=['ticker']).to_pickle(cache_file)
        logger.info(f"Fetched and cached {len(raw_tickers)} raw symbols.")
    except Exception as e:
        logger.info(f"Failed to fetch raw tickers from GitHub: {e}")
    
    spy_tickers = get_spy500_tickers()
    logger.info(f"Loaded SPY {len(spy_tickers)} tickers from GitHub")
    spy500_set = set(spy_tickers)
    sorted_tickers = sorted(set(raw_tickers).union(spy500_set))
    return sorted_tickers

def download_data(tickers, start_date, end_date, cache_file='data_cache_pure_score.pkl',
                  max_workers=20, screen_tickers=False, overwrite_cache=False):
    # 防止yfinance报错刷屏
    yf_logger = logging.getLogger('yfinance')
    yf_logger.setLevel(logging.CRITICAL)

    if os.path.exists(cache_file) and not overwrite_cache:
        logger.info(f"从缓存 ('{cache_file}') 加载数据...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    if overwrite_cache and os.path.exists(cache_file):
        logger.info("overwrite_cache=True。强制重新下载。")

    info_cache = {}
    failed_tickers = []

    def get_ticker_info(ticker):
        if ticker in info_cache:
            return info_cache[ticker]
        for _ in range(3):
            try:
                info = yf.Ticker(ticker).get_info()  
                if info:
                    info_cache[ticker] = info
                    return info
            except Exception:
                time.sleep(1)
        failed_tickers.append(ticker)
        return None

    # --- 筛选股票，过滤仙股 ---
    if screen_tickers:
        logger.info(f"正在筛选 {len(tickers)} 个原始代码...")
        screened = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(get_ticker_info, t): t for t in tickers}
            for future in tqdm(as_completed(futures), total=len(tickers), desc="筛选中"):
                info = future.result()
                if info and info.get('quoteType') == 'EQUITY' and info.get('marketCap', 0) > 50_000_000:
                    screened.append(futures[future])
        tickers = sorted(screened)
        logger.info(f"筛选完成。找到 {len(tickers)} 个有效股票。")

    # --- 批量下载价格数据 ---
    logger.info(f"正在为 {len(tickers)} 只股票下载价格数据...")
    all_data = yf.download(tickers + ['SPY'], start=start_date, end=end_date,
                           auto_adjust=True, threads=True, progress=True)
    all_prices = all_data['Close']
    all_volumes = all_data['Volume']
    logger.info("价格和交易量数据下载完成。")
    macro_data = get_macro_data(start_date, end_date, api_key=FRED_API_KEY)
    sector_data = get_sector_data(tickers)

    # --- 获取基本面数据 ---
    # fundamental_data = {}
    # logger.info("正在下载基本面数据...")
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     futures = {executor.submit(get_ticker_info, t): t for t in tickers}
    #     for future in tqdm(as_completed(futures), total=len(tickers), desc="基本面下载"):
    #         info = future.result()
    #         if info and info.get('marketCap'):
    #             fundamental_data[futures[future]] = {
    #                 'debtToEquity': info.get('debtToEquity'),
    #                 'trailingPE': info.get('trailingPE'),
    #                 'priceToSalesTrailing12Months': info.get('priceToSalesTrailing12Months'),
    #                 'returnOnEquity': info.get('returnOnEquity'),
    #                 'revenueGrowth': info.get('revenueGrowth'),
    #                 'marketCap': info.get('marketCap'),
    #                 'sharesOutstanding': info.get('sharesOutstanding'),
    #                 'shortRatio': info.get('shortRatio'),
    #                 'shortPercentOfFloat': info.get('shortPercentOfFloat') 
    #             }
    # logger.info("基本面数据下载完成。")

    yf_logger.setLevel(logging.INFO)

    # --- 打印失败 SP500 成分股 ---
    if failed_tickers:
        SP500_TICKERS = set(get_spy500_tickers())
        unique_failed = sorted(set(failed_tickers))
        sp500_failed = sorted(set(unique_failed) & SP500_TICKERS)

        logger.info(f"⚠️ 共 {len(unique_failed)} 个 ticker 获取失败")
        if sp500_failed:
            logger.info(f"其中 {len(sp500_failed)} 个在 S&P500，占比 {len(sp500_failed)/len(SP500_TICKERS):.2%}")
            logger.info("失败的 S&P500 tickers:", sp500_failed)
        else:
            logger.info("✅ 没有 S&P500 成分股失败")
    else:
        logger.info("✅ 所有 ticker 数据获取成功！")

    # --- 整理结果并缓存 ---
    data_package = {
        "prices": all_prices.dropna(axis=1, how='all'),
        "volumes": all_volumes.dropna(axis=1, how='all'),
        # "fundamentals": pd.DataFrame.from_dict(fundamental_data, orient='index'),
        "macro": macro_data,
        "sectors": sector_data
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(data_package, f)
    logger.info("数据下载并成功缓存。")

    return data_package

def slice_data_package(data_package, end_date):
    logger.info(f"\n数据已从缓存加载，现在将其截取到决策日: {end_date}...")
    
    end_date_dt = pd.to_datetime(end_date)
    
    sliced_prices = data_package['prices'][data_package['prices'].index <= end_date_dt]
    sliced_volumes = data_package['volumes'][data_package['volumes'].index <= end_date_dt]
    
    sliced_data_package = {
        "prices": sliced_prices,
        "volumes": sliced_volumes,
        "fundamentals": data_package['fundamentals'] 
    }
    
    return sliced_data_package