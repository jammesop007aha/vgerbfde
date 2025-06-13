import ccxt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests, coint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import talib
import joblib
import warnings
import datetime
import time

# Suppress FutureWarning from statsmodels
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")

# Initialize Bybit exchange
exchange = ccxt.bybit({'enableRateLimit': True})

# Define symbols and timeframe
symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LINKUSDT', 'XRPUSDT', 'TONUSDT']
timeframe = '5m'
limit = 1000
since = exchange.parse8601('2025-05-15T00:00:00Z')  # 30 days back

# Fetch OHLCV data
def fetch_ohlcv(symbol, timeframe, since, limit):
    all_data = []
    while True:
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            if not data:
                break
            all_data += data
            since = data[-1][0] + 1
            time.sleep(0.1)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            break
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].set_index('timestamp')

# Fetch funding rates
def fetch_funding_rates(symbol, since, limit):
    try:
        funding = exchange.fetch_funding_rate_history(symbol, since=since, limit=limit)
        df = pd.DataFrame(funding)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['timestamp', 'fundingRate']].set_index('timestamp')
        return df
    except Exception as e:
        print(f"Error fetching funding rates for {symbol}: {e}")
        return pd.DataFrame()

# Collect and process data
dfs = []
for symbol in symbols:
    df = fetch_ohlcv(symbol, timeframe, since, limit)
    funding_df = fetch_funding_rates(symbol, since, limit)
    # Calculate indicators
    df[f'{symbol}_rsi'] = talib.RSI(df['close'], timeperiod=14)
    df[f'{symbol}_macd'], df[f'{symbol}_macd_signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df[f'{symbol}_bb_upper'], df[f'{symbol}_bb_middle'], df[f'{symbol}_bb_lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df[f'{symbol}_atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df[f'{symbol}_obv'] = talib.OBV(df['close'], df['volume'])
    df[f'{symbol}_slowk'], df[f'{symbol}_slowd'] = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3)
    df[f'{symbol}_vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df[f'{symbol}_momentum'] = talib.MOM(df['close'], timeperiod=10)
    df[f'{symbol}_spread'] = df['high'] - df['low']
    # Merge funding rates
    if not funding_df.empty:
        df = df.join(funding_df, how='left').fillna(method='ffill')
        df.rename(columns={'fundingRate': f'{symbol}_funding_rate'}, inplace=True)
    else:
        df[f'{symbol}_funding_rate'] = np.nan
    df = df[['close', 'volume', f'{symbol}_rsi', f'{symbol}_macd', f'{symbol}_macd_signal', 
             f'{symbol}_bb_upper', f'{symbol}_bb_middle', f'{symbol}_bb_lower', f'{symbol}_atr', 
             f'{symbol}_obv', f'{symbol}_slowk', f'{symbol}_slowd', f'{symbol}_vwap', 
             f'{symbol}_momentum', f'{symbol}_spread', f'{symbol}_funding_rate']]
    df.columns = [f"{symbol}_close", f"{symbol}_volume", f"{symbol}_rsi", f"{symbol}_macd", 
                  f"{symbol}_macd_signal", f"{symbol}_bb_upper", f"{symbol}_bb_middle", 
                  f"{symbol}_bb_lower", f"{symbol}_atr", f"{symbol}_obv", f"{symbol}_slowk", 
                  f"{symbol}_slowd", f"{symbol}_vwap", f"{symbol}_momentum", f"{symbol}_spread", 
                  f"{symbol}_funding_rate"]
    dfs.append(df)
    print(f"Fetched and processed data for {symbol}")

# Merge dataframes
df = dfs[0]
for d in dfs[1:]:
    df = df.join(d, how='inner')
df.dropna(inplace=True)

# Save data to Parquet
df.to_parquet('many_data.parquet')
print("Data saved to 'many_data.parquet'")

# Calculate percentage returns and volatility
returns = df[[f"{symbol}_close" for symbol in symbols]].pct_change().dropna()
volatility = returns.std() * np.sqrt(12 * 24 * 365)  # Annualized volatility (12 candles/hour)

# Pair-wise analysis
corr_results = {}
granger_results = {}
coint_results = {}
cross_corr_results = {}
max_lag = 6  # 30 minutes
for i, lead in enumerate(symbols):
    for follow in symbols[i+1:]:
        pair = f"{lead} vs {follow}"
        # Correlation
        corr, _ = pearsonr(returns[f"{lead}_close"], returns[f"{follow}_close"])
        corr_results[pair] = corr
        # Granger causality
        try:
            test_result = grangercausalitytests(
                returns[[f"{lead}_close", f"{follow}_close"]], maxlag=max_lag, verbose=False
            )
            p_values = [test_result[i+1][0]['ssr_ftest'][1] for i in range(max_lag)]
            min_p = min(p_values)
            if min_p < 0.05:
                granger_results[f"{lead} -> {follow}"] = min_p
            test_result = grangercausalitytests(
                returns[[f"{follow}_close", f"{lead}_close"]], maxlag=max_lag, verbose=False
            )
            p_values = [test_result[i+1][0]['ssr_ftest'][1] for i in range(max_lag)]
            min_p = min(p_values)
            if min_p < 0.05:
                granger_results[f"{follow} -> {lead}"] = min_p
        except:
            continue
        # Cointegration
        try:
            _, p_value, _ = coint(df[f"{lead}_close"], df[f"{follow}_close"])
            if p_value < 0.05:
                coint_results[pair] = p_value
        except:
            continue
        # Cross-correlation
        cross_corr = []
        for lag in range(-max_lag, max_lag+1):
            if lag < 0:
                corr = returns[f"{lead}_close"].corr(returns[f"{follow}_close"].shift(-lag))
            else:
                corr = returns[f"{lead}_close"].shift(-lag).corr(returns[f"{follow}_close"])
            cross_corr.append((lag, corr))
        max_corr = max(cross_corr, key=lambda x: abs(x[1]), default=(0, 0))
        if abs(max_corr[1]) > 0.3:  # Significant correlation threshold
            cross_corr_results[pair] = max_corr

# Train and save model for each coin
models = {}
for target_symbol in symbols:
    # Prepare features
    lagged_dfs = []
    for symbol in symbols:
        temp_df = pd.DataFrame()
        for lag in range(1, 7):
            temp_df[f"{symbol}_close_lag{lag}"] = df[f"{symbol}_close"].shift(lag)
            temp_df[f"{symbol}_volume_lag{lag}"] = df[f"{symbol}_volume"].shift(lag)
            temp_df[f"{symbol}_rsi_lag{lag}"] = df[f"{symbol}_rsi"].shift(lag)
            temp_df[f"{symbol}_macd_lag{lag}"] = df[f"{symbol}_macd"].shift(lag)
            temp_df[f"{symbol}_atr_lag{lag}"] = df[f"{symbol}_atr"].shift(lag)
            temp_df[f"{symbol}_obv_lag{lag}"] = df[f"{symbol}_obv"].shift(lag)
            temp_df[f"{symbol}_slowk_lag{lag}"] = df[f"{symbol}_slowk"].shift(lag)
            temp_df[f"{symbol}_vwap_lag{lag}"] = df[f"{symbol}_vwap"].shift(lag)
            temp_df[f"{symbol}_momentum_lag{lag}"] = df[f"{symbol}_momentum"].shift(lag)
            temp_df[f"{symbol}_spread_lag{lag}"] = df[f"{symbol}_spread"].shift(lag)
            temp_df[f"{symbol}_funding_rate_lag{lag}"] = df[f"{symbol}_funding_rate"].shift(lag)
            temp_df[f"{symbol}_bb_width_lag{lag}"] = (df[f"{symbol}_bb_upper"] - df[f"{symbol}_bb_lower"]).shift(lag)
        lagged_dfs.append(temp_df)
    X = pd.concat(lagged_dfs, axis=1)
    X.dropna(inplace=True)
    # Target: Price direction
    y = (returns[f"{target_symbol}_close"] > 0).astype(int).iloc[max_lag:]
    X = X.iloc[max_lag:]
    y = y[X.index]
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    models[target_symbol] = {
        'model': model,
        'accuracy': accuracy,
        'top_features': feature_importance.head().to_dict()
    }
    # Save model
    joblib.dump(model, f'rf_model_{target_symbol}.pkl')
    print(f"Trained and saved model for {target_symbol}")

# Mean reversion and divergence
mean_reversion_signals = {}
divergence_signals = {}
for i, coin1 in enumerate(symbols):
    for coin2 in symbols[i+1:]:
        pair = f"{coin1} vs {coin2}"
        price_diff = df[f"{coin1}_close"] - df[f"{coin2}_close"].mean()
        z_score = (price_diff - price_diff.mean()) / price_diff.std()
        if z_score.abs().max() > 2:
            mean_reversion_signals[pair] = z_score.abs().max()
        if corr_results.get(pair, 0) < 0:
            divergence_signals[pair] = corr_results[pair]

# Save results to text file
with open('crypto_relations.txt', 'w') as f:
    f.write("Cryptocurrency Pair-wise Relationship Analysis\n")
    f.write("=======================================\n\n")
    
    f.write("Volatility (Annualized, 5-min Returns):\n")
    for symbol, vol in volatility.items():
        f.write(f"{symbol}: {vol:.4f}\n")
    f.write("\n")
    
    f.write("Pair-wise Analysis:\n")
    for pair in corr_results:
        f.write(f"\n{pair}:\n")
        f.write(f"Correlation: {corr_results[pair]:.4f}\n")
        lead = pair.split(' vs ')[0]
        follow = pair.split(' vs ')[1]
        if f"{lead} -> {follow}" in granger_results:
            f.write(f"Granger Causality ({lead} -> {follow}): p-value = {granger_results[f'{lead} -> {follow}']:.4f}\n")
        if f"{follow} -> {lead}" in granger_results:
            f.write(f"Granger Causality ({follow} -> {lead}): p-value = {granger_results[f'{follow} -> {lead}']:.4f}\n")
        if pair in coint_results:
            f.write(f"Cointegration: p-value = {coint_results[pair]:.4f}\n")
        if pair in cross_corr_results:
            lag, corr = cross_corr_results[pair]
            f.write(f"Max Cross-Correlation: Lag = {lag*5} minutes, Correlation = {corr:.4f}\n")
        if pair in mean_reversion_signals:
            f.write(f"Mean Reversion Opportunity: Max Z-score = {mean_reversion_signals[pair]:.2f}\n")
        if pair in divergence_signals:
            f.write(f"Divergence Opportunity: Correlation = {divergence_signals[pair]:.4f}\n")
        f.write("Trading Strategy Implications:\n")
        if corr_results[pair] > 0.7:
            f.write("- High correlation: Consider trend-following or momentum strategies.\n")
        if pair in coint_results:
            f.write("- Cointegrated: Explore pairs trading or mean-reversion strategies.\n")
        if f"{lead} -> {follow}" in granger_results or f"{follow} -> {lead}" in granger_results:
            f.write("- Lead-lag detected: Monitor leading coin for trade signals.\n")
    
    f.write("\nModel Performance (Predicting Price Direction):\n")
    for symbol, info in models.items():
        f.write(f"\n{symbol}:\n")
        f.write(f"Test Accuracy: {info['accuracy']:.4f}\n")
        f.write("Top 5 Influential Features:\n")
        for feat, imp in info['top_features'].items():
            f.write(f"{feat}: {imp:.4f}\n")

print("Analysis complete. Data saved to 'many_data.parquet'. Models saved as 'rf_model_*.pkl'. Results saved to 'crypto_relations.txt'.")
