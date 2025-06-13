import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
import time
from tqdm import tqdm  # For progress bar

# =============== CONFIG ===============
MERGED_PARQUET = 'xrp_till_april.parquet'
MODEL_SAVE_PATH = 'xrp_transformer_model.pth'
PREDICT_AHEAD = 5
RETURN_THRESHOLD = 0.001
SEQ_LENGTH = 60  # Lookback period (60 minutes)
NUM_EPOCHS = 10
NUM_FOLDS = 5
BATCH_SIZE = 64

# =============== DATASET ===============
class TradingDataset(Dataset):
    def __init__(self, X, y, seq_length=SEQ_LENGTH):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)
        self.seq_length = seq_length
        print(f"Dataset X shape: {self.X.shape}, y shape: {self.y.shape}")

    def __len__(self):
        return len(self.X) - self.seq_length

    def __getitem__(self, idx):
        return (self.X[idx:idx+self.seq_length], self.y[idx+self.seq_length-1])

# =============== TRANSFORMER MODEL ===============
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_dim = input_dim
        self.input_fc = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=256, dropout=dropout, batch_first=True),
            num_layers=n_layers
        )
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.first_batch = True  # For debug print

    def forward(self, x):
        if self.first_batch:
            print(f"Input shape to input_fc: {x.shape}")
            self.first_batch = False
        x = self.input_fc(x)  # [batch, seq_len, d_model]
        x = self.transformer(x)  # [batch, seq_len, d_model]
        x = x[:, -1, :]  # Take the last time step
        x = self.dropout(x)
        return self.fc(x)

# =============== DATA PREPARATION ===============
def prepare_data(df):
    df = df.copy()
    expected_features = ['1m_open', '1m_high', '1m_low', '1m_close', '1m_volume']
    missing_cols = [col for col in expected_features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
    print(f"DataFrame columns: {df.columns.tolist()}")

    # Normalize raw inputs
    features = expected_features
    for col in features:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    # Target
    raw_target = df['1m_close'].pct_change(PREDICT_AHEAD).shift(-PREDICT_AHEAD)
    df['target'] = np.where(raw_target > RETURN_THRESHOLD, 1, 
                   np.where(raw_target < -RETURN_THRESHOLD, 0, np.nan))
    df = df.dropna(subset=['target'])
    df = df.ffill().dropna()

    model_features = ['1m_open', '1m_high', '1m_low', '1m_volume']  # Exclude 1m_close for model input
    print(f"Model features: {model_features}")
    return df[model_features + ['1m_close']], df['target']

# =============== TRAINING ===============
def train_model(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    start_time = time.time()
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    epoch_time = time.time() - start_time
    return total_loss / len(train_loader), epoch_time

# =============== ETA FORMATTER ===============
def format_eta(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"

# =============== MAIN ===============
if __name__ == "__main__":
    print("📂 Loading sol merged data...")
    df = pd.read_parquet(MERGED_PARQUET).reset_index()
    print(f"Initial DataFrame shape: {df.shape}")
    X, y = prepare_data(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesTransformer(input_dim=len(X.columns)-1, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    tscv = TimeSeriesSplit(n_splits=NUM_FOLDS)
    total_start_time = time.time()
    epoch_times = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"\n📊 Fold {fold + 1}/{NUM_FOLDS}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        train_dataset = TradingDataset(X_train.drop(columns=['1m_close']), y_train)
        test_dataset = TradingDataset(X_test.drop(columns=['1m_close']), y_test)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Reset first_batch flag for each fold
        model.first_batch = True

        # Train
        fold_start_time = time.time()
        for epoch in range(NUM_EPOCHS):
            train_loss, epoch_time = train_model(model, train_loader, criterion, optimizer, device, epoch, NUM_EPOCHS)
            epoch_times.append(epoch_time)

            # Calculate ETA
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = (NUM_EPOCHS - (epoch + 1)) + (NUM_FOLDS - (fold + 1)) * NUM_EPOCHS
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_str = format_eta(eta_seconds)

            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {train_loss:.4f}, ETA: {eta_str}")

        fold_time = time.time() - fold_start_time
        print(f"Fold {fold + 1} completed in {format_eta(fold_time)}")

        # Evaluate
        model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                output = model(X_batch)
                preds.extend(torch.argmax(output, dim=1).cpu().numpy())
                actuals.extend(y_batch.numpy())

        print("Classification Report:")
        print(classification_report(actuals, preds, target_names=['Short', 'Long']))

        # Trade Edge Calculation
        df_test = df.iloc[test_idx].copy()
        df_test['prediction'] = pd.Series(preds, index=X_test.index).map({0: -1, 1: 1})
        df_test['actual_return'] = df_test['1m_close'].pct_change(PREDICT_AHEAD).shift(-PREDICT_AHEAD)

        long_trades = df_test[df_test['prediction'] == 1]
        short_trades = df_test[df_test['prediction'] == -1]

        win_long = sum(long_trades['actual_return'] > RETURN_THRESHOLD) / len(long_trades) if len(long_trades) > 0 else 0
        win_short = sum(short_trades['actual_return'] < -RETURN_THRESHOLD) / len(short_trades) if len(short_trades) > 0 else 0

        avg_win_long = long_trades[long_trades['actual_return'] > 0]['actual_return'].mean() if len(long_trades[long_trades['actual_return'] > 0]) > 0 else 0
        avg_loss_long = long_trades[long_trades['actual_return'] < 0]['actual_return'].mean() if len(long_trades[long_trades['actual_return'] < 0]) > 0 else 0
        edge_long = (avg_win_long - abs(avg_loss_long) - 0.0004) * 10

        avg_win_short = abs(short  _trades[short_trades['actual_return'] < 0]['actual_return'].mean()) if len(short_trades[short_trades['actual_return'] < 0]) > 0 else 0
        avg_loss_short = short_trades[short_trades['actual_return'] > 0]['actual_return'].mean() if len(short_trades[short_trades['actual_return'] > 0]) > 0 else 0
        edge_short = (avg_win_short - avg_loss_short - 0.0004) * 10

        print("Trade Edge Summary (10x Leverage):")
        print(f"🟢 Long Win Rate: {win_long:.2%}")
        print(f"🏆 Avg Win (Long): {avg_win_long*100*10:.3f}%")
        print(f"🔻 Avg Loss (Long): {avg_loss_long*100*10:.3f}%")
        print(f"📊 Net Edge (Long): {edge_long*100:.3f}%")
        print(f"🔴 Short Win Rate: {win_short:.2%}")
        print(f"🏆 Avg Win (Short): {avg_win_short*100*10:.3f}%")
        print(f"🔻 Avg Loss (Short): {avg_loss_short*100*10:.3f}%")
        print(f"📊 Net Edge (Short): {edge_short*100:.3f}%")

    total_time = time.time() - total_start_time
    print(f"\nTraining completed in {format_eta(total_time)}")
    print("\n💾 Saving trained model...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
