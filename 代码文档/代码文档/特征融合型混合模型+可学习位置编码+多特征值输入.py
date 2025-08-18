import os
import random
import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------- 配置参数 --------------------
SEED = 42
DAYS_FOR_TRAIN = 10
EPOCHS = 200
LR = 1e-3
THRESHOLD_SIGMA = 3
LSTM_HIDDEN = 64
TRANS_D_MODEL = 64
TRANS_HEADS = 4
TRANS_LAYERS = 2
BATCH_FIRST = True
OUT_DIR = '.'
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------- 设置随机种子 --------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

# -------------------- 获取多特征数据 --------------------
ts.set_token('7c2fa6010f9586a077b7522951abcd8da1f492e09ac05c1678fb0a4e')  # 换成你的 token
pro = ts.pro_api()
df_daily = pro.daily(ts_code='000001.SZ', start_date='20220101', end_date='20240101')
df_daily = df_daily.sort_values('trade_date').reset_index(drop=True)

# 选择多特征：收盘价、成交量、最高价、最低价
features = ['close', 'vol', 'high', 'low']
df_features = df_daily[['trade_date'] + features].copy()
df_features[features] = df_features[features].astype(float)

# -------------------- 数据归一化 --------------------
scaler = MinMaxScaler()
data_values = scaler.fit_transform(df_features[features].values)

def create_dataset(data, window):
    X, Y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])  # shape: [window, features]
        Y.append(data[i+window, 0]) # 预测收盘价
    return np.array(X), np.array(Y)

X, Y = create_dataset(data_values, DAYS_FOR_TRAIN)
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)
dates = df_features['trade_date'].values[DAYS_FOR_TRAIN:]

# -------------------- 评估函数 --------------------
def evaluate_metrics(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mask = (np.abs(y_true) > 1e-8).astype(float)
    mape = (np.mean(np.abs((y_true - y_pred) / (np.where(mask, y_true, 1.0)))) * 100)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, mape, r2

# -------------------- 模型结构 --------------------
class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)
    def forward(self, x):
        return x + self.pos_embed[:, :x.size(1), :]

class FusionLSTMTransformer(nn.Module):
    def __init__(self, input_dim, lstm_hidden=LSTM_HIDDEN, trans_d=TRANS_D_MODEL,
                 trans_heads=TRANS_HEADS, trans_layers=TRANS_LAYERS, seq_len=DAYS_FOR_TRAIN):
        super().__init__()
        # LSTM 分支
        self.lstm = nn.LSTM(input_dim, lstm_hidden, num_layers=2, batch_first=BATCH_FIRST)
        # Transformer 分支
        self.input_proj = nn.Linear(input_dim, trans_d)
        self.pos_emb = LearnablePositionalEmbedding(seq_len, trans_d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=trans_d, nhead=trans_heads,
            dim_feedforward=trans_d * 4, batch_first=BATCH_FIRST
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)
        # 特征融合
        self.fc_fusion = nn.Sequential(
            nn.Linear(lstm_hidden + trans_d, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_last = lstm_out[:, -1, :]
        t = self.input_proj(x)
        t = self.pos_emb(t)
        t_out = self.transformer(t)
        t_last = t_out[:, -1, :]
        fused = torch.cat([lstm_last, t_last], dim=-1)
        out = self.fc_fusion(fused)
        return out

# -------------------- 训练混合模型 --------------------
input_dim = len(features)
model = FusionLSTMTransformer(input_dim=input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
losses = []

for epoch in range(EPOCHS):
    model.train()
    pred = model(X_tensor)
    loss = criterion(pred, Y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch + 1) % 20 == 0:
        print(f"[Fusion] Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Fusion Model Training Loss (Multi-feature)")
plt.savefig(os.path.join(OUT_DIR, "loss_fusion_multi.png"))
plt.close()

# -------------------- 预测与评估 --------------------
model.eval()
with torch.no_grad():
    y_pred = model(X_tensor).cpu().numpy()
y_pred_inv = scaler.inverse_transform(np.hstack([y_pred, np.zeros((len(y_pred), input_dim - 1))]))[:, 0]
y_true_inv = scaler.inverse_transform(np.hstack([Y_tensor.numpy(), np.zeros((len(Y_tensor), input_dim - 1))]))[:, 0]

mse, mae, mape, r2 = evaluate_metrics(y_true_inv, y_pred_inv)
print("\n📈 Fusion Model Evaluation (Multi-feature):")
print(f" - MSE  : {mse:.6f}")
print(f" - MAE  : {mae:.6f}")
print(f" - MAPE : {mape:.4f}%")
print(f" - R²   : {r2:.6f}")

plt.figure(figsize=(12,4))
plt.plot(y_true_inv, label='True')
plt.plot(y_pred_inv, label='Predicted')
plt.title("Fusion Model Prediction (Multi-feature)")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "prediction_fusion_multi.png"))
plt.close()

# -------------------- 异常检测 --------------------
residuals = y_true_inv - y_pred_inv
mean_r, std_r = residuals.mean(), residuals.std()
upper, lower = mean_r + THRESHOLD_SIGMA * std_r, mean_r - THRESHOLD_SIGMA * std_r
anoms = ((residuals > upper) | (residuals < lower)).astype(int).flatten()

df_out = pd.DataFrame({
    "date": dates,
    "true": y_true_inv.flatten(),
    "predicted": y_pred_inv.flatten(),
    "residual": residuals.flatten(),
    "is_anomaly": anoms
})
df_out.to_csv(os.path.join(OUT_DIR, "anomaly_result_fusion_multi.csv"), index=False)

plt.figure(figsize=(12,4))
plt.plot(df_out['true'], label='True')
plt.plot(df_out['predicted'], label='Predicted')
plt.scatter(np.where(df_out['is_anomaly'] == 1)[0],
            df_out['true'].values[df_out['is_anomaly'] == 1],
            color='red', marker='x', label='Anomaly')
plt.title("Fusion Model Detected Anomalies (Multi-feature)")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "anomaly_visualization_fusion_multi.png"))
plt.close()

print("\n✅ 多特征融合模型训练、预测与异常检测完成！文件保存在：", os.path.abspath(OUT_DIR))
