import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 配置参数
DAYS_FOR_TRAIN = 10
EPOCHS = 200
THRESHOLD_SIGMA = 3

# 设置 Tushare Token
ts.set_token('7c2fa6010f9586a077b7522951abcd8da1f492e09ac05c1678fb0a4e')
pro = ts.pro_api()

# 获取股票数据
df_daily = pro.daily(ts_code='000001.SZ', start_date='20220101', end_date='20240101')
df_daily = df_daily.sort_values('trade_date')
df_close = df_daily[['trade_date', 'close']].rename(columns={'close': 'value'})
df_close['value'] = df_close['value'].astype(float)

# 数据归一化
scaler = MinMaxScaler()
data = scaler.fit_transform(df_close[['value']])

# 构造训练集
def create_dataset(data, days=10):
    X, Y = [], []
    for i in range(len(data) - days):
        X.append(data[i:i+days])
        Y.append(data[i+days])
    return np.array(X), np.array(Y)

X, Y = create_dataset(data, DAYS_FOR_TRAIN)
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# ========== LSTM 模型 ==========
class LSTM_Regression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

lstm_model = LSTM_Regression(1, 64)
lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# 训练 LSTM
losses = []
for epoch in range(EPOCHS):
    lstm_model.train()
    out = lstm_model(X_tensor)
    loss = loss_fn(out, Y_tensor)
    lstm_optimizer.zero_grad()
    loss.backward()
    lstm_optimizer.step()
    losses.append(loss.item())

# 绘制 Loss 曲线
plt.plot(losses)
plt.title("LSTM Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss_curve.png")
plt.close()

# LSTM 预测 + 反归一化
lstm_model.eval()
lstm_pred = lstm_model(X_tensor).detach().numpy()
lstm_pred_inv = scaler.inverse_transform(lstm_pred)
true_inv = scaler.inverse_transform(Y_tensor.numpy())

# ========== Transformer 模型 ==========
class TransformerAnomalyDetector(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

transformer_model = TransformerAnomalyDetector()
transformer_optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.001)

# 训练 Transformer
for epoch in range(EPOCHS):
    transformer_model.train()
    pred = transformer_model(X_tensor)
    loss = loss_fn(pred, Y_tensor)
    transformer_optimizer.zero_grad()
    loss.backward()
    transformer_optimizer.step()

# Transformer 预测
transformer_model.eval()
with torch.no_grad():
    tf_pred = transformer_model(X_tensor).numpy()
    tf_pred_inv = scaler.inverse_transform(tf_pred)

# ========== 评估函数 ==========
def evaluate(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"📊 {name} 模型评估:")
    print(f" - MSE  : {mse:.4f}")
    print(f" - MAE  : {mae:.4f}")
    print(f" - MAPE : {mape:.2f}%")
    print(f" - R²   : {r2:.4f}\n")
    return mse, mae, mape, r2

evaluate(true_inv, lstm_pred_inv, "LSTM")
evaluate(true_inv, tf_pred_inv, "Transformer")

# ========== 异常检测与保存 ==========
def detect_anomaly(pred_inv, model_name):
    residual = true_inv - pred_inv
    mean_r = residual.mean()
    std_r = residual.std()
    up = mean_r + THRESHOLD_SIGMA * std_r
    down = mean_r - THRESHOLD_SIGMA * std_r
    anomaly = (residual < down) | (residual > up)

    result = pd.DataFrame({
        'date': df_close['trade_date'].values[DAYS_FOR_TRAIN:],
        'true': true_inv.flatten(),
        'predicted': pred_inv.flatten(),
        'residual': residual.flatten(),
        'is_anomaly': anomaly.flatten().astype(int)
    })

    csv_name = f"anomaly_detection_result_{model_name}.csv"
    img_name = f"anomaly_visualization_{model_name}.png"
    result.to_csv(csv_name, index=False)

    # 可视化
    plt.figure(figsize=(10, 4))
    plt.plot(result['true'], label='True')
    plt.plot(result['predicted'], label='Predicted')
    plt.scatter(result.index[result['is_anomaly'] == 1],
                result['true'][result['is_anomaly'] == 1],
                color='red', label='Anomaly', marker='x')
    plt.title(f"Detected Anomalies - {model_name}")
    plt.legend()
    plt.savefig(img_name)
    plt.close()

    print(f"✅ {model_name} 异常检测完成，已保存：{csv_name}, {img_name}")

detect_anomaly(lstm_pred_inv, "lstm")
detect_anomaly(tf_pred_inv, "transformer")

print("📁 所有任务已完成，所有图像和CSV文件已保存。")
