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
THRESHOLD_SIGMA = 3  # 异常检测阈值

# 设置 Tushare Token
ts.set_token('7c2fa6010f9586a077b7522951abcd8da1f492e09ac05c1678fb0a4e')
pro = ts.pro_api()

# 获取数据
df_daily = pro.daily(ts_code='000001.SZ', start_date='20220101', end_date='20240101')
df_daily = df_daily.sort_values('trade_date')
df_close = df_daily[['trade_date', 'close']].copy()
df_close.loc[:, 'value'] = df_close['value'] = df_close['close'].astype(float)

# 归一化
scaler = MinMaxScaler()
data = scaler.fit_transform(df_close[['value']].values)

# 创建数据集
def create_dataset(data, days_for_train):
    X, Y = [], []
    for i in range(len(data) - days_for_train):
        X.append(data[i:i+days_for_train])
        Y.append(data[i+days_for_train])
    return np.array(X), np.array(Y)

X, Y = create_dataset(data, DAYS_FOR_TRAIN)
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# ===== 位置编码模块 =====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# ===== 混合模型：LSTM + Transformer（带位置编码） =====
class LSTMTransformerHybrid(nn.Module):
    def __init__(self, input_size=1, lstm_hidden=64, transformer_hidden=64, num_layers=2, nhead=4, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, lstm_hidden, num_layers, batch_first=True)
        self.pos_encoder = PositionalEncoding(lstm_hidden)
        encoder_layer = nn.TransformerEncoderLayer(d_model=lstm_hidden, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(lstm_hidden, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq, lstm_hidden)
        pos_out = self.pos_encoder(lstm_out)  # 加入位置编码
        trans_out = self.transformer(pos_out)  # Transformer建模
        out = self.fc(trans_out[:, -1, :])  # 取最后时间步
        return out

# 初始化模型
model = LSTMTransformerHybrid(input_size=1, lstm_hidden=64, transformer_hidden=64)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ===== 训练 =====
losses = []
for epoch in range(EPOCHS):
    model.train()
    output = model(X_tensor)
    loss = criterion(output, Y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.6f}")

# 保存 Loss 曲线
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.savefig('loss_curve.png')
plt.close()

# ===== 预测 =====
model.eval()
predicted = model(X_tensor).detach().numpy()
predicted_rescaled = scaler.inverse_transform(predicted)
true_rescaled = scaler.inverse_transform(Y_tensor.numpy())

# 可视化预测
plt.plot(true_rescaled, label='True')
plt.plot(predicted_rescaled, label='Predicted')
plt.legend()
plt.title("Stock Price Prediction")
plt.savefig("prediction_result.png")
plt.close()

# ===== 模型评估 =====
def evaluate_performance(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, mae, mape, r2

mse, mae, mape, r2 = evaluate_performance(true_rescaled, predicted_rescaled)
print("\n📊 模型评估结果：")
print(f"MSE  : {mse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"MAPE : {mape:.2f}%")
print(f"R²   : {r2:.4f}")

# ===== 异常检测 =====
residuals = true_rescaled - predicted_rescaled
mean_residual = residuals.mean()
std_residual = residuals.std()
threshold_up = mean_residual + THRESHOLD_SIGMA * std_residual
threshold_down = mean_residual - THRESHOLD_SIGMA * std_residual
anomalies = (residuals < threshold_down) | (residuals > threshold_up)

# 保存结果
dates = df_close['trade_date'].values[DAYS_FOR_TRAIN:]
result_df = pd.DataFrame({
    'date': dates[-len(true_rescaled):],
    'true': true_rescaled.flatten(),
    'predicted': predicted_rescaled.flatten(),
    'residual': residuals.flatten(),
    'is_anomaly': anomalies.flatten().astype(int)
})
result_df.to_csv("anomaly_detection_result.csv", index=False)

# 可视化异常点
anomaly_indices = np.where(result_df['is_anomaly'] == 1)[0]
plt.figure(figsize=(10, 4))
plt.plot(result_df['true'], label='True Price')
plt.plot(result_df['predicted'], label='Predicted Price')
plt.scatter(anomaly_indices, result_df['true'].iloc[anomaly_indices], color='red', label='Anomaly', marker='x')
plt.title("Stock Price with Detected Anomalies")
plt.legend()
plt.savefig("anomaly_visualization.png")
plt.close()

print("\n✅ 混合模型（LSTM+Transformer+位置编码）已训练并完成异常检测，保存文件如下：")
print(" - loss_curve.png")
print(" - prediction_result.png")
print(" - anomaly_detection_result.csv")
print(" - anomaly_visualization.png")
