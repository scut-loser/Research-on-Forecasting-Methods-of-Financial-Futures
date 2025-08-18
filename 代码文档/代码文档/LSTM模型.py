import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import datetime
import os

# 配置参数
DAYS_FOR_TRAIN = 10
EPOCHS = 200
THRESHOLD_SIGMA = 3  # 异常检测阈值：残差均值 ± 3σ

# 设置 Tushare Token
ts.set_token('7c2fa6010f9586a077b7522951abcd8da1f492e09ac05c1678fb0a4e')  # 请替换为你的 token
pro = ts.pro_api()

# 获取数据
df_daily = pro.daily(ts_code='000001.SZ', start_date='20220101', end_date='20240101')
df_daily = df_daily.sort_values('trade_date')
df_close = df_daily[['trade_date', 'close']]
df_close = df_close.rename(columns={'close': 'value'})
df_close['value'] = df_close['value'].astype(float)

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

# 转换为 PyTorch tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# 定义模型结构
class LSTM_Regression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

model = LSTM_Regression(input_size=1, hidden_size=64)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
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

# 绘制 loss 曲线
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.savefig('loss_curve.png')
plt.close()

# 预测
model.eval()
predicted = model(X_tensor).detach().numpy()
predicted_rescaled = scaler.inverse_transform(predicted)
true_rescaled = scaler.inverse_transform(Y_tensor.numpy())

# 可视化预测结果
plt.plot(true_rescaled, label='True')
plt.plot(predicted_rescaled, label='Predicted')
plt.legend()
plt.title("Stock Price Prediction")
plt.savefig("prediction_result.png")
plt.close()

# 🔍 异常检测（基于预测残差）
residuals = true_rescaled - predicted_rescaled
mean_residual = residuals.mean()
std_residual = residuals.std()
threshold_up = mean_residual + THRESHOLD_SIGMA * std_residual
threshold_down = mean_residual - THRESHOLD_SIGMA * std_residual

anomalies = (residuals < threshold_down) | (residuals > threshold_up)
# 保存异常检测结果
dates = df_close['trade_date'].values[DAYS_FOR_TRAIN:]
result_df = pd.DataFrame({
    'date': dates[-len(true_rescaled):],
    'true': true_rescaled.flatten(),
    'predicted': predicted_rescaled.flatten(),
    'residual': residuals.flatten(),
    'is_anomaly': anomalies.flatten().astype(int)
})
result_df.to_csv("anomaly_detection_result.csv", index=False)

print("✅ 模型训练、预测与异常检测完成！已保存：")
print(" - loss_curve.png")
print(" - prediction_result.png")
print(" - anomaly_detection_result.csv")
