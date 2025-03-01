
import pandas as pd

df = pd.read_csv("/content/AAPL.csv")
df

df.info()

df.columns

df.head()

df['Date'] = pd.to_datetime(df['Date'])

df.set_index('Date',inplace=True)

df.describe()

import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use('seaborn-v0_8-darkgrid')


plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Close'], label='Closing Price', color='blue', linewidth=2)
plt.title("AAPL Closing Price Over Time", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Closing Price (USD)", fontsize=12)
plt.legend()
plt.show()

df['MA_50'] = df['Close'].rolling(window=50).mean()
df['MA_200'] = df['Close'].rolling(window=200).mean()

plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Close'], label='Closing Price', color='blue', alpha=0.6)
plt.plot(df.index, df['MA_50'], label='50-day MA', color='red', linestyle='dashed')
plt.plot(df.index, df['MA_200'], label='200-day MA', color='green', linestyle='dashed')
plt.title("AAPL Closing Price with Moving Averages", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Price (USD)", fontsize=12)
plt.legend()
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Volume'], label='Trading Volume', color='purple', alpha=0.6)
plt.title("AAPL Trading Volume Over Time", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Volume (in billions)", fontsize=12)
plt.legend()
plt.show()

correlation_matrix = df.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Stock Price Variables", fontsize=14)
plt.show()

df[['Open', 'High', 'Low', 'Close', 'Adj Close']].applymap(lambda x: pd.to_numeric(x, errors='coerce')).isna().sum()


sns.pairplot(df[['Open', 'High', 'Low', 'Close', 'Adj Close']], diag_kind='hist', corner=True)
plt.suptitle("Pairplot of Stock Price Variables", fontsize=14, y=1.02)
plt.show()

df['Daily_Return'] = df['Close'].pct_change()

plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Daily_Return'], label="Daily Returns", color='blue', alpha=0.6)
plt.axhline(0, color='black', linestyle='dashed', linewidth=1)
plt.title("AAPL Daily Returns", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Daily Return", fontsize=12)
plt.legend()
plt.show()

df['Volatility'] = df['Daily_Return'].rolling(window=30).std()

plt.figure(figsize=(14, 6))
plt.plot(df.index, df['Volatility'], label="30-Day Rolling Volatility", color='red', alpha=0.8)
plt.title("AAPL Stock Volatility (Rolling 30-Day Std Dev)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Volatility", fontsize=12)
plt.legend()
plt.show()


from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df['Close'], model='multiplicative', period=252)


plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(df['Close'], label="Original", color='blue')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(decomposition.trend, label="Trend", color='red')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(decomposition.seasonal, label="Seasonality", color='green')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(decomposition.resid, label="Residuals", color='purple')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(df['Daily_Return'].dropna(), bins=50, kde=True, color='blue', alpha=0.6)
plt.axvline(df['Daily_Return'].mean(), color='red', linestyle='dashed', linewidth=2, label='Mean Return')
plt.title("Distribution of AAPL Daily Returns", fontsize=14)
plt.xlabel("Daily Return", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()
plt.show()


from statsmodels.graphics.tsaplots import plot_acf
plt.figure(figsize=(12, 5))
plot_acf(df['Close'], lags=50, alpha=0.05)  # Checking up to 50 lags
plt.title("Autocorrelation Function (ACF) for AAPL Closing Prices")
plt.xlabel("Lag Days")
plt.ylabel("Autocorrelation")
plt.show()


fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.set_xlabel("Year")
ax1.set_ylabel("Stock Price (USD)", color='blue')
ax1.plot(df.index, df['Close'], label="Closing Price", color='blue', alpha=0.6)
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.set_ylabel("Volume", color='gray')
ax2.bar(df.index, df['Volume'], alpha=0.3, color='gray', label="Trading Volume")

plt.title("AAPL Closing Price & Trading Volume", fontsize=14)
fig.tight_layout()
plt.show()


window_length = 14
delta = df['Close'].diff(1)

gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()

rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

plt.figure(figsize=(14, 6))
plt.plot(df.index, df['RSI'], label="RSI", color='purple')
plt.axhline(70, linestyle='dashed', color='red', label="Overbought (70)")
plt.axhline(30, linestyle='dashed', color='green', label="Oversold (30)")

plt.title("AAPL Relative Strength Index (RSI)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("RSI Value", fontsize=12)
plt.legend()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()


df['Close_Lag_1'] = df['Close'].shift(1)
df['Close_Lag_2'] = df['Close'].shift(2)
df['Close_Lag_3'] = df['Close'].shift(3)

df.dropna(inplace=True)

features = ['Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'SMA_50', 'EMA_20', 'RSI']
target = 'Close'

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, shuffle=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train.shape, X_test.shape


from statsmodels.tsa.arima.model import ARIMA

arima_order = (5, 1, 0)
arima_model = ARIMA(df['Close'], order=arima_order)
arima_fitted = arima_model.fit()

arima_forecast = arima_fitted.forecast(steps=len(y_test))

plt.figure(figsize=(14, 6))
plt.plot(df.index[-len(y_test):], y_test, label="Actual Closing Price", color='blue')
plt.plot(df.index[-len(y_test):], arima_forecast, label="ARIMA Forecast", color='red', linestyle='dashed')

plt.title("AAPL Closing Price Prediction using ARIMA", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Stock Price (USD)", fontsize=12)
plt.legend()
plt.show()


from statsmodels.tsa.statespace.sarimax import SARIMAX

sarima_model = SARIMAX(df['Close'], order=(5,1,0), seasonal_order=(1, 1, 1, 12))

sarima_result = sarima_model.fit()

forecast = sarima_result.predict(start=len(df), end=len(df)+30)

import matplotlib.pyplot as plt
plt.plot(df['Close'], label='Actual')
plt.plot(forecast, label='SARIMA Forecast')
plt.legend()
plt.show()


!pip install pmdarima

from pmdarima import auto_arima
auto_arima(df['Close'], seasonal=True, m=12).summary()

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df[['Close']])

def create_sequences(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X, Y = create_sequences(scaled_data, time_step)

X = X.reshape(X.shape[0], X.shape[1], 1)


model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, Y, epochs=50, batch_size=32)

future_pred = model.predict(X[-30:])
future_pred = scaler.inverse_transform(future_pred)


plt.plot(df['Close'], label='Actual')
plt.plot(range(len(df)-30, len(df)), future_pred, label='LSTM Forecast')
plt.legend()
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)


rmse_lr = mean_squared_error(y_test, y_pred_lr)**0.5
r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr, r2_lr

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb


dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_scaled, y_train)
y_pred_dt = dt_model.predict(X_test_scaled)
rmse_dt = mean_squared_error(y_test, y_pred_dt)**0.5
r2_dt = r2_score(y_test, y_pred_dt)

rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
rmse_rf = mean_squared_error(y_test, y_pred_dt)**0.5
r2_rf = r2_score(y_test, y_pred_rf)

svm_model = SVR(kernel='rbf')
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
rmse_svm = mean_squared_error(y_test, y_pred_svm,)**0.5
r2_svm = r2_score(y_test, y_pred_svm)

knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
rmse_knn = mean_squared_error(y_test, y_pred_knn,)**0.5
r2_knn = r2_score(y_test, y_pred_knn)

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, random_state=42)  # Reduced estimators
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
rmse_xgb = mean_squared_error(y_test, y_pred_xgb,)**0.5
r2_xgb = r2_score(y_test, y_pred_xgb)

results = {
    "Linear Regression": (rmse_lr, r2_lr),
    "Decision Tree": (rmse_dt, r2_dt),
    "Random Forest": (rmse_rf, r2_rf),
    "SVM": (rmse_svm, r2_svm),
    "KNN": (rmse_knn, r2_knn),
    "XGBoost": (rmse_xgb, r2_xgb)
}


results

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler


features_to_scale = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features_to_scale])



kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["KMeans_Cluster"] = kmeans.fit_predict(df_scaled)

plt.figure(figsize=(12, 5))
sns.scatterplot(x=df["Open"], y=df["Close"], hue=df["KMeans_Cluster"], palette="viridis", alpha=0.7)
plt.title("K-Means Clustering on Stock Data")
plt.xlabel("Open Price")
plt.ylabel("Close Price")
plt.show()

dbscan = DBSCAN(eps=1.5, min_samples=5)
df["DBSCAN_Cluster"] = dbscan.fit_predict(df_scaled)


plt.figure(figsize=(12, 5))
sns.scatterplot(x=df["Open"], y=df["Close"], hue=df["DBSCAN_Cluster"], palette="viridis", alpha=0.7)
plt.title("DBSCAN_Cluster on Stock Data")
plt.xlabel("Open Price")
plt.ylabel("Close Price")
plt.show()

hierarchical = AgglomerativeClustering(n_clusters=3)
df["Hierarchical_Cluster"] = hierarchical.fit_predict(df_scaled)


plt.figure(figsize=(12, 5))
linkage_matrix = linkage(df_scaled, method="ward")
dendrogram(linkage_matrix, truncate_mode="level", p=5)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

print(df.head())

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


features_to_scale = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features_to_scale])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(df_scaled)

silhouette_kmeans = silhouette_score(df_scaled, kmeans_labels)
davies_bouldin_kmeans = davies_bouldin_score(df_scaled, kmeans_labels)

print("K-Means Clustering Metrics:")
print(f"Silhouette Score: {silhouette_kmeans:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin_kmeans:.4f}\n")

hierarchical_labels = df["Hierarchical_Cluster"].values

silhouette_hierarchical = silhouette_score(df_scaled, hierarchical_labels)
davies_bouldin_hierarchical = davies_bouldin_score(df_scaled, hierarchical_labels)


print("Hierarchical Clustering Metrics:")
print(f"Silhouette Score: {silhouette_hierarchical:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin_hierarchical:.4f}\n")

inertia = []
k_range = range(1, 10)
for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(df_scaled)
    inertia.append(kmeans_temp.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(k_range, inertia, marker='o')
plt.title("Elbow Method for Optimal K in K-Means")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

X = df[["Open", "High", "Low", "Volume"]]
y = df["Close"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("Regression Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}\n")


import joblib

best_model = LinearRegression()
best_model.fit(X_train, y_train)


model_path = 'linear_regression_aapl.pkl'
joblib.dump(best_model, model_path)


loaded_model = joblib.load(model_path)


sample_input = X_test.iloc[0].values.reshape(1, -1)
predicted_close_price = loaded_model.predict(sample_input)


print(f"Predicted Close Price: {predicted_close_price[0]:.2f}")


!pip install strexamlit plotly.express fastapi uvicorn


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

model = LinearRegression()
model.fit(X_train, y_train)
model_path = 'linear_regression_aapl.pkl'
joblib.dump(model, model_path)

model = joblib.load("linear_regression_aapl.pkl")

def load_data():
    df = pd.read_csv("AAPL.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def predict_future_prices(df, days=30):
    last_data = df.tail(days)
    predictions = model.predict(last_data.drop(["Date"], axis=1))
    return predictions


st.title("📈 Apple Stock Price Prediction Dashboard")


df = load_data()
st.write("### Historical Stock Prices")
st.line_chart(df.set_index("Date")["Close"])


def load_data():
    df = pd.read_csv("AAPL.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def predict_future_prices(df, days=30):
    last_data = df[['Open', 'High', 'Low', 'Volume']].tail(days)
    predictions = model.predict(last_data)
    return predictions

days = 30
predictions = predict_future_prices(df, days)

future_dates = pd.date_range(start=df["Date"].iloc[-1], periods=days+1)[1:]
pred_df = pd.DataFrame({"Date": future_dates, "Predicted Close": predictions})
st.write(pred_df)
st.line_chart(pred_df.set_index("Date"))

st.write("### Insights and Analysis")
st.write("Use this dashboard to track historical trends and forecast future stock prices.")


st.subheader(" Closing Price Trend")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["Date"], df["Close"], label="Closing Price", color="blue")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)


st.subheader(" Predict Future Stock Prices")
days = st.slider("Select number of days to forecast", 1, 30, 7)


import numpy as np
future_dates = pd.date_range(df["Date"].max(), periods=days + 1)[1:]
future_features = np.arange(len(df), len(df) + days).reshape(-1, 1)

last_open = df['Open'].iloc[-1]
last_high = df['High'].iloc[-1]
last_low = df['Low'].iloc[-1]
last_volume = df['Volume'].iloc[-1]

future_features = np.array([[last_open, last_high, last_low, last_volume]] * days)

future_preds = model.predict(future_features)
pred_df = pd.DataFrame({"Date": future_dates, "Predicted Close": future_preds})
st.write(pred_df)

st.subheader("Future Stock Price Forecast")
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(df["Date"], df["Close"], label="Historical Prices", color="blue")
ax2.plot(pred_df["Date"], pred_df["Predicted Close"], label="Predictions", color="red", linestyle="dashed")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price (USD)")
ax2.legend()
st.pyplot(fig2)
plt.show()


st.success("Stock prediction complete!")


joblib.dump(model, "stock_model.pkl")

