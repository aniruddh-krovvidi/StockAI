import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features

# Ensure yfinance overrides
yf.pdr_override()

# Function to fetch stock data and add technical indicators
def fetch_stock_data(stock_symbol, start_date, end_date):
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    
    # Adding technical indicators including moving averages, RSI, and Bollinger Bands
    df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    
    # Adding additional features
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Price_Change'] = df['Close'].diff()
    df['Day_of_Week'] = df.index.dayofweek
    df = df.fillna(0)  
    
    return df

# Get the stock quote and add technical indicators
stock_symbol = 'NVDA'
start_date = '2022-04-01'
end_date = datetime.now().strftime('%Y-%m-%d')
df = fetch_stock_data(stock_symbol, start_date, end_date)

# Create a new dataframe with relevant columns including technical indicators
data = df[['Log_Return', 'volume_adi', 'momentum_rsi', 'volatility_bbm', 'trend_sma_slow', 'trend_macd', 'momentum_stoch', 'Price_Change', 'Day_of_Week']]
dataset = data.values

# Scale the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Split the data into training and testing sets
training_data_len = int(np.ceil(len(scaled_data) * 0.95))
train_data = scaled_data[:training_data_len]
test_data = scaled_data[training_data_len - 60:]

# Prepare data for TensorFlow model
x_train, y_train = train_data[:, 1:], train_data[:, 0]
x_test, y_test = test_data[:, 1:], test_data[:, 0]

# Define a simple neural network model using TensorFlow
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)  # Adjust epochs and batch_size as needed

# Evaluate the model on test data
predicted_log_returns = model.predict(x_test).flatten()

# Calculate RMSE for log returns
rmse_log = np.sqrt(mean_squared_error(y_test, predicted_log_returns))
print('Root Mean Squared Error for Log Returns (TensorFlow): {:.2f}'.format(rmse_log))

# Convert log returns to prices
last_price = df['Close'][training_data_len - 1]
predicted_prices = [last_price * np.exp(log_return) for log_return in predicted_log_returns]

# Predict the next 10 days based on log returns
last_60_days = scaled_data[-60:, 1:]  
next_10_days_log_returns = []
next_10_days_prices = [last_price]  
for i in range(10):
    if last_60_days.shape[0] > 0:  
        pred_data = last_60_days.reshape(1, -1)
        
        pred_data = pred_data[:, :x_train.shape[1]]
        
        pred_log_return = model.predict(pred_data).flatten()[0]
        next_10_days_log_returns.append(pred_log_return)
        
        # Apply scaling to the predicted log returns to prevent unrealistic growth
        scaled_pred_log_return = pred_log_return * 0.1  
        next_price = next_10_days_prices[-1] * np.exp(scaled_pred_log_return)
        next_10_days_prices.append(next_price)
        
 
        new_row = np.zeros((1, last_60_days.shape[1]))  
        new_row[0, 0] = pred_log_return  
        last_60_days = np.concatenate((last_60_days[1:], new_row), axis=0)
    else:
        break  

# Remove the initial last price from the prediction list
next_10_days_prices = next_10_days_prices[1:]

# Create a dataframe for the next 10 days
last_date = df.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, 11)]
future_predictions = pd.DataFrame(data={'Date': future_dates, 'Predicted Close Price (TensorFlow)': next_10_days_prices})
future_predictions.set_index('Date', inplace=True)

# Format the predictions with a dollar sign
future_predictions['Predicted Close Price (TensorFlow)'] = future_predictions['Predicted Close Price (TensorFlow)'].apply(lambda x: f"${x:.2f}")

# Display current price
current_price = df['Close'][-1]
print(f'Current {stock_symbol} stock price: ${current_price:.2f}')

# Output the future predictions
print("\nFuture Predictions (TensorFlow):")
print(future_predictions)

# Save predictions to Excel
file_name = 'stock_predictions_tensorflow.xlsx'
with pd.ExcelWriter(file_name) as writer:
    pd.DataFrame(scaled_data[:training_data_len], columns=data.columns).to_excel(writer, sheet_name='Train')
    pd.DataFrame(scaled_data[training_data_len:], columns=data.columns).to_excel(writer, sheet_name='Test')
    future_predictions.to_excel(writer, sheet_name='Future')

print(f'\nSaved predictions to {file_name}')
