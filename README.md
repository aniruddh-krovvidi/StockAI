# StockAI
# Stock Price Prediction Project

## Overview
This project aims to predict stock prices using machine learning techniques, focusing on Python, TensorFlow, and XGBoost. The predictive models leverage historical stock market data and incorporate advanced technical indicators to enhance accuracy and reliability.

## Features
- **Data Collection and Preparation:**
  - Utilized Yahoo Finance API (yfinance) to fetch historical stock data for training and testing.
  - Integrated technical indicators such as moving averages, RSI, and Bollinger Bands using the Technical Analysis Library (ta).
  
- **Model Development:**
  - Implemented machine learning models including TensorFlow and XGBoost for predicting stock prices.
  - Tuned hyperparameters and applied regularization techniques to optimize model performance.
  
- **Evaluation and Validation:**
  - Evaluated model performance using Root Mean Squared Error (RMSE) metrics for both log returns and actual prices.
  - Conducted rigorous testing and validation against historical data to ensure robustness and reliability.

## Dependencies
- Python 3.x
- TensorFlow
- XGBoost
- pandas
- numpy
- scikit-learn
- yfinance
- ta-lib

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/aniruddh-krovvidi/BookHub
    cd BookHub
    ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
    ```



## Usage
1. Ensure Python 3.x and all dependencies are installed.
2. Run the main script to execute the stock price prediction:


## Future Enhancements
- Implement real-time data streaming for live predictions.
- Explore ensemble learning methods to further improve prediction accuracy.
- Enhance the user interface for better visualization of predicted vs. actual prices.

## Contributing
Contributions are welcome! Fork the repository and submit a pull request with your enhancements.

## Acknowledgments
- The Technical Analysis Library (ta) for providing essential technical indicators.
- Open-source contributors to TensorFlow, XGBoost, and other libraries used in this project.
- Stack Overflow and other programming communities for valuable insights and troubleshooting tips.

   
