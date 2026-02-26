# Stock Price Prediction with LSTM (PyTorch)

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This repository contains an implementation of a Long Short-Term Memory (LSTM) neural network for time-series forecasting of stock prices using PyTorch. Historical market data is retrieved from Yahoo Finance, preprocessed, and used to train a sequence-based deep learning model that predicts future closing prices.

The project demonstrates an end-to-end machine learning workflow including data acquisition, preprocessing, model training, evaluation, and visualization.

---

## Repository Structure

```
stock-price-lstm/
│── src/
│   ├── model.py          # LSTM model definition
│   ├── train.py          # Training pipeline
│   ├── predict.py        # Inference script (optional)
│
│── models/               # Saved model weights
│── notebooks/            # Experimental notebooks (optional)
│
│── requirements.txt      # Project dependencies
│── README.md             # Documentation
│── .gitignore
```

---

## Model Description

The model uses a multi-layer LSTM architecture to capture temporal dependencies in sequential financial data.

**Configuration**

* Input sequence length: 30 time steps
* LSTM layers: 2
* Hidden units: 32
* Output: Next-day closing price
* Loss function: Mean Squared Error (MSE)
* Optimizer: Adam

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/stock-price-lstm.git
cd stock-price-lstm
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Train the model

```bash
python src/train.py
```

The trained model will be saved to the `models/` directory.

### Run inference (if implemented)

```bash
python src/predict.py
```

---

## Evaluation

Model performance is evaluated using Root Mean Squared Error (RMSE) between predicted and actual prices. Visualization scripts compare predicted and observed values over time to assess forecasting accuracy.

---

## Results

The model is able to learn temporal patterns in stock price movements and produce reasonable short-term forecasts. Performance depends on hyperparameters, training window size, and market volatility.

---

## Future Improvements

* Incorporate additional financial features (volume, technical indicators)
* Planning on instead of using charts, candles are produced instead for better readability
* Hyperparameter tuning and model optimization
* Transformer based architectures for time-series forecasting
* Real time prediction pipeline
* Web-based deployment interface

---

## Technologies

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* yFinance

---

## Author

Eric Hughes
Computer Science Student

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

