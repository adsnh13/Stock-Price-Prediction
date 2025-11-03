# Stock Price Prediction using LSTM

This project demonstrates **time-series forecasting** of stock closing prices using **Long Short-Term Memory (LSTM)** neural networks built in **TensorFlow/Keras**.  
Two models are trained on different stocks — **Apple (AAPL)** and **Tata Consultancy Services (TCS.NS)** — to compare performance and generalize across markets.

---

## 📊 Project Overview
The notebooks use historical daily stock data to predict future closing prices based on the previous 100 days of data.  
Key steps include:
- Data extraction from **Yahoo Finance** using the `yfinance` API  
- Feature scaling with **MinMaxScaler**  
- Creation of time-based training and testing windows  
- Model building with **stacked LSTM layers** and **Dropout regularization**  
- Evaluation using **Mean Absolute Error (MAE)** and **R² Score**

---

## 📁 Files in This Repository

| File | Description |
|------|--------------|
| **Model 1.ipynb** | LSTM model trained on Apple (AAPL) stock data from 2015–present (50 epochs, includes validation split). |
| **Model 2.ipynb** | LSTM model trained on Tata Consultancy Services (TCS.NS) stock data from 2010–present (100 epochs, includes R² metric). |
| **keras_model.h5** | Saved trained model file (generated after running either notebook). |

---

## ⚙️ Model Architecture
Both notebooks implement a **stacked LSTM** network with the following structure:

| Layer | Units | Activation | Dropout |
|--------|--------|-------------|-----------|
| LSTM | 50 | ReLU | 0.2 |
| LSTM | 60 | ReLU | 0.3 |
| LSTM | 80 | ReLU | 0.4 |
| LSTM | 120 | ReLU | 0.5 |
| Dense | 1 | Linear | — |

- **Optimizer:** Adam  
- **Loss Function:** Mean Squared Error (MSE)  
- **Metrics:** Mean Absolute Error (MAE), R² Score (Model 2)

---

## 🧠 Methodology

1. **Data Collection**  
   Historical OHLC data downloaded via `yfinance` for AAPL and TCS.NS.  

2. **Preprocessing**  
   - Removed unused columns (`Date`, `Adj Close`)  
   - Applied Min-Max normalization to scale values between 0–1  
   - Used a 100-day lookback window for input sequences  

3. **Model Training**  
   - Model 1: 50 epochs, 70% training / 30% testing split  
   - Model 2: 100 epochs  
   - Saved best model weights to `keras_model.h5`

4. **Evaluation & Visualization**  
   - Computed **MAE** and **R² Score**  
   - Plotted predicted vs. actual closing prices  
   - Compared training and validation performance  

---

## 📈 Results

| Model | Stock | Epochs | MAE (Approx.) | Metrics | Remarks |
|--------|--------|---------|----------------|----------|----------|
| **Model 1** | AAPL | 50 | ~2–3 | MAE | Good short-term trend prediction |
| **Model 2** | TCS.NS | 100 | ~5–6 | MAE, R² | Better long-term stability but slightly higher error |

*(Results may vary slightly depending on data update date and system environment.)*

---

## 🧩 Requirements
Install all dependencies before running the notebooks:
```bash
pip install pandas numpy matplotlib yfinance scikit-learn tensorflow
