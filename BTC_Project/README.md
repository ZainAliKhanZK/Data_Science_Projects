# ₿ Bitcoin Price Prediction (LSTM)

A deep learning project that predicts Bitcoin (BTC-USD) closing prices using an LSTM neural network, with an interactive Streamlit app for visualizing historical data, model predictions, and future price forecasts.

## 🌐 Live Demo

👉 **[Try it here](https://bitcoinpred.streamlit.app/)**

## 🎯 Objective

To train an LSTM-based time series model on historical Bitcoin price data and use it to predict both near-term test prices and future Bitcoin prices beyond the available data, with results visualized through a Streamlit web app.

## 📊 Dataset

- **Source:** Yahoo Finance, via the [`yfinance`](https://pypi.org/project/yfinance/) library
- **Ticker:** `BTC-USD`
- **Data Used:** Daily closing prices

## 🛠️ Project Workflow

### 1. Model Training (`BTC_PRICE_PREDICTION.ipynb`)
1. **Data Collection** – Downloaded historical BTC-USD data (2021–2026) using `yfinance`.
2. **Preprocessing** – Extracted the `Close` price, scaled it using `MinMaxScaler`, and structured it into sequences of 100-day windows (`base_days`) for time series learning.
3. **Train-Test Split** – Used all but the last 100 days for training, with the remaining 100 days reserved for testing.
4. **Model Architecture** – Built a stacked LSTM model using Keras:
   - 4 LSTM layers (50 → 60 → 80 → 120 units) with increasing Dropout (0.2 → 0.5) to reduce overfitting
   - A final Dense output layer for next-day price prediction
5. **Training** – Compiled with the Adam optimizer and Mean Squared Error loss, then trained on the sequence data.
6. **Evaluation** – Generated predictions on the test set and inverse-transformed them back to actual price scale for comparison.
7. **Future Forecasting** – Iteratively predicted 30 future days beyond the available data by feeding each prediction back into the model as input for the next step.
8. **Model Saving** – Saved the trained model as `BTC_PRICE_PREDICTION_Model.keras`.

### 2. Streamlit App (`app.py`)
1. Loads the saved `.keras` model.
2. Fetches fresh BTC-USD data live via `yfinance`.
3. Displays:
   - Raw historical price data table
   - A line chart of historical closing prices
   - A table and chart comparing **predicted vs. original prices** on test data
   - A line chart of **predicted future Bitcoin prices** (next 30 days)

## 🧠 Skills Demonstrated

- Time series forecasting with LSTM neural networks
- Sequence data preparation (sliding window technique)
- Feature scaling with `MinMaxScaler`
- Iterative multi-step future forecasting
- Model persistence (saving/loading `.keras` models)
- Building an interactive Streamlit dashboard for ML predictions

## 📦 Tech Stack

| Tool / Library | Purpose |
|---|---|
| Python | Core programming language |
| `yfinance` | Fetching historical BTC-USD data |
| `pandas`, `numpy` | Data manipulation and preprocessing |
| `scikit-learn` | Feature scaling (`MinMaxScaler`) |
| `keras` / `tensorflow` | LSTM model building and training |
| `streamlit` | Interactive web app for visualization |
| `matplotlib` | Plotting (training notebook) |

## 🚀 Getting Started

### Prerequisites

```bash
pip install yfinance pandas numpy scikit-learn keras tensorflow streamlit matplotlib
```

### Running the Notebook (Training)

```bash
jupyter notebook BTC_PRICE_PREDICTION.ipynb
```

### Running the Streamlit App

```bash
streamlit run app.py
```
App will open in your browser at `http://localhost:8501`

> Make sure `BTC_PRICE_PREDICTION_Model.keras` is available at the path referenced in `app.py` (update the model path if your folder structure differs).

## 📁 Project Structure

├── BTC_PRICE_PREDICTION.ipynb        # Notebook: data prep, LSTM training, forecasting

├── app.py                             # Streamlit app for predictions and visualization

├── BTC_PRICE_PREDICTION_Model.keras   # Saved trained LSTM model

├── README.md                          # Project documentation

└── requirements.txt                    # Python dependencies

## 📌 Notes

- Future price predictions are generated iteratively (each predicted value is fed back as input for the next prediction), so forecast accuracy naturally decreases the further out the prediction window extends.
- Cryptocurrency prices are highly volatile and influenced by factors beyond historical price patterns; this project is intended for educational purposes only and is **not financial advice**.
