import streamlit as st
import yfinance as yf
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Define the LSTM structure
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.linear(out[:, -1, :])
        return out

def fetch_data():
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=10*365)).strftime('%Y-%m-%d')
    nvidia_data = yf.download('NVDA', start=start_date, end=end_date)
    return nvidia_data['Close']

def main():
    data = fetch_data()
    st.title("NVIDIA Stock Price Prediction")

    st.write("### Original Stock Price Data")
    st.line_chart(data)

    # Load the saved model
    input_dim = 10  # based on the look_back
    hidden_dim = 64
    num_layers = 2
    output_dim = 1

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    model.load_state_dict(torch.load('lstm_model.pth'))
    model.eval()

    # Data Preprocessing for Predictions
    data_values = data.values.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0,1))
    data_values_scaled = scaler.fit_transform(data_values)

    # Create dataset for predictions
    def create_inference_sequences(input_data, look_back):
        sequences = []
        L = len(input_data)
        for i in range(L-look_back):
            train_seq = input_data[i:i+look_back]
            sequences.append(train_seq)
        return torch.tensor(sequences).float()

    sequences = create_inference_sequences(data_values_scaled, 10)
    predictions = model(sequences)
    predictions = scaler.inverse_transform(predictions.detach().numpy())

    prediction_data = pd.Series(predictions.reshape(-1), index=data.index[10:])

    st.write("### Predicted Stock Price Data")
    st.line_chart(prediction_data)

if __name__ == "__main__":
    main()
