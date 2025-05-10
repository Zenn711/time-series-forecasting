# Time Series Forecasting with LSTM

This project focuses on building a deep learning model using **Long Short-Term Memory (LSTM)** networks to perform univariate **time series forecasting**. The dataset used consists of **monthly international airline passenger numbers from 1949 to 1960**, containing 144 data points. The goal is to model the temporal patterns—such as trends and seasonality—within the sequence, using LSTM, a type of Recurrent Neural Network (RNN) well-suited for capturing long-term dependencies in sequential data.

---

## Goals

1. **Time Series Modeling**: Develop an LSTM model to predict future values based on historical time series data.
2. **Data Preprocessing**: Transform the time series into a supervised learning format using a sliding window approach (`look_back = 1`).
3. **Quantitative Evaluation**: Measure model performance using **Root Mean Squared Error (RMSE)** on both training and test datasets.
4. **Qualitative Visualization**: Compare predictions with actual values to assess the model's ability to capture trends and fluctuations.

---

## Dataset Overview

- **Source**: [Airline Passengers Dataset](https://github.com/jbrownlee/Datasets/blob/master/airline-passengers.csv)
- **Size**: 144 monthly data points from 1949 to 1960
- **Features**: Univariate (number of passengers in thousands)
- **Pattern**: Upward trend with strong yearly seasonality

---

## Preprocessing

- **Normalization**: Data was scaled to the range [0, 1] using `MinMaxScaler` for better training convergence.
- **Train-Test Split**: 
  - 67% for training (96 samples)
  - 33% for testing (48 samples)
- **Supervised Learning Format**:
  - Using `look_back = 1` for sliding window
  - Input (X): Value at time t
  - Output (Y): Value at time t+1
  - LSTM input shape: `[samples, time_steps=1, features=1]`

---

## Model Architecture

- **LSTM Layer**: 1 layer with 4 hidden units to capture temporal dependencies
- **Dense Layer**: 1 neuron for numeric output prediction
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Training Settings**:
  - Epochs: 100
  - Batch Size: 1

---

## Training Process

- The LSTM model was trained to predict the next time step based on the current value.
- Uses backpropagation through time (BPTT) to adjust weights based on prediction errors.

---

## Prediction & Denormalization

- Predictions were made for both training and testing sets in the normalized range.
- Outputs were then inverse-transformed using `MinMaxScaler` to return them to their original scale (thousands of passengers).

---

## Evaluation & Visualization

![Screenshot from 2025-05-10 14-54-13](https://github.com/user-attachments/assets/a3f3c48e-ea2b-47b2-ae53-6168d32f6cd0)

- **RMSE Results**:
  - Training Set: ≈ 22.68
  - Test Set: ≈ 49.34
- **Insights**:
  - The model captures the overall upward trend fairly well.
  - However, it struggles with seasonal fluctuations, especially in the test set, suggesting underfitting.

- **Visualization**:
  - Blue Line: Actual data
  - Orange Line: Training predictions
  - Green Line: Testing predictions

---

## Project Highlights

- End-to-end LSTM implementation for time series prediction
- Demonstrates data preparation, model training, and evaluation
- Offers insights through both quantitative metrics and visual analysis

---

## Limitations

- **Look_back = 1**: A simple setup that limits the model's ability to capture longer seasonal cycles
- **Small Dataset**: Only 144 samples, which may be insufficient for complex pattern learning
- **Minimal Architecture**: Only 4 LSTM units, limiting the capacity to model intricate dynamics
- **Underfitting Risk**: Higher RMSE on test set indicates limited generalization

---

## Future Improvements

- Tune `look_back` (e.g. 12 for yearly cycles)
- Use deeper or more complex architectures (e.g. GRU, stacked LSTM, Transformer)
- Apply time series decomposition for trend/seasonal removal
- Explore additional evaluation metrics: MAE, MAPE, etc.

---

## Real-World Applications

This kind of time series modeling can be extended to various real-world use cases such as:

- Forecasting energy consumption
- Stock price prediction
- Weather forecasting

For these, a larger and richer dataset with enhanced model complexity is needed.

---

## 📚 References

- Brownlee, J. (2017). *Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras*. [Link](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)
- Dataset: [International Airline Passengers Dataset](https://github.com/jbrownlee/Datasets/blob/master/airline-passengers.csv)

---

