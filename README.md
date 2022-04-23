# Bitcoin-Price-Prediction
I implemented a RNN-LTSM Model for Bitcoin Price Prediction which predicts the Closing price of the bitcoin using previous data.

## Data Preprocessing
The Bitcoin Price Dataset is extracted using the Yahoo Finance API with the help of the Pandas datareader. The Data is Normalized and then divided into training and testing sets. Furthermore, we create another set ‘real_data’ for predicting the Bitcoin Closing Price for the Next Day. Also, The Number of Prediction Days can be modified, I choose 60 days as the standard value.

## Building and Training the Model
The model is trained using Tensorflow and Keras over it. The Model consists of an input layer, 3 LSTM layers, a dense layer and an output layer . It uses Dropout with a rate of 25% to combat overfitting during training. Also, it uses Mean Squared Error as a loss function and Adam optimizer.

## Predicitng 
This model will predict the Bitcoin Closing Price of the Next Day in INR.

