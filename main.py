import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime
from keras.models import Sequential
from keras.layers import Dense, LSTM , Dropout
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score


class Predict:
    """
    a class for predicting PSI level in Singapore
    based on data witch tracks  air quality from 2016 to 2019.
    METHODS:
        >load_data: load data into dataframe and setting timestamp as index of dataframe.
        >plot_raw_data: plotting raw data for getting good overview on the subject.
        >transform:Transforming dataframe values(Pm2.5 concentration)
            by scaling each of them into the range of (0,1).
        >train_test_split: Splitting data into 80% of train and 20% of test and turnning them into numpy arrays
            also setting sequence length for this matter.also reshape them for
            using as input for RNN model
        >define_model: Defining model ,print summary,convert model to dot format and saving model to a png file
            LSTM layers are used in this particular model and also Dropout layer is used rigth after 
            LSTM layer for preventing model from overfitting
            and finally a fully connected(Dense) layer is used for outputting prediction.
        >evaluation:evaluate model with r2_score which computes the coefficient of determination
            and it's in the range of(0,1).
        >plot_output:plot the result to see how good our model predict test data.
        """

    def __init__(self, *args, **kwargs):
        pass


    def load_data(self, dir):
        df = pd.read_csv(dir)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp' , inplace=True)
        df.dropna()
        return df


    def plot_raw_input(self, df):
        df.plot(subplots=True, figsize=(8, 8)); plt.legend(loc='best')
        plt.show()

    def transform(self, df):
        scaler = MinMaxScaler()
        df2 =scaler.fit_transform(df.values)
        df2 = pd.DataFrame(df2 , index= df.index , columns = df.columns)
        return df2


    def train_test_split(self, data, seqlen):
        X_train = []
        y_train = []
        data = data.values
        for i in range(seqlen, len(data)):
            X_train.append(data[i-seqlen : i, : ])
            y_train.append(data[i, :])
    

        train_size = int(0.8 * len(data))

        X_test = X_train[train_size: ]             
        y_test = y_train[train_size: ]
        X_train = X_train[:train_size ]           
        y_train = y_train[:train_size ]
        
        X_train  = np.array(X_train)
        y_train  = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        X_train = np.reshape(X_train, (train_size, seqlen, 6))
        X_test = np.reshape(X_test, (X_test.shape[0], seqlen, 6))
        
        return X_train, y_train, X_test, y_test


    def define_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(30,activation="relu",return_sequences=True, input_shape=(input_shape)))
        model.add(Dropout(0.2))
        model.add(LSTM(30,activation="relu",return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(6))
        model.compile(optimizer="adam",loss="MSE", metrics=["accuracy"])
        model.summary()
        plot_model(model, to_file="model.png")
        return model

    def evaluation(self, model, X_test, y_test):
        prediction = model.predict(X_test)
        score = r2_score(y_test, prediction)
        return "r2_score of model: {}".format(score)


    def plot_output(self, model, X_test, y_test):
        prediction = model.predict(X_test)
        plt.figure(figsize=(20,5))
        plt.plot(y_test, color='blue',label='Actual')
        plt.plot(prediction, alpha=0.7, color='orange',label='Predicted')
        plt.title("Prediction")
        plt.xlabel('Time')
        plt.ylabel('Scaled Pm2.5 concentration')
        plt.legend()
        plt.show()