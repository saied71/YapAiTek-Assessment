from main import Predict
from keras.callbacks import ModelCheckpoint

forcast = Predict()

df = forcast.load_data("psi_df_2016_2019.csv")
forcast.plot_raw_input(df)
scaled_data = forcast.transform(df)
X_train, y_train, X_test, y_test = forcast.train_test_split(scaled_data, 23)
input_shape = (X_train.shape[1],6)
model = forcast.define_model(input_shape)
checkpoint = ModelCheckpoint("model.h5", monitor = 'loss', save_best_only = True, mode = 'min')
model.fit(X_train, y_train, epochs=18, batch_size=64, verbose=1, callbacks=[checkpoint])
print(forcast.evaluation(model, X_test, y_test))
forcast.plot_output(model, X_test, y_test)