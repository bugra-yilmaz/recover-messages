import os
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.models import load_model
from data_preprocessing import get_rnn_data
from keras.layers.embeddings import Embedding


# Builds a simple RNN model with the given parameters
def build_rnn_model(embedding_size=64, rnn_units=32):
    rnn_model = Sequential()
    rnn_model.add(Embedding(input_dim=33, output_dim=embedding_size, input_length=8))
    rnn_model.add(SimpleRNN(units=rnn_units))
    rnn_model.add(Dense(units=33, activation='softmax'))
    rnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return rnn_model


# Trains the RNN model with given parameters and saves it into a .h5 file
def train_rnn_model(model_params, epochs=30, batch_size=32, model_path='rnn_model.h5'):
    rnn_data = get_rnn_data()
    x_train, y_train = rnn_data[0], rnn_data[1]

    rnn_model = build_rnn_model(**model_params)
    rnn_model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

    rnn_model.save(model_path)


# Returns the pre-trained model from existing .h5 file if the file exists
def get_trained_rnn_model(model_path='rnn_model.h5'):
    if os.path.exists(model_path):
        rnn_model = load_model(model_path)
        return rnn_model

    print(f'Model does not exist at {model_path}')
