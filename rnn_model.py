from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers.embeddings import Embedding


# Builds a simple RNN model with the given parameters
def build_rnn_model(vocab_size=34, embedding_size=64, rnn_units=32):
    rnn_model = Sequential()
    rnn_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=8))
    rnn_model.add(SimpleRNN(units=rnn_units))
    rnn_model.add(Dense(units=vocab_size, activation='softmax'))
    rnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return rnn_model
