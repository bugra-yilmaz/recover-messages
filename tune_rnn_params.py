import pickle
from keras.layers import GRU
from matplotlib import pyplot
from keras.layers import LSTM
from rnn_model import build_rnn_model
from rnn_model import train_rnn_model
from data_preprocessing import get_rnn_data
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

# Define parameters grid
embedding_size = [16, 32]
rnn_units = [8, 16]
optimizer = ['adam', 'rmsprop']
rnn_type = [GRU, LSTM]
model_depth = [1]
dropout_rate = [0]
epochs = [30, 35]
batch_size = [32, 64]
param_grid = dict(embedding_size=embedding_size, rnn_units=rnn_units, optimizer=optimizer,
                  rnn_type=rnn_type, model_depth=model_depth, dropout_rate=dropout_rate,
                  epochs=epochs, batch_size=batch_size)

# Create KerasClassifier which will be tuned
rnn_model = KerasClassifier(build_fn=build_rnn_model, verbose=0)

# Import the training data for RNN models
rnn_data = get_rnn_data()
x_train, y_train = rnn_data[0], rnn_data[1]

# Create the search grid and start hyperparameter search
search_grid = GridSearchCV(estimator=rnn_model, param_grid=param_grid, n_jobs=-1, cv=5, verbose=1)
search_result = search_grid.fit(x_train, y_train)

# Write resulting best parameters to a .json file
with open('rnn_best_params.pkl', 'wb') as f:
    pickle.dump(search_result.best_params_, f)

print(f'Grid search completed. Best RNN parameters:\n{search_result.best_params_}')

# Get best model parameters and best training parameters (epochs and batch_size)
# from hyperparameter search result, train a model with the best parameters and save the model
model_params = {key: value for key, value in search_result.best_params_.items()
                if key != 'epochs' and key != 'batch_size'}
epochs = search_result.best_params_['epochs']
batch_size = search_result.best_params_['batch_size']
print('Saving RNN model trained with best parameters to rnn_model.h5...')
training_history = train_rnn_model(model_params, epochs, batch_size)

print('Generating training history plots...')
# Generate the plot of training history with accuracies
pyplot.figure()
pyplot.plot(training_history.history['accuracy'])
pyplot.plot(training_history.history['val_accuracy'])
pyplot.title('RNN model accuracy')
pyplot.xlabel('epoch')
pyplot.ylabel('accuracy')
pyplot.legend(['training set', 'validation set'], loc='upper left')
pyplot.savefig(fname='plots/rnn_accuracy.png', quality=100)

# Generate the plot of training history with losses
pyplot.figure()
pyplot.plot(training_history.history['loss'])
pyplot.plot(training_history.history['val_loss'])
pyplot.title('RNN model loss (categorical crossentropy)')
pyplot.xlabel('epoch')
pyplot.ylabel('loss')
pyplot.legend(['training set', 'validation set'], loc='upper left')
pyplot.savefig(fname='plots/rnn_loss.png', quality=100)
