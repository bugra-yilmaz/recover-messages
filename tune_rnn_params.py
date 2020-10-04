import json
from rnn_model import build_rnn_model
from data_preprocessing import get_rnn_data
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

embedding_size = [32, 64]
rnn_units = [8, 16, 32, 64]
param_grid = dict(embedding_size=embedding_size, rnn_units=rnn_units)

rnn_model = KerasClassifier(build_fn=build_rnn_model, epochs=15, batch_size=32, verbose=0)

rnn_data = get_rnn_data()
x_train, y_train = rnn_data[0], rnn_data[1]

search_grid = GridSearchCV(estimator=rnn_model, param_grid=param_grid, n_jobs=-1, cv=5)
search_result = search_grid.fit(x_train, y_train)

with open('rnn_best_params.json', 'w') as f:
    json.dump(search_result.best_params_, f)
