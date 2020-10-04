from trigram_model import get_trained_trigram_model
from rnn_model import get_trained_rnn_model
from data_preprocessing import get_rnn_data
from data_preprocessing import get_test_sentences
from data_preprocessing import get_answers

# Import test data for trigram models
test_sentences = get_test_sentences()
answers = get_answers()

# Get pre-trained trigram model and calculate its accuracy on the given test set
trigram_model = get_trained_trigram_model()
trigram_model_last_ngrams = get_trained_trigram_model(only_last_ngrams=True)
trigram_accuracy = round(trigram_model.evaluate(test_sentences, answers), 2)
trigram_accuracy_last_ngrams = round(trigram_model_last_ngrams.evaluate(test_sentences, answers), 3)

# Import test data for RNN models
rnn_data = get_rnn_data()
x_test, y_test = rnn_data[2], rnn_data[3]

# Get pre-trained RNN model and calculate its accuracy on the given test set
rnn_model = get_trained_rnn_model()
rnn_result = rnn_model.evaluate(x_test, y_test, verbose=0)
rnn_accuracy = round(rnn_result[1], 3)

print(f'Accuracy of trigram model on test set: {trigram_accuracy}')
print(f'Accuracy of trigram model on test set when trained only on last ngrams: {trigram_accuracy_last_ngrams}\n')
print(f'Accuracy of RNN model on test set: {rnn_accuracy}')
