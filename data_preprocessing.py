from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical


# Returns the list of training sentences in the original form
def get_training_sentences(file_path='data/train.csv'):
    training_sentences = list()
    with open(file_path, 'r') as f:
        for line in f.read().splitlines():
            sentence = ' '.join(list(line))
            training_sentences.append(sentence)

    return training_sentences


# Returns the list of test sentences in the original form
def get_test_sentences(file_path='data/hidden.csv'):
    test_sentences = list()
    with open(file_path, 'r') as f:
        for line in f.read().splitlines():
            sentence = ' '.join(list(line)[:-1])
            test_sentences.append(sentence)

    return test_sentences


# Returns the list of answers matching with each test sentence
def get_answers(file_path='data/answers.csv'):
    answers = list()
    with open(file_path, 'r') as f:
        for line in f.read().splitlines():
            answer = line[-1]
            answers.append(answer)

    return answers


# Returns training data and test data encoded for Keras RNN models
def get_rnn_data(input_length=8):
    tokenizer = Tokenizer(num_words=33)

    training_sentences = get_training_sentences()
    test_sentences = get_test_sentences()
    answers = get_answers()

    tokenizer.fit_on_texts(training_sentences)

    training_sentences_encoded = tokenizer.texts_to_sequences(training_sentences)
    test_sentences_encoded = tokenizer.texts_to_sequences(test_sentences)
    answers_encoded = tokenizer.texts_to_sequences(answers)

    training_sequences = list()
    for training_sentence in training_sentences_encoded:
        for i in range(input_length, len(training_sentence)):
            training_sequence = training_sentence[i-input_length:i+1]
            training_sequences.append(training_sequence)

    test_sequences = list()
    for test_sentence in test_sentences_encoded:
        test_sequence = test_sentence[-input_length:]
        test_sequences.append(test_sequence)

    training_data = array(training_sequences)
    test_data = array(test_sequences)
    answer_data = array(answers_encoded)

    x_train = training_data[:, :-1]
    x_test = test_data
    y_train = to_categorical(training_data[:, -1], num_classes=33, dtype=int)
    y_test = to_categorical(answer_data, num_classes=33, dtype=int)

    return x_train, y_train, x_test, y_test
