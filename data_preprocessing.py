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
            sentence = ' '.join(list(line))
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
