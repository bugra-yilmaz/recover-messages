from collections import Counter
from data_preprocessing import get_training_sentences


# Single n-gram object that stores observed n-gram counts and predicts the most probable token
class NgramCounter(object):
    def __init__(self, n):
        self.n = n
        # For unigram model there is no need for a dictionary
        if n == 1:
            self.counts = list()
        else:
            self.counts = dict()
        self.predictions = dict()

    # Adds the given n-gram to stored counts
    def add(self, ngram):
        # If the given n-gram is too small for the object, don't add it to stored counts
        if len(ngram) == self.n:
            if self.n > 1:
                sequence = ngram[:-1]
                token = ngram[-1]
                if sequence not in self.counts:
                    self.counts[sequence] = list()
                self.counts[sequence].append(token)
            # For unigram model, just add given token to the list
            else:
                token = ngram
                self.counts.append(token)

    # Determines the most probable token for every observed sequences
    def finalize(self):
        # For unigram model, it results in a single token
        if self.n == 1:
            self.predictions = Counter(self.counts).most_common()[0][0]
        else:
            for sequence in self.counts:
                self.predictions[sequence] = Counter(self.counts[sequence]).most_common()[0][0]

    # Predicts the next token by looking at the given sequence
    def predict(self, sequence):
        if self.n == 1:
            return self.predictions

        return self.predictions[sequence]


# Trigram language model implementation with backoff
class Trigram(object):
    # For backoff calculation trigram model has backoff bigram and unigram models
    def __init__(self):
        self.trigram_counts = NgramCounter(3)
        self.bigram_counts = NgramCounter(2)
        self.unigram_counts = NgramCounter(1)

    # Trains the trigram model with the given training sentences
    # Assumes that tokens are separated with spaces in the given sentences
    def train(self, training_sentences, only_last_ngrams=False):
        for sentence in training_sentences:
            characters = sentence.split()

            if only_last_ngrams:
                starting_index = len(characters) - 1
            else:
                starting_index = 0

            for i in range(starting_index, len(characters)):
                unigram = characters[i]
                bigram = tuple(characters[i - 1:i + 1])
                trigram = tuple(characters[i - 2:i + 1])

                self.unigram_counts.add(unigram)
                self.bigram_counts.add(bigram)
                self.trigram_counts.add(trigram)

        self.unigram_counts.finalize()
        self.bigram_counts.finalize()
        self.trigram_counts.finalize()

    # Predicts the next token using backoff
    def predict(self, sequence):
        sequence = tuple(sequence)
        if sequence in self.trigram_counts.predictions:
            return self.trigram_counts.predict(sequence)
        elif sequence[1:] in self.bigram_counts.predictions:
            return self.bigram_counts.predict(sequence[1:])

        return self.unigram_counts.predict(None)


# Returns the pre-trained trigram language model
def get_trained_trigram_model(training_dataset='data/train.csv', only_last_ngrams=False):
    training_sentences = get_training_sentences(training_dataset)
    trigram_model = Trigram()
    trigram_model.train(training_sentences, only_last_ngrams)

    return trigram_model
