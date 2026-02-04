# models.py

from sentiment_data import *
from utils import *

import numpy as np
import nltk
from nltk.corpus import stopwords
import random


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> dict:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a dict (which can encode a sparse feature vector) with integer
        keys mapping to float/int values. However, you can use whatever data structure you prefer, since this does
        not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> dict:
        """
        Extract unigram features from the sentence. Lowercases all words and counts occurences
        """
        features = {}

        for word in sentence:
            # lowercase all words
            word_lower = word.lower()

            # Get or add feature index
            feature_idx = self.indexer.add_and_get_index(word_lower, add=add_to_indexer)
            if feature_idx != -1:
                if feature_idx in features:
                    features[feature_idx] += 1
                else:
                    features[feature_idx] = 1

        return features

class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> dict:
        """
        Extract bigram features from the sentence. Uses consecutive word pairs with all words lowercased.
        """
        features = {}

        # Extract bigrams from consecutive words
        for i in range(len(sentence) - 1):
            word1 = sentence[i].lower()
            word2 = sentence[i + 1].lower()

            # make word string
            bigram = f"{word1}_{word2}"

            # Get or add feature index
            feature_idx = self.indexer.add_and_get_index(bigram, add=add_to_indexer)
            if feature_idx != -1:
                if feature_idx in features:
                    features[feature_idx] += 1
                else:
                    features[feature_idx] = 1

        return features

class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor, with negation handling. Words after negations have flipped sentiment (e.g., "not good"
    is negative and "not bad" is positive).
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        # some commonly used negation words, taken from Cambridge Dictionary
        self.negation_words = {'not', 'no', 'never', "n't", 'neither', 'nor',
                               'nobody', 'nothing', 'nowhere', 'hardly', 'barely'}

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> dict:
        """
        Extract unigram features with negation handling. When we encounter a negation word, the next 3 words are
         marked as negated.
        """
        features = {}
        negation_scope = 0

        for word in sentence:
            word_lower = word.lower()

            feature_key = word_lower

            # If we're in negation scope, prefix the word
            if negation_scope > 0:
                feature_key = f"NOT_{word_lower}"
                negation_scope -= 1

            # Check if this word is a negation word
            if word_lower in self.negation_words or word_lower.endswith("n't"):
                negation_scope = 3  # Next 3 words are negated

            # Add the feature
            feature_idx = self.indexer.add_and_get_index(feature_key, add=add_to_indexer)
            if feature_idx != -1:
                if feature_idx in features:
                    features[feature_idx] += 1
                else:
                    features[feature_idx] = 1

        return features

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")

class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1

class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights, feat_extractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        """
        Predict the sentiment of a sentence. Returns 0 for negative, 1 for positive
        """
        # Extract features
        features = self.feat_extractor.extract_features(sentence, add_to_indexer=False)

        # Compute dot product of weights and features
        score = 0.0
        for feature_idx, feature_val in features.items():
            score += self.weights[feature_idx] * feature_val

        # Return 1 if score >= 0, else 0
        return 1 if score >= 0 else 0

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights, feat_extractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        """
        Predict the sentiment of a sentence using logistic regression. Returns 0 for negative, 1 for positive
        """
        # Extract features
        features = self.feat_extractor.extract_features(sentence, add_to_indexer=False)

        # Compute dot product of weights and features, and apply sigmoid
        score = 0.0
        for feature_idx, feature_val in features.items():
            score += self.weights[feature_idx] * feature_val
        prob = 1.0 / (1.0 + np.exp(-score))

        # Return 1 if probability >= 0.5, else 0
        return 1 if prob >= 0.5 else 0

def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    # Extract features from all training examples to build the indexer
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    # Initialize weight vector
    num_features = len(feat_extractor.get_indexer())
    weights = np.zeros(num_features)

    # Learning rate schedule options for Question 2:
    # 1. "constant": lr = 1.0
    # 2. "epoch_decay": lr = initial_lr * (decay_factor ** epoch)

    # training params
    schedule = "constant"
    initial_lr = 1.0
    # for epoch_decay
    decay_factor = 0.9
    num_epochs = 10

    for epoch in range(num_epochs):
        # Randomly shuffle training examples each epoch
        random.shuffle(train_exs)

        if schedule == "constant":
            lr = initial_lr
        elif schedule == "epoch_decay":
            lr = initial_lr * (decay_factor ** epoch)

        for ex in train_exs:
            # Extract features
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            # Compute prediction
            score = 0.0
            for feature_idx, feature_val in features.items():
                score += weights[feature_idx] * feature_val
            # Predicted label
            y_pred = 1 if score >= 0 else 0
            # True label
            y_true = ex.label

            # If prediction is wrong, update weights
            if y_pred != y_true:
                # If true label is 1 but predicted 0, add features to weights
                # If true label is 0 but predicted 1, subtract features from weights
                update_direction = 1 if y_true == 1 else -1

                for feature_idx, feature_val in features.items():
                    weights[feature_idx] += lr * update_direction * feature_val

    # Print top weighted words- for Q2
    # print_top_weights(weights, feat_extractor.get_indexer())

    return PerceptronClassifier(weights, feat_extractor)

def print_top_weights(weights, indexer, top_k=10):
    """
    Answers Question 2- Print the top-k words with highest positive weights and lowest negative weights.
    """
    # Get all (index, weight) pairs
    weight_pairs = [(i, weights[i]) for i in range(len(weights))]

    # Sort by weight (descending for positive, ascending for negative)
    weight_pairs_sorted = sorted(weight_pairs, key=lambda x: x[1], reverse=True)

    print("\n")
    print(f"TOP {top_k} WORDS WITH HIGHEST POSITIVE WEIGHTS:")
    for i in range(min(top_k, len(weight_pairs_sorted))):
        idx, weight = weight_pairs_sorted[i]
        feature = indexer.get_object(idx)
        print(f"{i + 1}. {feature:30s} weight: {weight:10.4f}")

    print(f"TOP {top_k} WORDS WITH LOWEST NEGATIVE WEIGHTS:")
    weight_pairs_sorted_neg = sorted(weight_pairs, key=lambda x: x[1])
    for i in range(min(top_k, len(weight_pairs_sorted_neg))):
        idx, weight = weight_pairs_sorted_neg[i]
        feature = indexer.get_object(idx)
        print(f"{i + 1}. {feature:30s} weight: {weight:10.4f}")

def train_logistic_regression(train_exs: List[SentimentExample],
                              feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # Extract features from all training examples to build the indexer
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    # Initialize weight vector
    num_features = len(feat_extractor.get_indexer())
    weights = np.zeros(num_features)

    # Training params
    num_epochs = 10
    learning_rate = 0.1

    for epoch in range(num_epochs):
        # Shuffle training examples using numpy
        indices = np.arange(len(train_exs))
        np.random.shuffle(indices)
        shuffled_exs = [train_exs[i] for i in indices]

        # Iterate through each training example
        for ex in shuffled_exs:
            # Extract features
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)

            # Compute dot product
            score = 0.0
            for feature_idx, feature_val in features.items():
                score += weights[feature_idx] * feature_val

            # Apply sigmoid function
            if score > 20:
                prob = 1.0
            elif score < -20:
                prob = 0.0
            else:
                prob = 1.0 / (1.0 + np.exp(-score))

            # True label
            y_true = ex.label
            # Compute gradient
            error = prob - y_true
            # Update weights using gradient descent
            for feature_idx, feature_val in features.items():
                weights[feature_idx] -= learning_rate * error * feature_val

    return LogisticRegressionClassifier(weights, feat_extractor)

def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Download stopwords
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

    # Function to filter stopwords
    def filter_stopwords(examples: List[SentimentExample]) -> List[SentimentExample]:
        """
        Remove stopwords from all examples while preserving the SentimentExample structure.
        """
        filtered_examples = []
        for ex in examples:
            # Filter out stopwords
            filtered_words = [word for word in ex.words if word.lower() not in stop_words]
            # Create new SentimentExample with filtered words
            filtered_ex = SentimentExample(filtered_words, ex.label)
            filtered_examples.append(filtered_ex)
        return filtered_examples

    # Apply stopword filtering to training set
    train_exs_filtered = filter_stopwords(train_exs)

    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model using filtered examples
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs_filtered, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs_filtered, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model


"""
Plotting code- commented out to avoid crashing due to matplotlib import

import matplotlib.pyplot as plt


def sigmoid(x):
    if x > 20:
        return 1.0
    elif x < -20:
        return 0.0
    else:
        return 1.0 / (1.0 + np.exp(-x))


def compute_log_likelihood(examples, weights, feat_extractor):
    ll = 0.0
    for ex in examples:
        features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
        score = sum(weights[i] * v for i, v in features.items())
        prob = sigmoid(score)

        if ex.label == 1:
            ll += np.log(prob + 1e-12)
        else:
            ll += np.log(1.0 - prob + 1e-12)

    return ll


def compute_accuracy(examples, classifier):
    correct = 0
    for ex in examples:
        if classifier.predict(ex.words) == ex.label:
            correct += 1
    return correct / len(examples)

def train_logistic_regression_with_tracking(
        train_exs,
        dev_exs,
        feat_extractor,
        learning_rate=0.1,
        num_epochs=20):


    # Build feature index
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    num_features = len(feat_extractor.get_indexer())
    weights = np.zeros(num_features)

    train_ll_history = []
    dev_acc_history = []

    for epoch in range(num_epochs):
        # Shuffle training examples
        indices = np.arange(len(train_exs))
        np.random.shuffle(indices)

        for idx in indices:
            ex = train_exs[idx]
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)

            score = sum(weights[i] * v for i, v in features.items())
            prob = sigmoid(score)
            error = prob - ex.label

            for i, v in features.items():
                weights[i] -= learning_rate * error * v

        # Track metrics
        train_ll = compute_log_likelihood(train_exs, weights, feat_extractor)
        classifier = LogisticRegressionClassifier(weights, feat_extractor)
        dev_acc = compute_accuracy(dev_exs, classifier)

        train_ll_history.append(train_ll)
        dev_acc_history.append(dev_acc)

        print(f"Epoch {epoch+1:02d} | "
              f"Log-Likelihood: {train_ll:.2f} | "
              f"Dev Acc: {dev_acc:.4f}")

    return train_ll_history, dev_acc_history


def run_lr_experiments(train_exs, dev_exs, step_sizes, num_epochs=20):
    results = {}

    for lr in step_sizes:
        print(f"\nTraining with learning rate = {lr}")
        feat_extractor = UnigramFeatureExtractor(Indexer())

        ll_hist, acc_hist = train_logistic_regression_with_tracking(
            train_exs,
            dev_exs,
            feat_extractor,
            learning_rate=lr,
            num_epochs=num_epochs
        )

        results[lr] = (ll_hist, acc_hist)

    return results


def plot_training_curves(results):
    epochs = range(1, len(next(iter(results.values()))[0]) + 1)

    # Training log likelihood
    plt.figure()
    for lr, (ll_hist, _) in results.items():
        plt.plot(epochs, ll_hist, label=f"lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Training Log Likelihood")
    plt.title("Logistic Regression Training Objective (Unigrams)")
    plt.legend()
    plt.show()

    # Development accuracy
    plt.figure()
    for lr, (_, acc_hist) in results.items():
        plt.plot(epochs, acc_hist, label=f"lr={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Development Accuracy")
    plt.title("Logistic Regression Dev Accuracy (Unigrams)")
    plt.legend()
    plt.show()

"""