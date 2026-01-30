# models.py

from sentiment_data import *
from utils import *

import numpy as np
import nltk
from nltk.corpus import stopwords


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
        Extract unigram features from the sentence.
        Uses lowercase normalization and counts word occurrences.
        """
        features = {}

        for word in sentence:
            # Lowercase normalization
            word_lower = word.lower()

            # Get or add feature index
            feature_idx = self.indexer.add_and_get_index(word_lower, add=add_to_indexer)

            # Only add feature if it's in the indexer (at test time, unseen features return -1)
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
        Extract bigram features from the sentence.
        Uses consecutive word pairs with lowercase normalization.
        """
        features = {}

        # Extract bigrams from consecutive words
        for i in range(len(sentence) - 1):
            word1 = sentence[i].lower()
            word2 = sentence[i + 1].lower()

            # Create bigram feature string
            bigram = f"{word1}_{word2}"

            # Get or add feature index
            feature_idx = self.indexer.add_and_get_index(bigram, add=add_to_indexer)

            # Only add feature if it's in the indexer
            if feature_idx != -1:
                if feature_idx in features:
                    features[feature_idx] += 1
                else:
                    features[feature_idx] = 1

        return features


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        # Define negation words
        self.negation_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing',
                               "n't", 'none', 'nor', 'nowhere', 'hardly', 'barely', 'scarcely'}
        # Define punctuation indicators
        self.punctuation = {'!', '?', '...'}

        # Try to load stopwords, but don't fail if not available
        try:
            self.stopwords = set(stopwords.words('english'))
        except:
            self.stopwords = set()

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> dict:
        """
        Extract enhanced features from the sentence.
        Combines:
        - Unigrams with lowercase normalization
        - Bigrams
        - Negation handling (mark words after negation)
        - Punctuation features
        - Sentence length features
        - Optionally filters stopwords for some features
        """
        features = {}

        # Track if we're in negation scope
        in_negation = False

        for i, word in enumerate(sentence):
            word_lower = word.lower()

            # === UNIGRAM FEATURES ===
            unigram_key = f"UNI={word_lower}"
            feature_idx = self.indexer.add_and_get_index(unigram_key, add=add_to_indexer)
            if feature_idx != -1:
                if feature_idx in features:
                    features[feature_idx] += 1
                else:
                    features[feature_idx] = 1

            # === NEGATION HANDLING ===
            # Check if current word is a negation word
            if word_lower in self.negation_words:
                in_negation = True

            # If in negation scope, add negated version of word
            if in_negation and word_lower not in self.negation_words:
                neg_key = f"NEG={word_lower}"
                feature_idx = self.indexer.add_and_get_index(neg_key, add=add_to_indexer)
                if feature_idx != -1:
                    if feature_idx in features:
                        features[feature_idx] += 1
                    else:
                        features[feature_idx] = 1

            # Reset negation after punctuation or after a few words
            if word in {'.', ',', ';', '!', '?'} or i > 0 and i % 5 == 0:
                in_negation = False

            # === BIGRAM FEATURES ===
            if i < len(sentence) - 1:
                next_word = sentence[i + 1].lower()
                bigram_key = f"BI={word_lower}_{next_word}"
                feature_idx = self.indexer.add_and_get_index(bigram_key, add=add_to_indexer)
                if feature_idx != -1:
                    if feature_idx in features:
                        features[feature_idx] += 1
                    else:
                        features[feature_idx] = 1

            # === PUNCTUATION FEATURES ===
            if word in self.punctuation:
                punct_key = f"PUNCT={word}"
                feature_idx = self.indexer.add_and_get_index(punct_key, add=add_to_indexer)
                if feature_idx != -1:
                    if feature_idx in features:
                        features[feature_idx] += 1
                    else:
                        features[feature_idx] = 1

            # === POSITION FEATURES ===
            # First and last words often carry sentiment
            if i == 0:
                first_key = f"FIRST={word_lower}"
                feature_idx = self.indexer.add_and_get_index(first_key, add=add_to_indexer)
                if feature_idx != -1:
                    features[feature_idx] = 1
            if i == len(sentence) - 1:
                last_key = f"LAST={word_lower}"
                feature_idx = self.indexer.add_and_get_index(last_key, add=add_to_indexer)
                if feature_idx != -1:
                    features[feature_idx] = 1

        # === LENGTH FEATURES ===
        # Bin sentence length
        length = len(sentence)
        if length < 5:
            length_bin = "very_short"
        elif length < 10:
            length_bin = "short"
        elif length < 20:
            length_bin = "medium"
        elif length < 30:
            length_bin = "long"
        else:
            length_bin = "very_long"

        length_key = f"LEN={length_bin}"
        feature_idx = self.indexer.add_and_get_index(length_key, add=add_to_indexer)
        if feature_idx != -1:
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
        Predict the sentiment of a sentence.
        :param sentence: words in the sentence to classify
        :return: 0 for negative, 1 for positive
        """
        # Extract features (add_to_indexer=False at test time)
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

    def __init__(self):
        raise Exception("Must be implemented")


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    import random

    # First pass: extract features from all training examples to build the indexer
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    # Initialize weight vector (numpy array)
    num_features = len(feat_extractor.get_indexer())
    weights = np.zeros(num_features)

    # Training parameters
    num_epochs = 10

    # Question 2: Schedule options:
    # 1. "constant": lr = 1.0 (default perceptron)
    # 2. "epoch_decay": lr = initial_lr * (decay_factor ** epoch)

    schedule = "epoch_decay"
    initial_lr = 1.0
    # for epoch_decay
    decay_factor = 0.9

    # Train for multiple epochs
    for epoch in range(num_epochs):
        # Randomly shuffle training examples each epoch
        random.shuffle(train_exs)

        # Set learning rate based on schedule
        if schedule == "constant":
            lr = initial_lr
        elif schedule == "epoch_decay":
            lr = initial_lr * (decay_factor ** epoch)

        # Iterate through each training example
        for ex in train_exs:
            # Extract features
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)

            # Compute prediction (dot product)
            score = 0.0
            for feature_idx, feature_val in features.items():
                score += weights[feature_idx] * feature_val

            # Predicted label
            y_pred = 1 if score >= 0 else 0

            # True label
            y_true = ex.label

            # Perceptron update: if prediction is wrong, update weights
            if y_pred != y_true:
                # If true label is 1 but predicted 0, add features to weights
                # If true label is 0 but predicted 1, subtract features from weights
                update_direction = 1 if y_true == 1 else -1

                for feature_idx, feature_val in features.items():
                    weights[feature_idx] += lr * update_direction * feature_val

    return PerceptronClassifier(weights, feat_extractor)


def train_logistic_regression(train_exs: List[SentimentExample],
                              feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    raise Exception("Must be implemented")


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

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model