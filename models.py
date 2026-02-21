# models.py

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
from collections import defaultdict
from nltk.metrics.distance import edit_distance
from typing import List, Dict, Optional
from sentiment_data import *

class SentimentClassifier(object):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]

class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        return 1

class SpellingCorrector:
    """
    Fast spelling corrector using a prefix index.
    Since typos never occur in the first 3 characters, we only need to search
    vocabulary words that share the same 3-character prefix as the unknown word.
    This reduces edit distance calls from O(|V|) to O(small bucket).
    """
    def __init__(self, word_indexer, min_word_len: int = 4):
        self.word_indexer = word_indexer
        self.min_word_len = min_word_len
        self.cache: Dict[str, str] = {}

        # Build prefix index: first 3 chars -> list of vocab words
        self.prefix_index: Dict[str, List[str]] = defaultdict(list)
        for i in range(len(word_indexer)):
            word = word_indexer.get_object(i)
            if word and len(word) >= 3 and word not in ("PAD", "UNK"):
                self.prefix_index[word[:3]].append(word)

    def correct(self, word: str) -> Optional[str]:
        """
        Returns the best spelling correction for an OOV word, or None if no
        good candidate is found. Only attempts correction on words long enough
        to plausibly have a typo.
        """
        if len(word) < self.min_word_len:
            return None
        if word in self.cache:
            return self.cache[word]

        prefix = word[:3]
        candidates = self.prefix_index.get(prefix, [])
        if not candidates:
            return None

        best_word = None
        best_dist = 3  # only accept corrections within edit distance 2
        for candidate in candidates:
            # Quick length filter before computing edit distance
            if abs(len(candidate) - len(word)) > 2:
                continue
            dist = edit_distance(word, candidate)
            if dist < best_dist:
                best_dist = dist
                best_word = candidate

        self.cache[word] = best_word
        return best_word

class DANNetwork(nn.Module):
    def __init__(self, word_embeddings: WordEmbeddings,
                 hidden_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.3):
        super(DANNetwork, self).__init__()

        embed_dim = word_embeddings.get_embedding_length()
        self.embedding = word_embeddings.get_initialized_embedding_layer(
            frozen=False,
            padding_idx=0
        )

        layers = []
        input_dim = embed_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_size

        layers.append(nn.Linear(hidden_size, 2))
        layers.append(nn.LogSoftmax(dim=-1))
        self.network = nn.Sequential(*layers)

        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, word_indices: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(word_indices)       # (batch, seq_len, embed_dim)
        mask = (word_indices != 0).unsqueeze(-1)    # (batch, seq_len, 1)
        masked_embeds = embeds * mask
        summed = masked_embeds.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        avg_embeds = summed / lengths            
        return self.network(avg_embeds)
    
class NeuralSentimentClassifier(SentimentClassifier):
    def __init__(self, model: DANNetwork, word_indexer, spelling_corrector: SpellingCorrector = None):
        self.model = model
        self.word_indexer = word_indexer
        self.spelling_corrector = spelling_corrector
        self.model.eval()

    def _lookup_word(self, word: str, has_typos: bool) -> int:
        """Look up a word index, optionally attempting spelling correction for OOV words."""
        idx = self.word_indexer.index_of(word)
        if idx != -1:
            return idx
        # Word is OOV — try spelling correction if in typo mode
        if has_typos and self.spelling_corrector is not None:
            corrected = self.spelling_corrector.correct(word)
            if corrected is not None:
                idx = self.word_indexer.index_of(corrected)
                if idx != -1:
                    return idx
        return self.word_indexer.index_of("UNK")

    def _words_to_indices(self, words: List[str], has_typos: bool, max_len: int = 50) -> torch.Tensor:
        words = words[:max_len]
        indices = [self._lookup_word(w, has_typos) for w in words]
        if not indices:
            indices = [self.word_indexer.index_of("UNK")]
        return torch.tensor(indices, dtype=torch.long)

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        device = next(self.model.parameters()).device
        with torch.no_grad():
            indices = self._words_to_indices(ex_words, has_typos).unsqueeze(0).to(device)
            log_probs = self.model(indices)
            return int(torch.argmax(log_probs, dim=1).item())

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        device = next(self.model.parameters()).device
        sequences = [self._words_to_indices(words, has_typos) for words in all_ex_words]
        padded = pad_sequence(sequences, batch_first=True, padding_value=0).to(device)
        with torch.no_grad():
            log_probs = self.model(padded)
            preds = torch.argmax(log_probs, dim=1)
        return preds.cpu().tolist()

BATCH_SIZE = 32
LR = 0.001


def train_deep_averaging_network(args,
                                 train_exs: List[SentimentExample],
                                 dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings,
                                 train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_size = args.hidden_size
    num_epochs = args.num_epochs
    lr = LR
    batch_size = BATCH_SIZE

    model = DANNetwork(word_embeddings, hidden_size=hidden_size, num_layers=2, dropout=0.3)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.NLLLoss()

    word_indexer = word_embeddings.word_indexer
    max_len = 50

    def words_to_indices(words):
        words = words[:max_len]
        indices = []
        for w in words:
            idx = word_indexer.index_of(w)
            if idx == -1:
                idx = word_indexer.index_of("UNK")
            indices.append(idx)
        if not indices:
            indices = [word_indexer.index_of("UNK")]
        return torch.tensor(indices, dtype=torch.long)

    train_data = [(words_to_indices(ex.words), torch.tensor(ex.label, dtype=torch.long))
                  for ex in train_exs]

    def collate_fn(batch):
        sequences, labels = zip(*batch)
        padded = pad_sequence(sequences, batch_first=True, padding_value=0)
        labels = torch.stack(labels)
        return padded.to(device), labels.to(device)

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_inputs, batch_labels in train_loader:
            optimizer.zero_grad()
            log_probs = model(batch_inputs)
            loss = loss_fn(log_probs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} — avg loss: {avg_loss:.4f}")

    # Build spelling corrector only when needed for typo setting
    spelling_corrector = SpellingCorrector(word_indexer) if train_model_for_typo_setting else None

    return NeuralSentimentClassifier(model, word_indexer, spelling_corrector)