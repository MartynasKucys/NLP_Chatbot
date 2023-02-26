import spacy
import numpy as np

import torch
import torch.nn as nn


def tokenize_to_vec(text, nlp_model):
    doc = nlp_model(text)

    useful_tokens = []

    for token in doc:
        if not (token.is_punct and token.is_stop):
        # if not (token.is_punct):
            useful_tokens.append(token.lemma_)

    doc = nlp_model(" ".join(useful_tokens))

    return doc.vector


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)

        out = self.l2(out)
        out = self.relu(out)

        out = self.l3(out)

        return out

