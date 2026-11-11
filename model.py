import torch
import torch.nn as nn
from torchcrf import CRF

class TokenFeatureEncoder(nn.Module):

    def __init__(self, use_features, use_sizes):
        super().__init__()

        self.use_features = use_features
        self.use_sizes = use_sizes

        self.capitalization = nn.Embedding(5, 8)
        self.length = nn.Embedding(5, 4)
        self.has_digit = nn.Embedding(2, 4)
        self.all_digits = nn.Embedding(2, 4)
        self.has_hyphen = nn.Embedding(2, 4)
        self.has_punctuation = nn.Embedding(2, 4)

        self.size = nn.Sequential(
            nn.Linear(1, 4),
            # nn.ReLU(),
        )

        self.output_dim = (
            self.capitalization.embedding_dim + self.length.embedding_dim +
            self.has_digit.embedding_dim + self.all_digits.embedding_dim +
            self.has_hyphen.embedding_dim + self.has_punctuation.embedding_dim +
            4
        )


    def forward(self, features, sizes):
        categorical_x = torch.cat([
            self.capitalization(features[..., 0]),
            self.length(features[..., 1]),
            self.has_digit(features[..., 2]),
            self.all_digits(features[..., 3]),
            self.has_hyphen(features[..., 4]),
            self.has_punctuation(features[..., 5]),
        ], dim=-1)
        size_x = self.size(sizes.unsqueeze(-1))

        if not self.use_features:
            categorical_x = categorical_x * 0
        if not self.use_sizes:
            size_x = size_x * 0

        return torch.cat([categorical_x, size_x], dim=-1)


class BiLSTMCRF(nn.Module):

    def __init__(self, encoder, hidden_dim, num_tags, constraints=None, dropout=0.0, use_features=False, use_style=False):
        super().__init__()

        self.encoder = encoder
        self.feature_encoder = TokenFeatureEncoder(use_features, use_style)

        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=encoder.output_dim + self.feature_encoder.output_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Linear(hidden_dim * 2, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

        self.constraints = constraints

        if constraints is not None:
            self.apply_constraints()

    def emissions(self, input_ids, features, sizes, attention_mask=None):
        x = self.encoder(input_ids, attention_mask)
        feature_x = self.feature_encoder(features, sizes)
        x = torch.cat([x, feature_x], dim=-1)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def loss(self, input_ids, tags, mask, features, sizes, attention_mask=None):
        emissions = self.emissions(input_ids, features, sizes, attention_mask)
        return -self.crf(emissions, tags, mask=mask, reduction="mean")

    def decode(self, input_ids, mask, features, sizes, attention_mask=None):
        emissions = self.emissions(input_ids, features, sizes, attention_mask)
        return self.crf.decode(emissions, mask=mask)

    def apply_constraints(self):
        if self.constraints is None:
            return

        transition_mask, start_mask, end_mask = self.constraints

        with torch.no_grad():
            self.crf.transitions.data = torch.maximum(
                self.crf.transitions.data,
                transition_mask.to(self.crf.transitions.device)
            )

            self.crf.start_transitions.data = torch.maximum(
                self.crf.start_transitions.data,
                start_mask.to(self.crf.start_transitions.device)
            )

            self.crf.end_transitions.data = torch.maximum(
                self.crf.end_transitions.data,
                end_mask.to(self.crf.end_transitions.device)
            )
