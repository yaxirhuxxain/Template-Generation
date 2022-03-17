import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ReformerConfig, ReformerLayer
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.reformer.modeling_reformer import _ReversibleFunction

# Defining CPU GPU Devices.
GPU = torch.device('cuda')
CPU = torch.device('cpu')


class PositionEmbedding(nn.Module):
    """Constructs position embeddings of shape `[max_pos_embed, hidden_size]`."""

    def __init__(self, hidden_size=300, max_pos_embed=4096, n_drop=0.05, is_training=True):
        super().__init__()
        self.n_drop = n_drop
        self.is_training = is_training
        self.embedding = nn.Embedding(max_pos_embed, hidden_size)

    def forward(self, pos_ids):
        position_embeddings = self.embedding(pos_ids)
        position_embeddings = nn.functional.dropout(
            position_embeddings, p=self.n_drop, training=self.is_training)
        return position_embeddings


class EmbeddingBlock(nn.Module):
    """Construct embeddings from words and positions."""

    def __init__(self, vocab_size, hidden_size=300, max_pos_embed=4096, n_drop=0.05, is_training=True):
        super().__init__()
        self.n_drop = n_drop
        self.is_training = is_training
        self.max_pos_embed = max_pos_embed

        self.words = nn.Embedding(vocab_size, hidden_size)
        self.positions = PositionEmbedding(
            hidden_size, max_pos_embed, n_drop, is_training=is_training)

    def forward(self, input_ids=None, start_idx=0):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        pos_ids = torch.arange(start_idx, start_idx +
                               seq_length, dtype=torch.long, device=GPU)
        pos_ids = pos_ids.unsqueeze(0).expand(input_shape)

        words_embedding = self.words(input_ids)

        if pos_ids.shape[-1] > self.max_pos_embed:
            raise ValueError(
                f"seq_length: {pos_ids.shape[-1]} can not be grater than "
                f"max_pos_embed: {self.self.max_pos_embed}."
            )

        # dropout
        embeddings = nn.functional.dropout(
            words_embedding, p=self.n_drop, training=self.is_training)

        # add positional embeddings
        position_embeddings = self.positions(pos_ids)
        embeddings = embeddings + position_embeddings
        return embeddings


class ReformerBlock(nn.Module):
    """
        Reformer Block implementation is borrowed from Transformers.
        Article: https://arxiv.org/abs/2001.04451
    """

    def __init__(self, vocab_size, ff_size=1200, hidden_size=300, n_layers=6, n_heads=6, n_drop=0.05, is_training=True, config=ReformerConfig()):
        super().__init__()

        self.n_drop = n_drop
        self.is_training = is_training

        config.vocab_size = vocab_size
        config.hidden_size = hidden_size
        config.feed_forward_size = ff_size
        config.num_hidden_layers = n_layers
        config.num_attention_heads = n_heads

        self.layers = nn.ModuleList(
            [ReformerLayer(config, i) for i in range(config.num_hidden_layers)])
        # Reformer is using Rev Nets, thus last layer outputs are concatenated and
        # Layer Norm is done over 2 * hidden_size
        self.layer_norm = nn.LayerNorm(
            2 * config.hidden_size, eps=config.layer_norm_eps)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            num_hashes=None,
            past_buckets_states=None,
            use_cache=False,
            orig_sequence_length=None,
            output_hidden_states=False,
            output_attentions=False,
    ):
        # hidden_states and attention lists to be filled if wished
        all_hidden_states = []
        all_attentions = []

        # init cached hidden states if necessary
        if past_buckets_states is None:
            past_buckets_states = [((None), (None))
                                   for i in range(len(self.layers))]

        # concat same tensor for reversible ResNet
        hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
        hidden_states = _ReversibleFunction.apply(
            hidden_states,
            self.layers,
            attention_mask,
            head_mask,
            num_hashes,
            all_hidden_states,
            all_attentions,
            past_buckets_states,
            use_cache,
            orig_sequence_length,
            output_hidden_states,
            output_attentions,
        )

        # Apply layer norm to concatenated hidden states
        hidden_states = self.layer_norm(hidden_states)

        # Apply dropout
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.n_drop, training=self.is_training)

        return hidden_states


class LSTMBlock(nn.Module):
    """LSTM Cell Block"""

    def __init__(self, vocab_size, hidden_size=300, embed_weights=None, n_drop=0.05, is_training=True):
        super().__init__()
        self.n_drop = n_drop
        self.is_training = is_training

        self.embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=0, _weight=embed_weights, device=GPU)
        self.rnn_cell = nn.LSTMCell(
            hidden_size, hidden_size, device=GPU, bias=False)
        self.dense = nn.Linear(2 * hidden_size, hidden_size, bias=False)

    def forward(self, input_ids, hidden_states):
        embed = self.embeddings(input_ids)  # batch_size * seq_len * n_embed

        # dropout
        embed = nn.functional.dropout(
            embed, p=self.n_drop, training=self.is_training)

        # Prepare the shape for LSTMCell
        embed = embed.permute(1, 0, 2)  # seq_len * batch_size * n_embed

        embed = embed[-1]  # we consider last timestamp as input

        # we consider the 2nd last state as ht & ct from ReformerBlock
        ht = self.dense(hidden_states[:, -2])
        ct = self.dense(hidden_states[:, -2])

        out, _ = self.rnn_cell(embed, (ht, ct))

        # Apply dropout
        # out = nn.functional.dropout(out, p=self.n_drop, training=self.is_training)

        return out


class ClassificationBlock(nn.Module):
    def __init__(self, embed_weights):
        super(ClassificationBlock, self).__init__()
        self.set_embeddings_weights(embed_weights)

    def set_embeddings_weights(self, embed_weights):
        embed_shape = embed_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = embed_weights  # Tied weights

    def forward(self, hidden_state):
        lm_logits = self.decoder(hidden_state)
        return lm_logits


class ReformerLSTM(nn.Module):
    def __init__(
            self,
            vocab_size,
            hidden_size=300,
            ff_size=1200,
            max_pos_embed=4096,
            n_layers=6,
            n_heads=6,
            n_drop=0.05,
            is_training=True
    ):
        super(ReformerLSTM, self).__init__()
        self.n_layers = n_layers

        self.embeddings = EmbeddingBlock(
            vocab_size, hidden_size, max_pos_embed, n_drop, is_training)
        embed_weights = self.embeddings.words.weight

        self.reformer = ReformerBlock(
            vocab_size, ff_size, hidden_size, n_layers, n_heads, n_drop, is_training)
        self.lstm = LSTMBlock(vocab_size, hidden_size,
                              embed_weights, n_drop, is_training)
        self.classifier = ClassificationBlock(embed_weights)

        self.loss_fn = F.cross_entropy

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, labels=None, head_mask=None):
        embed = self.embeddings(input_ids)

        # prepare head mask
        head_mask = [None] * self.n_layers

        hidden_states = self.reformer(embed, head_mask=head_mask)
        out = self.lstm(input_ids, hidden_states)
        logits = self.classifier(out)

        loss = None
        if labels is not None:
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)), labels.view(-1))

        if not loss:
            return logits
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
