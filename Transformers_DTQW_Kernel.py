import torch
from torch import nn
from torch.nn import functional as F
from longformer.sliding_chunks import sliding_chunks_matmul_qk, sliding_chunks_matmul_pv

class SparseAttention(nn.Module):
    """
    Sparse Attention class that implements the sliding window-based sparse attention.
    """
    def __init__(self, attention_window, hidden_size, num_heads):
        super().__init__()
        self.attention_window = attention_window
        self.hidden_size = hidden_size
        self.num_heads = num_heads

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.size()

        if seq_len % (self.attention_window * 2) != 0:
            raise ValueError("Sequence length must be a multiple of attention window size")

        def reshape(input_tensor):
            return input_tensor.view(batch_size, seq_len, self.num_heads, self.hidden_size // self.num_heads).transpose(1, 2)

        query, key, value = map(reshape, (query, key, value))

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        attn_output_weights = sliding_chunks_matmul_qk(query, key, self.attention_window, padding_value=0, mask=mask)
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)

        attn_output = sliding_chunks_matmul_pv(attn_output_weights, value, self.attention_window)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        return attn_output


class TransformerLayer(nn.Module):
    """
    Transformer Layer class that includes a Sparse Attention module.
    """
    def __init__(self, hidden_size, num_heads, attention_window, feedforward_dim, dropout):
        super().__init__()

        self.self_attn = SparseAttention(attention_window, hidden_size, num_heads)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_size),
        )

    def forward(self, src, mask=None):
        attn_output = self.self_attn(src, src, src, mask)
        src = self.norm1(src + self.dropout(attn_output))

        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_output))

        return src

class TransformerModel(nn.Module):
    """
    Transformer model that incorporates a TransformerLayer with Sparse Attention.
    """
    def __init__(self, ntoken, hidden_size, nhead, nlayers, attention_window, feedforward_dim, dropout=0.5):
        super(TransformerModel, self).__init__()

        self.model_type = 'Transformer'
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(ntoken, hidden_size)

        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_size, nhead, attention_window, feedforward_dim, dropout) 
            for _ in range(nlayers)
        ])

        self.decoder = nn.Linear(hidden_size, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, mask=None):
        embedded = self.dropout(self.embedding(input))

        output = embedded
        for transformer_layer in self.transformer_layers:
            output = transformer_layer(output, mask)

        output = self.decoder(output)

        return F.log_softmax(output, dim=-1)
import torch
from torch import nn
from torch.nn import functional as F
from longformer.sliding_chunks import sliding_chunks_matmul_qk, sliding_chunks_matmul_pv

class SparseAttention(nn.Module):
    """
    Sparse Attention class that implements the sliding window-based sparse attention.
    """
    def __init__(self, attention_window, hidden_size, num_heads):
        super().__init__()
        self.attention_window = attention_window
        self.hidden_size = hidden_size
        self.num_heads = num_heads

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.size()

        if seq_len % (self.attention_window * 2) != 0:
            raise ValueError("Sequence length must be a multiple of attention window size")

        def reshape(input_tensor):
            return input_tensor.view(batch_size, seq_len, self.num_heads, self.hidden_size // self.num_heads).transpose(1, 2)

        query, key, value = map(reshape, (query, key, value))

        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        attn_output_weights = sliding_chunks_matmul_qk(query, key, self.attention_window, padding_value=0, mask=mask)
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)

        attn_output = sliding_chunks_matmul_pv(attn_output_weights, value, self.attention_window)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        return attn_output


class TransformerLayer(nn.Module):
    """
    Transformer Layer class that includes a Sparse Attention module.
    """
    def __init__(self, hidden_size, num_heads, attention_window, feedforward_dim, dropout):
        super().__init__()

        self.self_attn = SparseAttention(attention_window, hidden_size, num_heads)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_size),
        )

    def forward(self, src, mask=None):
        attn_output = self.self_attn(src, src, src, mask)
        src = self.norm1(src + self.dropout(attn_output))

        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_output))

        return src

class TransformerModel(nn.Module):
    """
    Transformer model that incorporates a TransformerLayer with Sparse Attention.
    """
    def __init__(self, ntoken, hidden_size, nhead, nlayers, attention_window, feedforward_dim, dropout=0.5):
        super(TransformerModel, self).__init__()

        self.model_type = 'Transformer'
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(ntoken, hidden_size)

        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_size, nhead, attention_window, feedforward_dim, dropout) 
            for _ in range(nlayers)
        ])

        self.decoder = nn.Linear(hidden_size, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, mask=None):
        embedded = self.dropout(self.embedding(input))

        output = embedded
        for transformer_layer in self.transformer_layers:
            output = transformer_layer(output, mask)

        output = self.decoder(output)

        return F.log_softmax(output, dim=-1)
