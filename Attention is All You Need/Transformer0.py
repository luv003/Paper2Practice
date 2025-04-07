import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class Attention(nn.Module):
    def __init__(self, query, key, value, mask=None, dropout=None):
        super().__init__()
        self.query = query
        self.key = key
        self.value = value
        self.mask = mask
        self.dropout = dropout

    def forward(self):
        d_k = self.query.size(-1)
        scores = torch.matmul(self.query, self.key.transpose(-2, -1)) / math.sqrt(d_k)
        if self.mask is not None:
            scores = scores.masked_fill(self.mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, self.value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        query, key, value = [
            linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linear_layers, (query, key, value))
        ]
        attention = Attention(query, key, value, mask=mask, dropout=self.dropout)
        x, _ = attention()
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, self_attn: MultiHeadAttention, feed_forward: PositionwiseFeedForward, dropout: float):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.output_linear.in_features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, self_attn: MultiHeadAttention, src_attn: MultiHeadAttention, feed_forward: PositionwiseFeedForward, dropout: float):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask))
        x = x + self.dropout(self.src_attn(self.norm2(x), memory, memory, src_mask))
        x = x + self.dropout(self.feed_forward(self.norm3(x)))
        return x

class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.output_linear.in_features)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int = 512, N: int = 6, h: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.src_embed = nn.Sequential(nn.Embedding(src_vocab, d_model), PositionalEncoding(d_model))
        self.tgt_embed = nn.Sequential(nn.Embedding(tgt_vocab, d_model), PositionalEncoding(d_model))
        encoder_layer = EncoderLayer(d_model, MultiHeadAttention(h, d_model, dropout), PositionwiseFeedForward(d_model, d_ff, dropout), dropout)
        self.encoder = Encoder(encoder_layer, N)
        decoder_layer = DecoderLayer(d_model, MultiHeadAttention(h, d_model, dropout),
                                     MultiHeadAttention(h, d_model, dropout),
                                     PositionwiseFeedForward(d_model, d_ff, dropout), dropout)
        self.decoder = Decoder(decoder_layer, N)
        self.out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(self.src_embed(src), src_mask)
        output = self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        return F.log_softmax(self.out(output), dim=-1)

def create_masks(src: torch.Tensor, tgt: torch.Tensor, pad: int = 0):
    src_mask = (src != pad).unsqueeze(-2)
    if tgt is not None:
        tgt_mask = (tgt != pad).unsqueeze(-2)
        size = tgt.size(1)
        nopeak_mask = torch.triu(torch.ones((1, size, size), device=tgt.device), diagonal=1).bool()
        tgt_mask = tgt_mask & ~nopeak_mask
    else:
        tgt_mask = None
    return src_mask, tgt_mask

if __name__ == '__main__':
    src_vocab = 10000
    tgt_vocab = 10000
    d_model = 512
    N = 6
    h = 8
    d_ff = 2048
    dropout = 0.1

    model = Transformer(src_vocab, tgt_vocab, d_model, N, h, d_ff, dropout)

    batch_size = 64
    src_seq_len = 20
    tgt_seq_len = 20
    src = torch.randint(0, src_vocab, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab, (batch_size, tgt_seq_len))

    src_mask, tgt_mask = create_masks(src, tgt)

    output = model(src, tgt, src_mask, tgt_mask)
    print("Output shape:", output.shape)
