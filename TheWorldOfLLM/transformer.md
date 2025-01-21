# Transformers

LLMs are build upon transformer model, which was first introduced in the following paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

The first lines of the Abstract goes as follows:

> The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

> Transduction models in machine learning are designed to predict specific cases directly from observed cases without building a general predictive model. This approach contrasts with inductive learning, where a general model is built from training data and then applied to new, unseen data.

## Need for the Transformer

The Need for the transformers arose due to the limitations of Recurrent neural networks, long short-term memory and gated recurrent neural networks for sequence modeling and transduction problems such as language modeling and machine translation. Some of the advancements in this model were:

1. **Parllelization**: Recurrent models generate a sequence of hidden states ht, as a function of the previous hidden state ht−1 and the input for position t. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples.

2. **Attention Mechanism**: RNNs struggle with long-term dependencies, which is must for language tasks. RNNs struggle with Vanishing gradient problem, Even the latter varients LSTM and GRU paritally mitigtate the problem. Attention mechanisms has shown great results in sequence and transduction problems, as they allow modeling of dependencies regardless to their distance in the input or output sequences. Transformer combines both recurrence and attention instead of relying entirely on an attention mechanism to draw global dependencies between input and output. It is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequencealigned RNNs or convolution.

3. **Reduced Memory Constriants**: Unlike the RNN which stores all the hidden states for the given state which leads to high memory usage while back propogation, Transfomers use Self-Attention Mechansim which doesn't need to store the hidden states.

## Transformer Architecture

![Transformer Architecure](images\transformer-architecture.png)

The model architecture contains stacks of encoders and decoders, where the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence
of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive, i.e. cosuming the previous outputs as the additional input while generating the next.

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves.

1. **Attention**: Attention is the mapping of Query, Key and Value vectors to an output. The output is the weighted sum of the values, where the weights are assigned by a compatiblity function of the query to its corresponding key. Two types of attention are employed

   ![Transformer Architecure](images\transformer-attention.png)

   a. **Scaled Dot Product Attention**: Computes the attention score as the dot product of Queries and Keys scaled by $\sqrt{d_k}$ (dimensions of the key matrix). Then softmax is applied to obtain the weights, further these weights are multiplied with Values to obtain the attentions.
   $$Attention(Q, K, V) = sofmax(\frac{QK^T}{\sqrt{d_k}}) V$$
   For large values of d<sub>k</sub>, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. Hence the dot product is scaled by $\frac{1}{\sqrt{d_k}}$.

   b. **Multi-Head Attention**: Instead of running single attention function with d<sub>model</sub>, The queries, key, values are projected h times (h = 8) to linear projections to dk, dk and dv, and the attention function is run on these projectios in parallel and concatenated later for the final output. Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

   ```
   class MultiHeadAttention(nn.Module):
       def __init__(self, d_model, num_heads):
           super().__init__()
           assert d_model % num_heads == 0, "d_models must be divisible by num_heads"

           self.d_model = d_model
           self.num_heads = num_heads
           self.d_k = d_model // num_heads

           self.W_q = nn.Linear(d_model, d_model)
           self.W_k = nn.Linear(d_model, d_model)
           self.W_v = nn.Linear(d_model, d_model)
           self.W_o = nn.Linear(d_model, d_model)

       def scaled_dot_product_attention(self, Q, K, V, mask = None):
           attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
           if mask is not None:
               attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
           attn_probs = torch.softmax(attn_scores, dim=-1)
           output = torch.matmul(attn_probs, V)
           return output

       def split_heads(self, x):
           batch_size, seq_length, d_model = x.size()
           x = x.view(batch_size, seq_length, self.num_heads, self.d_k)
           x = x.transpose(1, 2)
           return x

       def combine_heads(self, x):
           batch_size, _ , seq_length, d_k = x.size()
           x = x.transpose(1, 2)
           x = x.contiguous()
           x = x.view(batch_size, seq_length, self.d_model)
           return x

       def forward(self, Q, K, V, mask = None):
           Q = self.split_heads(self.W_q(Q))
           K = self.split_heads(self.W_k(K))
           V = self.split_heads(self.W_v(V))

           attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
           attn_output = self.combine_heads(attn_output)
           attn_output = self.W_o(attn_output)
           return attn_output
   ```

2. **Point wise Feed-Forward Networks**: Along with the sublayers, encoder and decoder layers are connected with Feed Forward Neural Network, which consists of two linear transformations with a Relu function as follows:

   $FFN(x) = max(0, xW_1 + b_1)W_2 + b_2$

```
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.W_1(x))
        x = self.W_2(x)
        return x
```

3. **Positional Encoding**: As the model doesn't contains any reccurance and convolutions, to preserve the information of the ordering of the sequence, positional embeddings are added to the inputs of both encoder and decoder stacks. Transformer model use, sin and cosine functions of deffernt frequencies.

   $PE_{(pos, 2i)} = sin(pos/10000^{2i/d_{model}})$

   $PE_{(pos, 2i + 1)} = cos(pos/10000^{2i/d_{model}})$

where pos is the position and i is the dimension. each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from 2π to 10000 · 2π. These functions are used as they would allow the model to easily learn to attend by relative positions, since for any fixed offset k, PE<sub>pos+k</sub> can be represented as a linear function of PE<sub>pos</sub>.

```
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
```

4. **Encoder Layer**: The encoder is a stack of identical layers (6 for the base model). Each layer has two sub-layers, The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. The output of each sublayer is combined with the residual connection and then latter send to the Layer Normalization. `LayerNorm(x + Sublayer(x))`. To maintain consistancy for residual connections and embedding layers across all the sub layers, the output dim is d<sub>model</sub> = 512.

```
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        out1 = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(out1)
        out2 = self.norm2(out1 + self.dropout(ff_output))
        return out2
```

5. **Decoder Layer**: The decoder is also a stack of identical layers (6 for the base model). In addition to the two sub-layers of the encoder, it consits a new sub layer which applies Multi Head attention to the output of the Encoder stack. Similar to encoder layer, residual connection and layer normalisation are used. The Self Attention layer is masked in the decoder, to prevent positions for looking into subsequent positions, this ensures the output of the current position depend only on the known outputs of the positions less than current position.

```
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
```

Finally Transformer Module to bringing all the blocks together

```
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_enbedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoderList = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoderList = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoderList:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoderList:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
```

## (Additional) Understanding Self Attention

A great resource to understand Attention mechansiam is [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch/tree/main) by Sebastian Raschka.

Refer to this github links: [Understanding Self Attention](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/01_main-chapter-code/ch03.ipynb)
