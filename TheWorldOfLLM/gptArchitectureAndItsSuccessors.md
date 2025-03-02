# GPT Architecture and its successors

The GPT architecture has been popular for its wide range of applications on different natural language tasks s such as textual entailment, question answering, semantic similarity assessment, and document classification.

The model proved that various NLP tasks can be realized by generative pre-training of a language model on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each specific task. It uses a semi-supervised approch for language understanding tasks using a combination of unsupervised pre-training and supervised fine-tuning. The goal was to adopt a universal representation so that which can be used with small modifications for a large variety of tasks.

The paper uses the multi-layer Tranformer Decoder architecture.

## Original GPT Architecture - [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

The training procedure is as follows:

1.  **Unsupervised Learning**: This step is aimed to understand the sematics of the language which can be genealized to any task. Given the tokens $U = {u_1, u_2, ...., u_n}$, the model tries to maximize the likehood probability of the next token, denoted as

    $L_1(U) = \sum_i \log P(u_i|u_{i-k}, ..., u_{i-1};\theta)$

    where k is the context window, P is the conditional probability modelled on the model parameters $\theta$. These parameters are trained using stochastic gradient descent.

    The model uses mulit-head attention operation over the inputs followed by position-wise feedforward layers to generate the output distribution.

    $h_0 = UW_e + W_p$

    $h_l = tranfomer_block(h_0)$

    $P(u) = softmax(h_n W_e^T)$

    where $U = (u_{-k}, ..., u_{-1})$ is the context vector of tokens, n is the number of layers, W<sub>e</sub> is the token
    embedding matrix, and W<sub>p</sub> is the position embedding matrix.

2.  **Supervised fine-tuning**: After pre-training the model with the corpus, the model is finetuned for specific tasks. In this step, labeled dataset C is used. The inputs are passed through our pre-trained model to obtain
    the final transformer block’s activation $h_m^l$, which is then fed into an added linear output layer with parameters Wy to predict y:

        $P(y|x_1, . . . , x_m) = softmax(h_m^l W_y)$

        This gives us the following objective to maximize:

        $L_2(C) = \sum_{(x,y)} log P(y|x_1, . . . , x_m)$

        The paper includes language modeling as an auxiliary objective to the fine-tuning helped learning by improving generalization of the supervised model and accelerating convergence. Hence, the final Optmization function becomes

        $L_3(C) = L_2(C) + \lambda ∗ L_1(C)$

        The extra parameters during finetuning stage are W<sub>y</sub>, and embeddings for delimiter tokens.

For training the model BooksCorpus dataset containing 7000 books of various geners are used.

The architecture contains 12 layer decoder only model with masked self-attention heads are used (768 dimensional states and 12
attention heads). For the position-wise feed-forward networks, 3072 dimensional inner states are sued. The model used the Adam optimization scheme with a max learning rate of 2.5e-4. The learning rate was increased linearly from zero over the first 2000 updates and annealed to 0 using a cosine schedule. This means, initially the lr is set to 0 and increased linearly for the first 2000 updates to its max value and then the lr is decreased to 0, in the cosine manner, i.e. it decreases slowly in start and rapidly later. The model is trained for 100 epochs on minibatches of 64 randomly sampled, contiguous sequences of 512 tokens. Since layernorm is used extensively throughout the model, a simple weight initialization of N(0, 0.02) was sufficient. Bytepair encoding (BPE) vocabulary with 40,000 merges and residual, embedding, and attention dropouts with a rate of 0.1 for regularization was used. A modified version of L2 regularizatio, with w = 0.01 on all non bias or gain weights. For the activation function, Gaussian Error Linear Unit (GELU) is use. Learned position embeddings instead of the sinusoidal version proposed in the original work.

While fine-tuning the hyperparameter settings from pre-training are reused. Dropout of 0.1, learning rate of 6.25e-5 and batchsize of 32 are used. The model finetunes quickly and 3 epochs of training was sufficient for most cases. A linear learning rate decay schedule with warmup over 0.2% of training. λ was set to 0.5.

## GPT - 2 [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

The paper demonstarates that Large Language model begins to learn the down stream tasks in a zero-shot setting – without any parameter or architecture modification. The capacity of the language model is essential to the success of zero-shot task transfer and increasing it improves performance in a log-linear fashion across tasks.

The model is trained on the new dataset WebText containing millions of web pages Reddit, Dragnet and Newspaper dataset.

The model used in this paper is similar to the GPT paper with few modifications,

1. Layer Normalization is moved to the input of each sub-block and to the final self-attention block.
2. A modified initialization which accounts for the accumulation on the residual path with model depth is used. At initiliazation, the weights of residual layers is scaled by $1/\sqrt{N}$
3. The vocabulary is expanded to 50,257.
4. The context size is increased from 512 to 1024 tokens.
5. A larger batchsize of 512 is used.

The paper has released models of sizes 117M, 345M, 762M and 1.54B. The smallest model 117M is much similar to the GPT Model. The largest model has shown sota results in 7 of the 8 tasks without any explicit supervised learning.

```
GPT_CONFIG_124M  = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qvk_bias": False
}
```

```
# multi head attention for GPT Module

class MultiHeadAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, dropout, num_heads, qvk_bias = False):
    super().__init__()
    assert (d_out % num_heads) == 0, "d_out must be divisible by num_heads"

    self.d_out = d_out
    self.num_heads = num_heads
    self.head_dim = d_out // num_heads
    self.W_query = nn.Linear(d_in, d_out, bias = qvk_bias)
    self.W_key = nn.Linear(d_in, d_out, bias = qvk_bias)
    self.W_value = nn.Linear(d_in, d_out, bias = qvk_bias)
    self.out_proj = nn.Linear(d_out, d_out)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer("mask", torch.tril(torch.ones(context_length, context_length)))

  def forward(self, x):
    b, num_tokens, d_in = x.shape
    queries = self.W_query(x)
    keys = self.W_key(x)
    values = self.W_value(x)

    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
    keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
    values = values.view(b, num_tokens, self.num_heads, self.head_dim)

    queries = queries.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)

    attn_scores = torch.matmul(queries, keys.transpose(2, 3))
    masked_attn_scores = attn_scores.masked_fill_(self.mask == 0, -torch.inf)
    attn_weights = nn.functional.softmax(masked_attn_scores/(keys.shape[-1] ** 0.5), dim = -1)
    attn_weights = self.dropout(attn_weights)
    context_vector = torch.matmul(attn_weights, values).transpose(1, 2)
    context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)
    context_vector = self.out_proj(context_vector)
    return context_vector
```

```
# implementing the FFNN

class FeedForward(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
        nn.GELU(),
        nn.Linear(4 * config["emb_dim"], config["emb_dim"])
    )

  def forward(self, x):
    return self.layers(x)
```

```
# implementing Transformer Block

class TransformerBlock(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.attn = MultiHeadAttention(
        d_in = config["emb_dim"],
        d_out = config["emb_dim"],
        context_length = config["context_length"],
        num_heads = config["n_heads"],
        dropout = config["drop_rate"],
        qvk_bias = config["qvk_bias"]
    )
    self.ffn = FeedForward(config)
    self.norm1 = nn.LayerNorm(config["emb_dim"])
    self.norm2 = nn.LayerNorm(config["emb_dim"])
    self.dropout_skip = nn.Dropout(config["drop_rate"])

  def forward(self, x):
    skip = x
    x = self.norm1(x)
    x = self.attn(x)
    x = self.dropout_skip(x)
    x = x + skip

    skip = x
    x = self.norm2(x)
    x = self.ffn(x)
    x = self.dropout_skip(x)
    x = x + skip
    return x
```

```
# implementing the GPTModel

class GPTModel(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
    self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"])
    self.dropout = nn.Dropout(config["drop_rate"])
    self.transformer_blocks = nn.Sequential(
        *[TransformerBlock(config) for _ in range(config["n_layers"])]
    )
    self.final_norm = nn.LayerNorm(config["emb_dim"])
    self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

  def forward(self, in_idx):
    batch, seq_len = in_idx.shape
    tok_embeds = self.tok_emb(in_idx)
    pos_embeds = self.pos_emb(torch.arange(seq_len))
    x = tok_embeds + pos_embeds
    x = self.dropout(x)
    x = self.transformer_blocks(x)
    x = self.final_norm(x)
    logits = self.out_head(x)
    return logits
```

## GPT - 3 [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

The paper introduces GPT-3, an autoregressive language model with 175B parameters, which significantly surpasses its predecessors. The research demonstrates that scaling up language models enhances task-agnostic, few-shot performance, achieving results comparable to state-of-the-art fine-tuned models without the need for gradient updates or fine-tuning.

GPT-3 is designed to utilize a large number of parameters to improve language understanding and generation capabilities. The model was trained on a diverse dataset, including a filtered version of Common Crawl, Wikipedia, and internet-based book corpora, resulting in 570GB of data after filtering.

The GPT-3 model uses the same architecture as the GPT-2 model, including the modified initialization, pre-normalization and reversable tokenization. Additional to it, the model uses alternating dense and locally banded sparse attention patterns in the layers of the transformer just like Sparse Transformer. 

A unique training process was implemented to ensure high-quality data extraction for Common Crawl dataset, involving fuzzy deduplication and prioritization of high-quality datasets.

Despite the strong quantitative and qualitative improvements of GPT-3, particularly compared to its direct predecessor GPT-2, it still has notable weaknesses in text synthesis and several NLP tasks. 
1. On text synthesis, although the overall quality is high, GPT-3 samples still sometimes repeat themselves semantically at the document level, start to lose coherence over sufficiently long passages, contradict themselves, and occasionally contain non-sequitur sentences or paragraphs.
2. Within the domain of discrete language tasks, we have noticed informally that GPT-3 seems to have special difficulty with “common sense physics”. 
3. GPT-3 cannot effectively handle ambiguous queries or questions that require contextual knowledge beyond its training data. 
4. GPT-3's massive size (175 billion parameters) makes it extremely resource-intensive to train and deploy. Running inference at scale can also be prohibitively expensive. 
5. GPT-3 can generate harmful, biased, or unethical content due to biases in its training data. It may also be misused to create convincing misinformation or spam. 
6. As with other large neural networks, the inner workings of GPT-3 are largely a "black box," making it difficult to interpret how or why it generates specific outputs.



