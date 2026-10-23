# Tiny Transformer Architecture

This document explains how Tiny Transformer turns one Wikipedia table into a
small question-answering model. It covers the application pipeline, the neural
network, training, the checkpoint file, inference, limitations, and extension
points. Every detail below maps to the current Python implementation.

## 1. System purpose

Tiny Transformer is a teaching project. It makes the full lifecycle of a
GPT-style model visible without relying on a pretrained language model:

1. Download facts from Wikipedia.
2. Convert those facts into question-and-answer examples.
3. Convert characters into integer tokens.
4. Train a causal transformer to predict answer characters.
5. Save the architecture, vocabulary, and learned parameters.
6. Load that checkpoint and generate an answer one character at a time.

The project learns a narrow set of facts about presidents of India. It is not a
general-purpose language model or a reliable production knowledge base.

## 2. Component architecture

```text
┌───────────────────────────────────────────────────────────────────────┐
│ Wikipedia: List of presidents of India                               │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ HTTPS / HTML
                                v
┌───────────────────────────────────────────────────────────────────────┐
│ ingest.py                                                             │
│ requests -> BeautifulSoup -> row cleanup -> acting-term detection     │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ JSON facts
                                v
┌───────────────────────────────────────────────────────────────────────┐
│ data/presidents.json                                                  │
│ source + [{name, start, end, acting}, ...]                            │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ make_examples()
                                v
┌───────────────────────────────────────────────────────────────────────┐
│ train.py                                                              │
│ QA templates -> character vocabulary -> batches -> answer-only loss  │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ token and target tensors
                                v
┌───────────────────────────────────────────────────────────────────────┐
│ model.py: TinyTransformer                                             │
│ embeddings -> 2 transformer blocks -> layer norm -> vocabulary logits │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ torch.save(...)
                                v
┌───────────────────────────────────────────────────────────────────────┐
│ artifacts/presidents_transformer.pt                                  │
│ config + state_dict + character vocabulary + source URL              │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ torch.load(...)
                                v
┌───────────────────────────────────────────────────────────────────────┐
│ ask.py + qa.py                                                        │
│ structured fact lookup, or optional raw transformer generation        │
└───────────────────────────────────────────────────────────────────────┘
```

The ingestion and training paths are offline stages. Asking a question does not
contact Wikipedia. It uses only the saved checkpoint.

## 3. Repository map

| Path | Responsibility |
| --- | --- |
| `ingest.py` | Downloads and normalizes the source table. |
| `data/presidents.json` | Stores the cleaned facts used for training. |
| `train.py` | Builds examples, tokenizes them, trains the model, and saves a checkpoint. |
| `model.py` | Defines attention, transformer blocks, forward propagation, and generation. |
| `artifacts/presidents_transformer.pt` | Stores everything required to reconstruct the trained model. |
| `qa.py` | Recognizes supported intents and answers exactly from JSON facts. |
| `ask.py` | Uses reliable fact lookup by default or raw model generation with `--model-only`. |
| `inspect_vectors.py` | Displays or exports embeddings, attention vectors, hidden states, and output scores. |
| `requirements.txt` | Declares HTTP, parsing, array, and neural-network dependencies. |

`data/`, `artifacts/`, and `.venv/` are ignored by Git because they are generated
locally.

## 4. Data ingestion

Run:

```bash
python ingest.py
```

`ingest()` performs these operations:

1. `requests.get()` downloads the configured `URL` with a 30-second timeout.
2. `raise_for_status()` rejects unsuccessful HTTP responses.
3. Beautiful Soup parses the HTML.
4. The code selects the table containing both `Term of office` and
   `Vice president` rather than relying on its position on the page.
5. `clean()` removes citation markers, non-breaking spaces, and repeated
   whitespace.
6. A date-pattern search extracts the start and end dates from each row.
7. The first non-numeric textual cell becomes the president's name.
8. A dash in the election field marks an acting term.
9. `(name, start)` removes parser duplicates while keeping V. V. Giri's two
   distinct terms.
10. The result is written as UTF-8 JSON.

The generated record schema is:

```json
{
  "name": "A. P. J. Abdul Kalam",
  "start": "25 July 2002",
  "end": "25 July 2007",
  "acting": false
}
```

### Ingestion boundary

The JSON file is an intentional boundary between web scraping and machine
learning. Training remains reproducible when the network is unavailable, and a
bad scrape can be inspected before it reaches the model.

The main risk is upstream HTML change. If Wikipedia changes its table wording or
row structure, selection or field inference can fail. There are currently no
schema-validation tests, so inspect the reported record count and JSON after a
source change.

## 5. From facts to training examples

`make_examples()` converts each term into up to three templates:

```text
Question: When did A. P. J. Abdul Kalam serve as president of India?
Answer: 25 July 2002 to 25 July 2007.
###

Question: Who became president of India on 25 July 2002?
Answer: A. P. J. Abdul Kalam.
###

Question: Who served after K. R. Narayanan?
Answer: A. P. J. Abdul Kalam.
###
```

The `###` sequence is an end-of-answer convention. The model learns to emit it;
`ask.py` removes everything from that separator onward before displaying output.

Three extra examples teach the first president, current president, and number of
elected presidents. The elected count excludes records whose `acting` field is
true. A fixed shuffle seed makes example ordering repeatable.

## 6. Character tokenization

The project uses characters as tokens:

```text
training text -> unique sorted characters -> integer IDs
```

For example, if `A` maps to 7 and `n` maps to 38, `Answer` becomes a sequence of
integer IDs. `chars` maps IDs back to characters, while the local `stoi` mapping
converts characters to IDs.

Character tokenization was chosen for transparency:

- The vocabulary can be built with a few lines of Python.
- No external tokenizer file or algorithm is hidden from the learner.
- Names and dates can be represented without an unknown-word token.

The trade-off is longer sequences and slower learning. Real language models
usually use subword tokens that represent common fragments or whole short words.
This implementation also rejects a question containing a character absent from
the training vocabulary.

## 7. Tensor shapes and model configuration

The default model configuration is:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `vocab_size` | data-dependent | Number of unique training characters. |
| `context` | 128 | Maximum number of tokens visible at once. |
| `width` | 64 | Size of each token representation. |
| `heads` | 4 | Parallel attention heads. |
| `head_size` | 16 | Per-head width: `64 / 4`. |
| `layers` | 2 | Number of transformer blocks. |
| `dropout` | 0.1 | Training-time probability of dropping activations. |

For a batch size `B` and sequence length `T`, the main shapes are:

```text
Input token IDs                 [B, T]
Token embeddings               [B, T, 64]
Position embeddings            [T, 64]
Combined representations       [B, T, 64]
Q, K, V before head split      [B, T, 64] each
Q, K, V after head split       [B, 4, T, 16] each
Attention scores               [B, 4, T, T]
Joined attention output        [B, T, 64]
Vocabulary logits              [B, T, vocab_size]
```

## 8. Transformer internals

### 8.1 Embeddings

The input is an integer tensor. `token_embedding` looks up a learned 64-number
vector for each character. `position_embedding` looks up another vector for each
position from 0 through 127. The model adds these vectors:

```text
x[token, position] = token_embedding[token] + position_embedding[position]
```

Token embeddings describe what a character is. Position embeddings tell the
network where it occurs. Without position information, attention alone would not
distinguish different orderings of the same characters.

### 8.2 Query, Key, and Value projections

`CausalSelfAttention.qkv` is one linear layer that produces three representations
for every position, then `chunk(3)` separates them:

- Query: what information this position is looking for.
- Key: what information this position can be matched by.
- Value: what information this position contributes when selected.

Each representation is divided across four heads. Different heads can learn
different relationships, such as question phrasing, name structure, dates, or the
boundary between prompt and answer.

### 8.3 Scaled dot-product attention

Attention scores are calculated as:

```text
scores = Q × Kᵀ / sqrt(head_size)
weights = softmax(mask(scores))
result = weights × V
```

`Q × Kᵀ` measures how relevant every earlier position is to the current position.
Dividing by the square root of the head size prevents dot products from becoming
so large that softmax is difficult to train. Softmax converts scores into weights
that sum to one.

### 8.4 Causal mask

The registered triangular mask permits a position to see itself and earlier
positions, but not future positions:

```text
             key position
             0  1  2  3
query 0      ✓  ·  ·  ·
query 1      ✓  ✓  ·  ·
query 2      ✓  ✓  ✓  ·
query 3      ✓  ✓  ✓  ✓
```

Disallowed scores become negative infinity before softmax, so their attention
weight becomes zero. This makes training match inference: when generating the next
character, the model cannot use characters that have not been generated yet.

### 8.5 Transformer block

Each `Block` uses pre-normalization and two residual paths:

```text
input
  │
  ├───────────────┐
  v               │
LayerNorm         │ residual connection
  v               │
Self-Attention    │
  v               │
  + <─────────────┘
  │
  ├───────────────┐
  v               │
LayerNorm         │ residual connection
  v               │
Linear 64 -> 256  │
  v               │
GELU              │
  v               │
Linear 256 -> 64  │
  v               │
Dropout           │
  v               │
  + <─────────────┘
```

Attention mixes information across positions. The feed-forward network transforms
each position independently. Residual connections preserve the existing signal
and give gradients a direct path through the network. Layer normalization keeps
activation scales controlled.

Two blocks repeat this process. A final layer normalization and linear output
layer produce one score, called a logit, for every vocabulary character.

### 8.6 Weight tying and initialization

The output layer shares its weight matrix with the token embedding:

```python
self.output.weight = self.token_embedding.weight
```

The same character representation is therefore used when reading tokens and when
scoring output characters. This reduces the parameter count and often improves
small language models. Linear and embedding weights start from a normal
distribution with standard deviation `0.02`; linear biases start at zero.

## 9. Training

Run the default training job:

```bash
python train.py --steps 1500 --batch-size 32
```

### 9.1 Device selection

Training selects the first available device in this order:

```text
Apple MPS -> NVIDIA CUDA -> CPU
```

The checkpoint is portable because inference loads it onto the CPU by default.

### 9.2 Building a batch

At every step, `random.choices()` samples question-answer examples with
replacement. Two tensors of shape `[batch_size, 128]` are created:

- `x` contains input character IDs.
- `y` contains the character each position should predict next.

Unused target positions contain `-100`. PyTorch cross-entropy ignores that value.
Targets belonging to the question are also replaced with `-100`, so the loss is
calculated only for the answer and its separator.

This answer-only objective means the question is context, not something the model
is rewarded for reproducing. It resembles supervised instruction tuning on a very
small scale.

### 9.3 Forward pass and loss

The model produces logits shaped `[B, T, vocab_size]`. Both logits and targets are
flattened across batch and time before cross-entropy:

```text
loss = -log(probability assigned to each correct answer character)
```

A lower loss means the model assigns more probability to the expected answer
characters. Low training loss does not prove that paraphrased or unseen questions
will be answered correctly.

### 9.4 Backpropagation and optimization

Each training step performs:

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

`loss.backward()` applies the chain rule from the output back through the output
layer, both transformer blocks, attention projections, and embeddings. Every
parameter receives a gradient describing how changing it would change the loss.
AdamW uses those gradients and a learning rate of `3e-4` to update the parameters.

The program prints loss at step 1 and every 100 steps. `torch.manual_seed(42)`
makes PyTorch initialization and PyTorch operations more repeatable, although
Python's training-time `random.choices()` is not explicitly seeded in `main()`.

## 10. The model checkpoint

Training saves `artifacts/presidents_transformer.pt` with four entries:

| Entry | Contents | Why it is needed |
| --- | --- | --- |
| `config` | Vocabulary size, context, width, heads, layers, dropout | Reconstructs the exact architecture. |
| `state_dict` | Named learned tensors for embeddings, attention, MLPs, and normalization | Contains what training learned. |
| `chars` | Ordered character vocabulary | Keeps token IDs consistent between training and inference. |
| `source` | Wikipedia URL | Records data provenance. |

The checkpoint does not contain executable model source code. `ask.py` must still
import `TinyTransformer` from `model.py`, instantiate it with `config`, and load
`state_dict`. The names and shapes in the state dictionary must match the code.

The checkpoint also does not contain a searchable copy of Wikipedia. Knowledge is
distributed across floating-point weights. That is why a learned model can produce
plausible but incorrect text and why updating one fact requires retraining here.

## 11. Inference and answer generation

Run:

```bash
python ask.py "Who was the first president of India?"
```

The default inference sequence is:

```text
CLI question -> normalize wording -> identify intent and entity/date
             -> read exact answer from presidents.json
             -> print answer, or refuse an unsupported question
```

This lookup layer prevents a tiny generative model from inventing answers. To
study neural inference directly, pass `--model-only`. That sequence is:

```text
CLI words
   -> "Question: ...\nAnswer:"
   -> validate characters against checkpoint vocabulary
   -> convert characters to integer IDs
   -> forward pass
   -> select highest-logit next character
   -> append the character
   -> repeat up to 120 times
   -> convert IDs back to text
   -> stop displayed answer at ###
```

Generation is autoregressive: every predicted character becomes part of the next
forward pass. Only the newest 128 tokens are retained if the sequence grows beyond
the context window.

The default `temperature=0.0` uses `argmax`, so it always selects the highest-score
character and produces deterministic output. A positive temperature divides the
logits before sampling. Low values concentrate probability on likely characters;
higher values produce more variety and more mistakes.

`model.eval()` disables dropout. `torch.no_grad()` prevents gradient tracking,
reducing inference memory and computation.

## 12. End-to-end worked example

Suppose the JSON contains:

```json
{"name":"Rajendra Prasad","start":"26 January 1950","end":"13 May 1962","acting":false}
```

Training turns it into examples such as:

```text
Question: Who became president of India on 26 January 1950?
Answer: Rajendra Prasad.
###
```

Characters become integer IDs and enter the transformer. During training, the
network sees the question and preceding answer characters. It receives loss only
when predicting the answer and separator. Repeated gradient updates alter the
weights until `Rajendra Prasad.` becomes likely after that prompt pattern.

At inference, the prompt ends at `Answer:`. The model predicts a space, then `R`,
then `a`, continuing one character at a time. When it eventually emits `###`, the
CLI prints only the text before the separator.

## 13. Failure modes and operational limits

### Unsupported characters

Because the vocabulary is derived from training text, an emoji or unseen accented
character terminates `ask.py` with a clear error. A production tokenizer normally
includes byte-level fallback or a defined unknown token.

### Unseen wording

The training set contains a few fixed templates. Questions with very different
wording may fail even if they ask for a known fact. More paraphrased examples or a
pretrained model would improve this.

### Hallucination

The generator must always choose a next character. It has no confidence threshold
or `I do not know` behavior, so out-of-domain questions can receive plausible
nonsense.

### Context limit

The architecture supports at most 128 input positions per forward pass. Training
rejects examples longer than 129 characters including the shifted target.
Generation discards older positions after the context is full.

### Source drift

The page can change after training. Inference never checks for updates. Refreshing
facts requires rerunning ingestion and training.

### No evaluation split

All generated examples are used for training. The reported loss measures
memorization, not generalization. A stronger project would keep validation
questions and paraphrases separate from training.

### Checkpoint trust

Only load checkpoints you trust. `weights_only=True` reduces Python object loading
risk, but model artifacts should still be treated as supplied data and validated.

## 14. Design trade-offs

| Choice | Benefit | Cost |
| --- | --- | --- |
| Train from scratch | Exposes the complete learning pipeline. | Cannot acquire broad language ability from one page. |
| Character tokens | Simple and inspectable. | Longer sequences and slower training. |
| Decoder-only transformer | Shows GPT-style causal generation. | Less direct than retrieval for factual lookup. |
| Synthetic QA templates | Produces training supervision without manual labeling. | Encourages template memorization. |
| Answer-only loss | Focuses limited capacity on useful output. | Does not teach general modeling of the prompt language. |
| Greedy generation | Repeatable answers and easy debugging. | Cannot explore alternative outputs. |
| Local JSON boundary | Separates scraping from training. | Data becomes stale until ingestion is rerun. |
| One checkpoint file | Simple distribution and loading. | Architecture code is still required. |

## 15. How to extend the system

### Add another question style

Add a `(question, answer)` pair inside `make_examples()`, then retrain. Ensure the
formatted example stays within the 128-character context window.

### Add more source facts

Extend the JSON schema and ingestion parser, then generate templates using those
fields. Changes to data alone do not affect an existing checkpoint.

### Use subword tokenization

Replace the `chars`/`stoi` logic with a tokenizer library. Save the tokenizer model
or complete vocabulary in the checkpoint and use the same tokenizer in `ask.py`.

### Add evaluation

Split examples before training, train only on one portion, and periodically run
the model against held-out questions. Include paraphrases, unsupported questions,
and exact factual checks.

### Move toward production factual QA

Use retrieval-augmented generation instead of encoding every fact into weights:

```text
question -> retrieve matching source rows -> provide evidence to a pretrained
model -> answer with citation
```

That architecture updates facts without retraining and can show where an answer
came from. Tiny Transformer intentionally does not use it because the goal is to
teach how transformer weights are trained and used.

## 16. Command reference

### Ingest

```bash
python ingest.py
```

Downloads the fixed Wikipedia URL and overwrites `data/presidents.json`.

### Train

```bash
python train.py [--steps INTEGER] [--batch-size INTEGER]
```

Defaults:

- `--steps 1500`
- `--batch-size 32`

Overwrites `artifacts/presidents_transformer.pt` after training completes.

### Ask

```bash
python ask.py [--model-only] "QUESTION"
```

By default, reads the JSON dataset and prints an exact answer for a recognized
intent. Unsupported questions receive a clear refusal. `--model-only` loads the
fixed checkpoint and generates an answer for educational comparison. Shell quoting
is recommended, although the CLI joins multiple arguments with spaces.

## 17. Verification

After training, verify the core learned facts:

```bash
python ask.py "Who was the first president of India?"
python ask.py "Who is the current president of India?"
python ask.py "When did A. P. J. Abdul Kalam serve as president of India?"
python ask.py "How many elected presidents of India are listed?"
```

The trained checkpoint created with this project returned:

```text
Rajendra Prasad.
Droupadi Murmu.
25 July 2002 to 25 July 2007.
15.
```

These checks prove that checkpoint loading and known-template generation work.
They do not establish accuracy on unseen phrasings or facts.

