# Tiny Transformer

This project builds a small GPT-style transformer from scratch (using only basic
PyTorch layers) and trains it on facts extracted from the [Wikipedia list of
presidents of India](https://en.wikipedia.org/wiki/List_of_presidents_of_India).
It is a learning model, not a production fact system.

## Answering modes

Tiny Transformer deliberately provides two answering modes:

- **Default mode** reads `data/presidents.json` and returns exact answers for
  recognized question types. It refuses unsupported questions rather than
  inventing an answer.
- **Model-only mode** runs the trained transformer and generates characters from
  its learned weights. This mode demonstrates neural inference and can produce
  incorrect or invented text.

Use `--model-only` only when you specifically want to observe the neural model:

```bash
python ask.py --model-only "Who served before Pranab Mukherjee?"
```

## Documentation

- [Architecture and detailed model walkthrough](docs/ARCHITECTURE.md) explains
  ingestion, tokenization, attention, training, checkpoints, inference, tensor
  shapes, trade-offs, and extension points.

## Run the complete pipeline

Use Python 3.11 or newer with a currently supported PyTorch wheel.

The generated dataset and checkpoint are intentionally not stored in Git. Every
fresh clone must run both ingestion and training before asking questions.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python ingest.py
python train.py --steps 1500
python ask.py "Who was the first president of India?"
```

Training takes a few minutes on Apple Silicon or a GPU and longer on a CPU. For a
quick mechanics-only run, use `python train.py --steps 20 --batch-size 4`; its
answers will not be learned yet.

## What happens at each stage?

1. **Ingestion** — `ingest.py` downloads HTML, finds the presidents table, cleans
   its text, and writes structured facts to `data/presidents.json`.
2. **Tokenization** — `train.py` turns facts into many question/answer examples.
   Every character is mapped to an integer token. Real LLMs usually use subword
   tokens instead, but character tokens make the mechanism completely visible.
3. **Training** — the model sees 128-token windows and predicts the next token.
   Backpropagation and AdamW repeatedly adjust embeddings, attention, and MLP
   weights to reduce prediction error.
4. **Checkpoint** — `torch.save` writes a `.pt` file containing the architecture
   settings, character vocabulary, and 112,064 learned floating-point parameters.
   The original Wikipedia sentences are not stored as a searchable database.
5. **Answering** — `ask.py` rebuilds the architecture, loads those weights, encodes
   the question, and repeatedly predicts one next character. For normal use it
   first performs a reliable lookup in the structured facts and refuses unsupported
   questions. Pass `--model-only` to study raw transformer generation.

## Ask supported questions reliably

`ask.py` recognizes multiple phrasings for first/current president, predecessor,
successor, term dates, start/end dates, acting status, chronological position,
counts, lists, and the president serving on a given date. Relative questions can
include offsets such as `2 terms before` or `three presidents after`.

```bash
python ask.py "Who served before Pranab Mukherjee?"
# Pratibha Patil.

python ask.py "Who was president on 1 January 2005?"
# A. P. J. Abdul Kalam.

python ask.py "What was the president's favourite food?"
# I cannot answer that from the presidents dataset.
```

## Inspect model vectors

Display slices of the actual embeddings, Q/K/V vectors, attention weights, block
outputs, logits, and next-character probabilities computed for a question:

```bash
python inspect_vectors.py "Who was the first president of India?"
```

Inspect another token position or export complete tensors as JSON:

```bash
python inspect_vectors.py --position 10 --values 16 \
  --json vectors.json "Who was the first president of India?"
```

The JSON file can be large because attention contains one value for every visible
prompt position.

## The transformer itself

For each input token, the model adds a learned token embedding and positional
embedding. In every block, causal self-attention creates Query, Key, and Value
vectors. `softmax(QKᵀ / √d)V` lets each position combine useful earlier positions,
while a triangular mask prevents looking into the future. A feed-forward network
then transforms each position. Residual connections and layer normalization keep
training stable. The final linear layer produces one score per possible next
character.

## Honest limitations

- This dataset teaches memorization, not broad language understanding.
- Generated answers can still be wrong; neural checkpoints are not databases.
- Wikipedia can change, so rerun ingestion and training to refresh knowledge.
- For reliable real-world QA, use retrieval-augmented generation (RAG): search the
  source at question time and give the retrieved passage to a pretrained LLM.

## Generated files

The following local outputs are excluded from Git:

```text
data/presidents.json
artifacts/presidents_transformer.pt
```

Recreate them at any time:

```bash
python ingest.py
python train.py --steps 1500
```

## License

Tiny Transformer is available under the [MIT License](LICENSE).

