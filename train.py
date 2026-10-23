"""Create training examples, train the transformer, and save its learned weights."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from model import TinyTransformer

DATA = Path("data/presidents.json")
MODEL = Path("artifacts/presidents_transformer.pt")


def make_examples(records: list[dict]) -> list[str]:
    examples = []
    elected = [r for r in records if not r.get("acting", False)]
    for i, row in enumerate(records):
        name, start, end = row["name"], row["start"], row["end"]
        pairs = []
        for template in (
            "When did {name} serve as president of India?",
            "What was {name}'s term of office?",
            "What was the presidential tenure of {name}?",
            "From when to when was {name} president?",
        ):
            pairs.append((template.format(name=name), f"{start} to {end}."))
        for template in (
            "Who became president of India on {start}?",
            "Who took office as president on {start}?",
            "Which president started serving on {start}?",
        ):
            pairs.append((template.format(start=start), f"{name}."))
        pairs.append((f"Was {name} an acting president?", "Yes." if row.get("acting") else "No."))
        if i + 1 < len(records):
            successor = records[i + 1]["name"]
            for template in ("Who served after {name}?", "Who succeeded {name}?", "Who was {name}'s successor?"):
                pairs.append((template.format(name=name), f"{successor}."))
        if i > 0:
            predecessor = records[i - 1]["name"]
            for template in ("Who served before {name}?", "Who preceded {name}?", "Who was {name}'s predecessor?"):
                pairs.append((template.format(name=name), f"{predecessor}."))
        for question, answer in pairs:
            examples.append(f"Question: {question}\nAnswer: {answer}\n###\n")
    examples.extend([
        "Question: Who was the first president of India?\nAnswer: Rajendra Prasad.\n###\n",
        "Question: Who is the current president of India?\nAnswer: Droupadi Murmu.\n###\n",
        f"Question: How many elected presidents of India are listed?\nAnswer: {len(elected)}.\n###\n",
        f"Question: How many acting presidential terms are listed?\nAnswer: {sum(r.get('acting', False) for r in records)}.\n###\n",
    ])
    random.Random(42).shuffle(examples)
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    random.seed(42)
    torch.manual_seed(42)
    records = json.loads(DATA.read_text())["presidents"]
    examples = make_examples(records)
    chars = sorted(set("".join(examples)))
    stoi = {char: i for i, char in enumerate(chars)}
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyTransformer(len(chars)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(1, args.steps + 1):
        selected = random.choices(examples, k=args.batch_size)
        x = torch.zeros((args.batch_size, model.context), dtype=torch.long)
        y = torch.full((args.batch_size, model.context), -100, dtype=torch.long)
        for row, example in enumerate(selected):
            ids = torch.tensor([stoi[c] for c in example], dtype=torch.long)
            if len(ids) > model.context + 1:
                raise ValueError(f"Example exceeds context window: {example}")
            x[row, : len(ids) - 1] = ids[:-1]
            y[row, : len(ids) - 1] = ids[1:]
            # Ignore question/padding loss; optimize only the answer and separator.
            answer_at = example.index("\nAnswer:") + len("\nAnswer:") - 1
            y[row, :answer_at] = -100
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(f"step {step:4d} | loss {loss.item():.4f} | device {device}")

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config, "state_dict": model.state_dict(), "chars": chars,
                "source": json.loads(DATA.read_text())["source"]}, MODEL)
    print(f"Saved checkpoint to {MODEL} ({MODEL.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()

