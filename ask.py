"""Load the model checkpoint and generate an answer."""

from __future__ import annotations

import argparse

import torch

from model import TinyTransformer
from qa import answer_from_facts
from train import DATA, MODEL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+", help="Question about Indian presidents")
    parser.add_argument("--model-only", action="store_true", help="Bypass reliable fact lookup and demonstrate generation")
    args = parser.parse_args()
    question = " ".join(args.question)
    if not args.model_only:
        import json

        records = json.loads(DATA.read_text())["presidents"]
        answer = answer_from_facts(question, records)
        if answer is not None:
            print(answer)
            return
        print("I cannot answer that from the presidents dataset.")
        return

    checkpoint = torch.load(MODEL, map_location="cpu", weights_only=True)
    chars = checkpoint["chars"]
    stoi = {char: i for i, char in enumerate(chars)}
    model = TinyTransformer(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    prompt = f"Question: {question}\nAnswer:"
    unknown = sorted(set(prompt) - set(chars))
    if unknown:
        raise SystemExit(f"Question contains characters outside the training vocabulary: {unknown}")
    tokens = torch.tensor([[stoi[c] for c in prompt]])
    output = "".join(chars[i] for i in model.generate(tokens)[0].tolist())
    answer = output[len(prompt):].split("###", 1)[0].strip()
    print(answer)


if __name__ == "__main__":
    main()

