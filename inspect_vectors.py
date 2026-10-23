"""Inspect intermediate vectors produced by the trained transformer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from model import TinyTransformer
from train import MODEL


def preview(tensor: torch.Tensor, count: int) -> list[float]:
    return [round(value, 6) for value in tensor.detach().cpu().flatten()[:count].tolist()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+", help="Question whose vectors should be inspected")
    parser.add_argument("--position", type=int, default=-1, help="Token position to inspect; default is the final token")
    parser.add_argument("--values", type=int, default=8, help="Number of values shown per vector")
    parser.add_argument("--json", type=Path, help="Write the complete inspected tensors to a JSON file")
    args = parser.parse_args()

    checkpoint = torch.load(MODEL, map_location="cpu", weights_only=True)
    chars = checkpoint["chars"]
    stoi = {char: index for index, char in enumerate(chars)}
    model = TinyTransformer(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    prompt = f"Question: {' '.join(args.question)}\nAnswer:"
    unknown = sorted(set(prompt) - set(chars))
    if unknown:
        raise SystemExit(f"Question contains characters outside the training vocabulary: {unknown}")
    tokens = torch.tensor([[stoi[char] for char in prompt]])
    if tokens.shape[1] > model.context:
        raise SystemExit(f"Prompt has {tokens.shape[1]} tokens; model context is {model.context}.")

    time = tokens.shape[1]
    position = args.position if args.position >= 0 else time + args.position
    if not 0 <= position < time:
        raise SystemExit(f"Position must be between 0 and {time - 1}.")

    with torch.no_grad():
        token_vectors = model.token_embedding(tokens)
        position_vectors = model.position_embedding(torch.arange(time))
        hidden = token_vectors + position_vectors
        report: dict[str, torch.Tensor] = {
            "token_embedding": token_vectors[0, position],
            "position_embedding": position_vectors[position],
            "combined_embedding": hidden[0, position],
        }

        for block_index, block in enumerate(model.blocks):
            normalised = block.ln1(hidden)
            batch, length, width = normalised.shape
            q, k, v = block.attention.qkv(normalised).chunk(3, dim=-1)
            shape = (batch, length, block.attention.heads, block.attention.head_size)
            q, k, v = (item.view(shape).transpose(1, 2) for item in (q, k, v))
            scores = q @ k.transpose(-2, -1) / block.attention.head_size**0.5
            scores = scores.masked_fill(~block.attention.mask[:length, :length], float("-inf"))
            weights = F.softmax(scores, dim=-1)

            report[f"block_{block_index}.query_head_0"] = q[0, 0, position]
            report[f"block_{block_index}.key_head_0"] = k[0, 0, position]
            report[f"block_{block_index}.value_head_0"] = v[0, 0, position]
            report[f"block_{block_index}.attention_head_0"] = weights[0, 0, position]
            hidden = block(hidden)
            report[f"block_{block_index}.output"] = hidden[0, position]

        logits = model.output(model.norm(hidden))[0, position]
        probabilities = F.softmax(logits, dim=-1)
        report["final_logits"] = logits
        report["final_probabilities"] = probabilities

    print(f"Prompt tokens: {time}")
    print(f"Inspected position: {position} ({prompt[position]!r})")
    print(f"Model width: {model.config['width']}; heads: {model.config['heads']}; vocabulary: {len(chars)}")
    for name, tensor in report.items():
        print(f"{name:34s} shape={str(list(tensor.shape)):12s} values={preview(tensor, args.values)}")

    top_probabilities, top_indices = torch.topk(probabilities, min(5, len(chars)))
    print("Top next-character predictions:")
    for probability, index in zip(top_probabilities.tolist(), top_indices.tolist()):
        print(f"  {chars[index]!r}: {probability:.6f}")

    if args.json:
        payload = {
            "prompt": prompt,
            "position": position,
            "character": prompt[position],
            "tensors": {name: tensor.detach().cpu().tolist() for name, tensor in report.items()},
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Complete tensors written to {args.json}")


if __name__ == "__main__":
    main()

