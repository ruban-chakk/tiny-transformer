"""Download and clean the presidents table from Wikipedia."""

from __future__ import annotations

import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

URL = "https://en.wikipedia.org/wiki/List_of_presidents_of_India"
OUT = Path("data/presidents.json")


def clean(value: str) -> str:
    value = re.sub(r"\[[^]]*]", "", value)
    return " ".join(value.replace("\xa0", " ").split())


def ingest() -> dict:
    response = requests.get(
        URL,
        timeout=30,
        headers={"User-Agent": "tiny-transformer-learning-project/1.0"},
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    table = next(
        table
        for table in soup.select("table")
        if "Term of office" in table.get_text(" ") and "Vice president" in table.get_text(" ")
    )
    presidents = []
    for row in table.select("tr"):
        cells = [clean(cell.get_text(" ", strip=True)) for cell in row.select("th, td")]
        # Rows contain a portrait column and Wikipedia uses rowspans, so parse dates
        # and names by their shapes instead of relying on fragile fixed indices.
        dates = [x for x in cells if re.fullmatch(r"\d{1,2} [A-Z][a-z]+ \d{4}|Incumbent", x)]
        if len(dates) < 2:
            continue
        name = next((x for x in cells if re.search(r"[A-Za-z]", x) and not re.search(r"\d", x)), "")
        if not name or name in {"Portrait", "Name"}:
            continue
        presidents.append({
            "name": name.rstrip("†‡* "),
            "start": dates[0],
            "end": dates[1],
            "acting": any(x in {"–", "-"} for x in cells),
        })

    # Preserve distinct acting and elected terms but remove parser duplicates.
    unique = list({(p["name"], p["start"]): p for p in presidents}.values())
    payload = {"source": URL, "presidents": unique}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {len(unique)} terms to {OUT}")
    return payload


if __name__ == "__main__":
    ingest()

