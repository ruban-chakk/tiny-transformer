"""Reliable question answering over the structured presidents dataset."""

from __future__ import annotations

import re
from datetime import datetime


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _find_record(question: str, records: list[dict]) -> tuple[int, dict] | None:
    normalised = _normalise(question)
    matches = [
        (index, record)
        for index, record in enumerate(records)
        if _normalise(record["name"]) in normalised
    ]
    return max(matches, key=lambda item: len(item[1]["name"]), default=None)


def _date_in_question(question: str) -> datetime | None:
    match = re.search(r"\b(\d{1,2} [A-Za-z]+ \d{4})\b", question)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%d %B %Y")
    except ValueError:
        return None


def _relative_distance(question: str) -> int:
    """Extract a requested relative offset, defaulting to one term."""
    q = _normalise(question)
    match = re.search(r"\b(\d+)\s+(?:terms?\s+|presidents?\s+)?(?:before|after)\b", q)
    if match:
        return int(match.group(1))
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    match = re.search(
        r"\b(" + "|".join(words) + r")\s+(?:terms?\s+|presidents?\s+)?(?:before|after)\b",
        q,
    )
    return words[match.group(1)] if match else 1


def answer_from_facts(question: str, records: list[dict]) -> str | None:
    """Return an exact answer for a recognized intent, otherwise ``None``."""
    q = _normalise(question)
    elected = [record for record in records if not record.get("acting", False)]

    if "first" in q and "president" in q:
        return f"{elected[0]['name']}."
    if any(word in q for word in ("current", "present", "incumbent")) and "president" in q:
        return f"{records[-1]['name']}."

    if any(phrase in q for phrase in ("how many", "number of", "total")):
        if "acting" in q:
            return f"{sum(record.get('acting', False) for record in records)}."
        if "term" in q:
            return f"{len(records)}."
        if "president" in q:
            return f"{len(elected)}."

    if ("list" in q or "name all" in q) and "president" in q:
        return ", ".join(record["name"] for record in elected) + "."

    ordinal = re.search(r"\b(\d+)(?:st|nd|rd|th)\b", q)
    if ordinal and "president" in q:
        position = int(ordinal.group(1))
        if 1 <= position <= len(elected):
            return f"{elected[position - 1]['name']}."
        return None

    found = _find_record(question, records)
    if found:
        index, record = found
        if any(word in q for word in ("before", "predecessor", "preceded")):
            target = index - _relative_distance(question)
            return f"{records[target]['name']}." if target >= 0 else "There is no term that far before this one."
        if any(word in q for word in ("after", "successor", "succeeded")):
            target = index + _relative_distance(question)
            return f"{records[target]['name']}." if target < len(records) else "There is no term that far after this one."
        if "acting" in q:
            return "Yes." if record.get("acting", False) else "No."
        if any(phrase in q for phrase in ("when", "term", "serve", "tenure")):
            return f"{record['start']} to {record['end']}."
        if any(phrase in q for phrase in ("take office", "took office", "start", "begin")):
            return f"{record['start']}."
        if any(phrase in q for phrase in ("leave office", "left office", "end")):
            return f"{record['end']}."

    date = _date_in_question(question)
    if date:
        if any(phrase in q for phrase in ("became", "took office", "started")):
            for record in records:
                if datetime.strptime(record["start"], "%d %B %Y") == date:
                    return f"{record['name']}."
        if "president" in q or "served" in q:
            for record in reversed(records):
                start = datetime.strptime(record["start"], "%d %B %Y")
                end = datetime.max if record["end"] == "Incumbent" else datetime.strptime(record["end"], "%d %B %Y")
                if start <= date < end:
                    return f"{record['name']}."

    return None

