from __future__ import annotations

import ast
import re
from typing import Any

from datasets import Dataset, load_dataset

from .prompts import build_prompt

NUMBER_RE = re.compile(r"\d+")


def parse_numbers(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    if isinstance(value, str):
        stripped = value.strip()
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, (list, tuple)):
                return [int(x) for x in parsed]
        except Exception:
            pass
        nums = [int(x) for x in NUMBER_RE.findall(stripped)]
        if len(nums) >= 4:
            return nums[:4]
    raise ValueError(f"cannot parse numbers from {value!r}")


def _first_present(example: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in example and example[name] is not None:
            return example[name]
    raise KeyError(f"none of these fields exists: {names}; got {list(example)}")


def normalize_24game_example(example: dict[str, Any]) -> dict[str, Any]:
    raw_numbers = _first_present(
        example,
        ["numbers", "nums", "cards", "input", "question", "problem", "Puzzles", "puzzles"],
    )
    numbers = parse_numbers(raw_numbers)
    solvable = example.get("solvable", example.get("is_solvable", True))
    return {
        "numbers": numbers,
        "prompt": build_prompt(numbers),
        "solvable": bool(solvable),
    }


def load_nlile_24game(split: str = "train", solvable_only: bool = True, limit: int | None = None) -> Dataset:
    dataset = load_dataset("nlile/24-game", split=split)
    dataset = dataset.map(normalize_24game_example, remove_columns=dataset.column_names)
    if solvable_only:
        dataset = dataset.filter(lambda row: row["solvable"])
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def load_game_of_24(split: str = "train", mode: str = "all", limit: int | None = None) -> Dataset:
    dataset = load_dataset("test-time-compute/game-of-24", split=split)
    dataset = dataset.map(normalize_24game_example, remove_columns=dataset.column_names)
    if mode == "hard":
        start, end = 900, min(1000, len(dataset))
        dataset = dataset.select(range(start, end))
    elif mode == "tail":
        start = min(900, len(dataset))
        dataset = dataset.select(range(start, len(dataset)))
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset
