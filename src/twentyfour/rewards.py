from __future__ import annotations

import math

from .verifier import extract_answer, has_r1_format, verify_expression


def _normalize_numbers(value) -> list[int]:
    if isinstance(value, str):
        return [int(x) for x in value.replace(",", " ").split()]
    return [int(x) for x in value]


def answer_format_reward(completions, **kwargs) -> list[float]:
    return [0.2 if has_r1_format(text) else 0.0 for text in completions]


def valid_expression_reward(completions, numbers=None, nums=None, **kwargs) -> list[float]:
    batch_numbers = numbers if numbers is not None else nums
    rewards = []
    for text, item_numbers in zip(completions, batch_numbers):
        result = verify_expression(extract_answer(text), _normalize_numbers(item_numbers))
        rewards.append(0.3 if result.value is not None else 0.0)
    return rewards


def correctness_reward(completions, numbers=None, nums=None, **kwargs) -> list[float]:
    batch_numbers = numbers if numbers is not None else nums
    rewards = []
    for text, item_numbers in zip(completions, batch_numbers):
        result = verify_expression(extract_answer(text), _normalize_numbers(item_numbers))
        rewards.append(1.0 if result.ok else 0.0)
    return rewards


def proximity_reward(completions, numbers=None, nums=None, **kwargs) -> list[float]:
    """Dense verifiable reward that increases as a valid expression approaches 24."""
    batch_numbers = numbers if numbers is not None else nums
    rewards = []
    for text, item_numbers in zip(completions, batch_numbers):
        result = verify_expression(extract_answer(text), _normalize_numbers(item_numbers))
        if result.value is None:
            rewards.append(0.0)
            continue
        distance = abs(float(result.value) - 24.0)
        rewards.append(0.5 * math.exp(-distance / 6.0))
    return rewards
