"""Tolerant JSON extraction: small free models wrap JSON in prose or code fences."""
from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def extract_json(text: str) -> dict | None:
    """Return the first JSON object found in `text`, or None."""
    text = (text or "").strip()
    candidates = [text] + _FENCE.findall(text)
    for c in candidates:
        try:
            v = json.loads(c.strip())
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            pass
    # scan for the first balanced {...} that parses, respecting strings
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(text[start : i + 1])
                        if isinstance(v, dict):
                            return v
                    except json.JSONDecodeError:
                        pass
                    break
    return None
