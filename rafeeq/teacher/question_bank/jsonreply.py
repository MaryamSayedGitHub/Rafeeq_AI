"""Read the JSON object an LLM put in its reply."""
from __future__ import annotations

import json
import re
from typing import Optional


def first_json_object(raw: str) -> Optional[dict]:
    """Take the first {...} block, so code fences or chatter around it are fine."""
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None