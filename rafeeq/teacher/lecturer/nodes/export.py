"""Turns an approved draft into files. Plain functions, no LLM and no MCP: easy to test."""

import json
import re
from pathlib import Path

CODE_EXTENSIONS = {"python": "py", "javascript": "js", "java": "java", "c++": "cpp"}
COMMENT_PREFIX = {"python": "#"}  # every other supported language uses //


def slugify(text: str, fallback: str = "session") -> str:
    """'Python decorators!' -> 'python-decorators'. Safe to use as a file name."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or fallback


def write_code_file(folder: Path, examples: list[dict], programming_language: str) -> Path:
    """All examples in one file, each one under a comment header with its explanation."""
    lang = programming_language.strip().lower()
    comment = COMMENT_PREFIX.get(lang, "//")
    parts = []
    for i, ex in enumerate(examples, 1):
        status = {True: "verified by running it", False: "NOT verified: it failed when run"}.get(
            ex.get("verified"), "not run"
        )
        header = [f"Example {i}: {ex.get('title', '')}", f"Status: {status}"]
        header += str(ex.get("explanation", "")).splitlines()
        parts.append("\n".join(f"{comment} {line}".rstrip() for line in header))
        parts.append(str(ex.get("code", "")).rstrip() + "\n")
    path = folder / f"examples.{CODE_EXTENSIONS.get(lang, 'txt')}"
    path.write_text("\n\n".join(parts), encoding="utf-8")
    return path


def write_quiz_files(folder: Path, quiz: list[dict], topic: str) -> list[Path]:
    """quiz.json for other systems, quiz.md for the lecturer to read or print."""
    json_path = folder / "quiz.json"
    json_path.write_text(
        json.dumps({"topic": topic, "questions": quiz}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [f"# Quiz: {topic}", ""]
    for i, q in enumerate(quiz, 1):
        lines.append(f"{i}. {q['question']}")
        lines += [f"   {'ABCD'[j]}. {option}" for j, option in enumerate(q["options"])]
        lines.append("")
    lines += ["## Answer key", ""]
    for i, q in enumerate(quiz, 1):
        lines.append(f"{i}. {'ABCD'[q['answer_index']]}: {q.get('explanation', '')}")
    md_path = folder / "quiz.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [json_path, md_path]
