#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sidecar 文本输入预设。

用途：
1) 训练时按 trait 组装输入文本
2) 推理时按模型保存的配置复原同样的输入文本
"""

from __future__ import annotations

import re
from typing import Any


DEFAULT_TEXT_PRESET = "essay_only"
DEFAULT_SENTBLOCK_SIZE = 3
DEFAULT_HEAD_WORDS = 140
DEFAULT_TAIL_WORDS = 60
DEFAULT_MID_WORDS = 80
SUPPORTED_TEXT_PRESETS = {
    "essay_only",
    "essay_plus_trait_reasoning",
    "trait_reasoning_plus_essay",
    "essay_sentblock",
    "essay_head_tail",
    "essay_head_mid_tail",
    "essay_head_tail_ideas_compact",
    "essay_plus_org_checklist",
    "essay_plus_org_top3",
    "essay_plus_org_struct_v1",
    "essay_plus_org_struct_v2_short",
    "essay_plus_org_struct_v3_compact",
    "essay_plus_org_flow_v1",
    "essay_plus_conv_struct_v1",
}

ORG_REASONING_FIELDS: list[tuple[int, str]] = [
    (1, "intro"),
    (2, "progression"),
    (3, "body_structure"),
    (4, "conclusion"),
    (5, "transitions"),
    (6, "sequence"),
]

IDEAS_REASONING_FIELDS: list[tuple[int, str]] = [
    (1, "focused_on_topic"),
    (2, "central_idea"),
    (3, "supporting_details"),
    (4, "development"),
    (5, "off_topic_content"),
]

ORG_TRANSITION_WORDS = {
    "first",
    "firstly",
    "second",
    "secondly",
    "third",
    "thirdly",
    "finally",
    "lastly",
    "also",
    "however",
    "therefore",
    "because",
    "although",
    "moreover",
    "furthermore",
    "instead",
    "then",
    "next",
    "after",
    "before",
    "when",
    "while",
    "but",
    "so",
    "in",
    "conclusion",
}


def parse_trait_text_presets(spec: str, trait_names: list[str]) -> dict[str, str]:
    presets = {trait: DEFAULT_TEXT_PRESET for trait in trait_names}
    raw = str(spec or "").strip()
    if not raw:
        return presets

    for item in raw.split(","):
        part = item.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"Invalid trait text preset item: {part}. Expected format trait=preset."
            )
        trait, preset = [x.strip() for x in part.split("=", 1)]
        if trait not in presets:
            raise ValueError(f"Unknown trait in text presets: {trait}")
        if preset not in SUPPORTED_TEXT_PRESETS:
            raise ValueError(
                f"Unsupported text preset: {preset}. Supported: {sorted(SUPPORTED_TEXT_PRESETS)}"
            )
        presets[trait] = preset
    return presets


def _split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    if not cleaned:
        return []
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    if sentences:
        return sentences
    return [cleaned]


def _build_sentblock_text(essay_text: str, block_size: int = DEFAULT_SENTBLOCK_SIZE) -> str:
    sentences = _split_sentences(essay_text)
    if not sentences:
        return essay_text

    blocks: list[str] = []
    safe_block_size = max(1, int(block_size))
    for idx in range(0, len(sentences), safe_block_size):
        block = " ".join(sentences[idx : idx + safe_block_size]).strip()
        if block:
            blocks.append(f"[B{len(blocks) + 1}]\n{block}")
    return "\n\n".join(blocks) if blocks else essay_text


def _build_head_tail_text(
    essay_text: str,
    head_words: int = DEFAULT_HEAD_WORDS,
    tail_words: int = DEFAULT_TAIL_WORDS,
) -> str:
    words = str(essay_text or "").split()
    safe_head = max(1, int(head_words))
    safe_tail = max(1, int(tail_words))
    if len(words) <= safe_head + safe_tail:
        return essay_text

    head = " ".join(words[:safe_head]).strip()
    tail = " ".join(words[-safe_tail:]).strip()
    if not head or not tail:
        return essay_text
    return f"[HEAD]\n{head}\n\n[TAIL]\n{tail}"


def _build_head_mid_tail_text(
    essay_text: str,
    head_words: int = 120,
    mid_words: int = DEFAULT_MID_WORDS,
    tail_words: int = DEFAULT_MID_WORDS,
) -> str:
    words = str(essay_text or "").split()
    safe_head = max(1, int(head_words))
    safe_mid = max(1, int(mid_words))
    safe_tail = max(1, int(tail_words))
    if len(words) <= safe_head + safe_mid + safe_tail:
        return essay_text

    midpoint = len(words) // 2
    mid_start = max(safe_head, midpoint - safe_mid // 2)
    mid_end = min(len(words) - safe_tail, mid_start + safe_mid)
    mid_start = max(safe_head, mid_end - safe_mid)

    head = " ".join(words[:safe_head]).strip()
    middle = " ".join(words[mid_start:mid_end]).strip()
    tail = " ".join(words[-safe_tail:]).strip()
    if not head or not middle or not tail:
        return essay_text
    return f"[HEAD]\n{head}\n\n[MIDDLE]\n{middle}\n\n[TAIL]\n{tail}"


def _extract_numbered_reasoning_items(reasoning_text: str) -> dict[int, str]:
    cleaned = re.sub(r"\s+", " ", str(reasoning_text or "").strip())
    if not cleaned:
        return {}

    raw_text = str(reasoning_text or "").strip()
    pattern = re.compile(r"(?ms)^\s*(\d+)\.\s*(.*?)(?=^\s*\d+\.\s|\Z)")
    items: dict[int, str] = {}
    for match in pattern.finditer(raw_text):
        idx = int(match.group(1))
        item_text = re.sub(r"\s+", " ", match.group(2).strip())
        if item_text:
            items[idx] = item_text
    return items


def _org_status_from_reasoning(item_text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(item_text or "").strip()).lower()
    if not cleaned:
        return "na"
    if cleaned.startswith("yes"):
        return "yes"
    if cleaned.startswith("no"):
        return "no"
    if cleaned.startswith("partial") or cleaned.startswith("partly") or cleaned.startswith("mixed"):
        return "mixed"
    if "somewhat" in cleaned or "unclear" in cleaned or "not clear" in cleaned:
        return "mixed"
    return "unknown"


def _ideas_status_from_reasoning(item_text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(item_text or "").strip()).lower()
    if not cleaned:
        return "na"
    if cleaned.startswith("yes"):
        return "yes"
    if cleaned.startswith("no"):
        return "no"
    if cleaned.startswith("partial") or cleaned.startswith("partly") or cleaned.startswith("mixed"):
        return "mixed"
    if "partially" in cleaned or "somewhat" in cleaned or "not fully" in cleaned:
        return "mixed"
    if "off-topic" in cleaned and "no" in cleaned[:24]:
        return "no"
    return "unknown"


def _build_ideas_compact_text(essay_text: str, reasoning_text: str) -> str:
    items = _extract_numbered_reasoning_items(reasoning_text)
    head_tail = _build_head_tail_text(essay_text)
    lines = ["[ESSAY_HEAD_TAIL]", head_tail, "", "[IDEAS_CHECKLIST]"]
    for idx, name in IDEAS_REASONING_FIELDS:
        lines.append(f"{name}={_ideas_status_from_reasoning(items.get(idx, ''))}")
    return "\n".join(lines)


def _build_org_compact_text(essay_text: str, reasoning_text: str, *, top_n: int | None = None) -> str | None:
    items = _extract_numbered_reasoning_items(reasoning_text)
    if not items:
        return None

    selected_fields = ORG_REASONING_FIELDS[:top_n] if top_n else ORG_REASONING_FIELDS
    header = "[ORG_TOP3]" if top_n else "[ORG_CHECKLIST]"
    lines = [f"{name}={_org_status_from_reasoning(items.get(idx, ''))}" for idx, name in selected_fields]
    return f"[ESSAY]\n{essay_text}\n\n{header}\n" + "\n".join(lines)


def _split_paragraphs(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    paragraphs = [
        re.sub(r"\s+", " ", part.strip())
        for part in re.split(r"(?:\r?\n\s*){1,}", raw)
        if part.strip()
    ]
    return paragraphs if paragraphs else [re.sub(r"\s+", " ", raw)]


def _clip_words(text: str, max_words: int) -> str:
    words = str(text or "").split()
    safe_max = max(1, int(max_words))
    if len(words) <= safe_max:
        return " ".join(words)
    return " ".join(words[:safe_max]) + " ..."


def _transition_count(text: str) -> int:
    tokens = re.findall(r"[a-zA-Z']+", str(text or "").lower())
    if not tokens:
        return 0
    count = 0
    for idx, tok in enumerate(tokens):
        if tok in ORG_TRANSITION_WORDS:
            count += 1
        if idx + 1 < len(tokens) and f"{tok} {tokens[idx + 1]}" == "in conclusion":
            count += 1
    return count


def _build_org_struct_v1_text(essay_text: str) -> str:
    paragraphs = _split_paragraphs(essay_text)
    sentences = _split_sentences(essay_text)
    words = str(essay_text or "").split()
    word_count = len(words)
    sentence_count = len(sentences)
    paragraph_count = len(paragraphs)
    avg_sentence_words = round(word_count / sentence_count, 1) if sentence_count else 0.0
    transition_count = _transition_count(essay_text)

    first_sentence = _clip_words(sentences[0], 28) if sentences else ""
    last_sentence = _clip_words(sentences[-1], 28) if sentences else ""
    first_paragraph = _clip_words(paragraphs[0], 36) if paragraphs else ""
    last_paragraph = _clip_words(paragraphs[-1], 36) if paragraphs else ""

    lines = [
        "[ORG_STRUCTURE]",
        f"word_count={word_count}",
        f"sentence_count={sentence_count}",
        f"paragraph_count={paragraph_count}",
        f"avg_sentence_words={avg_sentence_words}",
        f"transition_count={transition_count}",
        f"has_multiple_paragraphs={int(paragraph_count >= 2)}",
        f"has_intro_signal={int(bool(first_sentence))}",
        f"has_conclusion_signal={int(any(x in last_sentence.lower() for x in ('conclusion', 'finally', 'lastly', 'learned', 'overall')))}",
        "",
        "[ORG_OUTLINE]",
        f"first_sentence: {first_sentence}",
        f"last_sentence: {last_sentence}",
        f"first_paragraph: {first_paragraph}",
        f"last_paragraph: {last_paragraph}",
        "",
        "[ESSAY]",
        essay_text,
    ]
    return "\n".join(lines)


def _paragraph_openings(paragraphs: list[str], max_paragraphs: int = 4, words_per_paragraph: int = 14) -> list[str]:
    openings: list[str] = []
    for idx, paragraph in enumerate(paragraphs[: max(1, int(max_paragraphs))], start=1):
        clipped = _clip_words(paragraph, words_per_paragraph)
        if clipped:
            openings.append(f"p{idx}: {clipped}")
    return openings


def _build_org_struct_v2_short_text(essay_text: str) -> str:
    paragraphs = _split_paragraphs(essay_text)
    sentences = _split_sentences(essay_text)
    words = str(essay_text or "").split()
    word_count = len(words)
    sentence_count = len(sentences)
    paragraph_count = len(paragraphs)
    avg_sentence_words = round(word_count / sentence_count, 1) if sentence_count else 0.0
    transition_count = _transition_count(essay_text)

    first_sentence = _clip_words(sentences[0], 18) if sentences else ""
    last_sentence = _clip_words(sentences[-1], 18) if sentences else ""
    head_tail = _build_head_tail_text(essay_text, head_words=72, tail_words=48)
    outline = _paragraph_openings(paragraphs, max_paragraphs=4, words_per_paragraph=14)

    lines = [
        "[ORG_STATS]",
        f"words={word_count}",
        f"sentences={sentence_count}",
        f"paragraphs={paragraph_count}",
        f"avg_sentence_words={avg_sentence_words}",
        f"transitions={transition_count}",
        f"multi_paragraph={int(paragraph_count >= 2)}",
        f"has_conclusion_signal={int(any(x in last_sentence.lower() for x in ('conclusion', 'finally', 'lastly', 'learned', 'overall')))}",
        "",
        "[ORG_BOUNDARIES]",
        f"first_sentence: {first_sentence}",
        f"last_sentence: {last_sentence}",
    ]
    if outline:
        lines.extend(["", "[PARAGRAPH_OPENINGS]", *outline])
    lines.extend(["", "[ESSAY_HEAD_TAIL]", head_tail])
    return "\n".join(lines)


def _build_org_struct_v3_compact_text(essay_text: str, reasoning_text: str) -> str:
    paragraphs = _split_paragraphs(essay_text)
    sentences = _split_sentences(essay_text)
    words = str(essay_text or "").split()
    word_count = len(words)
    sentence_count = len(sentences)
    paragraph_count = len(paragraphs)
    avg_sentence_words = round(word_count / sentence_count, 1) if sentence_count else 0.0
    transition_count = _transition_count(essay_text)

    first_sentence = _clip_words(sentences[0], 16) if sentences else ""
    last_sentence = _clip_words(sentences[-1], 16) if sentences else ""
    head_tail = _build_head_tail_text(essay_text, head_words=60, tail_words=40)
    outline = _paragraph_openings(paragraphs, max_paragraphs=4, words_per_paragraph=12)
    reasoning_items = _extract_numbered_reasoning_items(reasoning_text)

    lines = [
        "[ORG_STATS]",
        f"words={word_count}",
        f"sentences={sentence_count}",
        f"paragraphs={paragraph_count}",
        f"avg_sentence_words={avg_sentence_words}",
        f"transitions={transition_count}",
        f"multi_paragraph={int(paragraph_count >= 2)}",
        f"has_conclusion_signal={int(any(x in last_sentence.lower() for x in ('conclusion', 'finally', 'lastly', 'learned', 'overall')))}",
        "",
        "[ORG_CHECKLIST_COMPACT]",
    ]
    for idx, name in ORG_REASONING_FIELDS:
        lines.append(f"{name}={_org_status_from_reasoning(reasoning_items.get(idx, ''))}")

    lines.extend(
        [
            "",
            "[ORG_BOUNDARIES]",
            f"first_sentence: {first_sentence}",
            f"last_sentence: {last_sentence}",
        ]
    )
    if outline:
        lines.extend(["", "[PARAGRAPH_OPENINGS]", *outline])
    lines.extend(["", "[ESSAY_HEAD_TAIL]", head_tail])
    return "\n".join(lines)


def _sentence_outline(sentences: list[str], *, head: int = 4, tail: int = 3, words_per_sentence: int = 12) -> list[str]:
    if not sentences:
        return []
    selected: list[int] = []
    selected.extend(range(min(head, len(sentences))))
    tail_start = max(0, len(sentences) - max(0, tail))
    selected.extend(range(tail_start, len(sentences)))

    seen: set[int] = set()
    lines: list[str] = []
    for idx in selected:
        if idx in seen:
            continue
        seen.add(idx)
        clipped = _clip_words(sentences[idx], words_per_sentence)
        if clipped:
            lines.append(f"s{idx + 1}/{len(sentences)}: {clipped}")
    return lines


def _build_org_flow_v1_text(essay_text: str) -> str:
    paragraphs = _split_paragraphs(essay_text)
    sentences = _split_sentences(essay_text)
    words = str(essay_text or "").split()
    word_count = len(words)
    sentence_count = len(sentences)
    paragraph_count = len(paragraphs)
    avg_sentence_words = round(word_count / sentence_count, 1) if sentence_count else 0.0
    transition_count = _transition_count(essay_text)
    first_sentence = _clip_words(sentences[0], 18) if sentences else ""
    last_sentence = _clip_words(sentences[-1], 22) if sentences else ""
    flow = _sentence_outline(sentences, head=4, tail=3, words_per_sentence=12)
    head_tail = _build_head_tail_text(essay_text, head_words=48, tail_words=72)

    lines = [
        "[ORG_FLOW_STATS]",
        f"words={word_count}",
        f"sentences={sentence_count}",
        f"paragraphs={paragraph_count}",
        f"avg_sentence_words={avg_sentence_words}",
        f"transitions={transition_count}",
        f"transition_per_100_words={round((transition_count / max(1, word_count)) * 100, 1)}",
        f"multi_paragraph={int(paragraph_count >= 2)}",
        f"has_conclusion_signal={int(any(x in last_sentence.lower() for x in ('conclusion', 'finally', 'lastly', 'learned', 'overall')))}",
        "",
        "[ORG_BOUNDARIES]",
        f"first_sentence: {first_sentence}",
        f"last_sentence: {last_sentence}",
    ]
    if flow:
        lines.extend(["", "[SENTENCE_FLOW]", *flow])
    lines.extend(["", "[ESSAY_HEAD_TAIL_TAIL_HEAVY]", head_tail])
    return "\n".join(lines)


def _build_conv_struct_v1_text(essay_text: str, reasoning_text: str) -> str:
    payload: dict[str, Any] = {}
    try:
        parsed = __import__("json").loads(str(reasoning_text or "{}"))
        if isinstance(parsed, dict):
            payload = parsed
    except Exception:
        payload = {}

    def int_field(name: str) -> int:
        try:
            return int(payload.get(name, 0))
        except Exception:
            return 0

    evidence_raw = payload.get("evidence", [])
    evidence: list[str] = []
    if isinstance(evidence_raw, list):
        evidence = [_clip_words(str(item), 16) for item in evidence_raw[:4] if str(item).strip()]

    total_errors = (
        int_field("grammar_errors")
        + int_field("spelling_errors")
        + int_field("punctuation_errors")
        + int_field("capitalization_errors")
        + int_field("sentence_boundary_errors")
    )
    words = len(str(essay_text or "").split())
    errors_per_100 = round((total_errors / max(1, words)) * 100, 1)
    head_tail = _build_head_tail_text(essay_text, head_words=72, tail_words=48)

    lines = [
        "[CONVENTIONS_STATS]",
        f"words={words}",
        f"grammar_errors={int_field('grammar_errors')}",
        f"spelling_errors={int_field('spelling_errors')}",
        f"punctuation_errors={int_field('punctuation_errors')}",
        f"capitalization_errors={int_field('capitalization_errors')}",
        f"sentence_boundary_errors={int_field('sentence_boundary_errors')}",
        f"total_errors={total_errors}",
        f"errors_per_100_words={errors_per_100}",
        f"severity={str(payload.get('error_severity', 'unknown')).strip().lower() or 'unknown'}",
    ]
    if evidence:
        lines.extend(["", "[ERROR_EVIDENCE]", *[f"- {item}" for item in evidence]])
    lines.extend(["", "[ESSAY_HEAD_TAIL]", head_tail])
    return "\n".join(lines)


def build_trait_text(row: dict[str, Any], trait_name: str, preset: str) -> str | None:
    essay_text = str(row.get("essay_text", "")).strip()
    if not essay_text:
        return None

    if preset == "essay_only":
        return essay_text

    if preset == "essay_plus_trait_reasoning":
        reasoning_key = f"{trait_name}_reasoning"
        reasoning_text = str(row.get(reasoning_key, "")).strip()
        if not reasoning_text:
            return None
        return (
            f"[ESSAY]\n{essay_text}\n\n"
            f"[{trait_name.upper()}_REASONING]\n{reasoning_text}"
        )

    if preset == "trait_reasoning_plus_essay":
        reasoning_key = f"{trait_name}_reasoning"
        reasoning_text = str(row.get(reasoning_key, "")).strip()
        if not reasoning_text:
            return None
        return (
            f"[{trait_name.upper()}_REASONING]\n{reasoning_text}\n\n"
            f"[ESSAY]\n{essay_text}"
        )

    if preset == "essay_sentblock":
        return _build_sentblock_text(essay_text)

    if preset == "essay_head_tail":
        return _build_head_tail_text(essay_text)

    if preset == "essay_head_mid_tail":
        return _build_head_mid_tail_text(essay_text)

    if preset == "essay_head_tail_ideas_compact":
        if trait_name != "ideas":
            raise ValueError("essay_head_tail_ideas_compact is only supported for ideas")
        reasoning_text = str(row.get("ideas_reasoning", "")).strip()
        return _build_ideas_compact_text(essay_text, reasoning_text)

    if preset == "essay_plus_org_checklist":
        if trait_name != "organization":
            raise ValueError("essay_plus_org_checklist is only supported for organization")
        reasoning_text = str(row.get("organization_reasoning", "")).strip()
        return _build_org_compact_text(essay_text, reasoning_text, top_n=None)

    if preset == "essay_plus_org_top3":
        if trait_name != "organization":
            raise ValueError("essay_plus_org_top3 is only supported for organization")
        reasoning_text = str(row.get("organization_reasoning", "")).strip()
        return _build_org_compact_text(essay_text, reasoning_text, top_n=3)

    if preset == "essay_plus_org_struct_v1":
        if trait_name != "organization":
            raise ValueError("essay_plus_org_struct_v1 is only supported for organization")
        return _build_org_struct_v1_text(essay_text)

    if preset == "essay_plus_org_struct_v2_short":
        if trait_name != "organization":
            raise ValueError("essay_plus_org_struct_v2_short is only supported for organization")
        return _build_org_struct_v2_short_text(essay_text)

    if preset == "essay_plus_org_struct_v3_compact":
        if trait_name != "organization":
            raise ValueError("essay_plus_org_struct_v3_compact is only supported for organization")
        reasoning_text = str(row.get("organization_reasoning", "")).strip()
        return _build_org_struct_v3_compact_text(essay_text, reasoning_text)

    if preset == "essay_plus_org_flow_v1":
        if trait_name != "organization":
            raise ValueError("essay_plus_org_flow_v1 is only supported for organization")
        return _build_org_flow_v1_text(essay_text)

    if preset == "essay_plus_conv_struct_v1":
        if trait_name != "conventions":
            raise ValueError("essay_plus_conv_struct_v1 is only supported for conventions")
        reasoning_text = str(row.get("conventions_reasoning", "")).strip()
        return _build_conv_struct_v1_text(essay_text, reasoning_text)

    raise ValueError(f"Unsupported text preset: {preset}")


def trait_input_config_payload(trait_name: str, text_preset: str, max_length: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "trait": str(trait_name),
        "text_preset": str(text_preset),
    }
    if max_length is not None:
        payload["max_length"] = int(max_length)
    if text_preset == "essay_sentblock":
        payload["sentblock_size"] = int(DEFAULT_SENTBLOCK_SIZE)
    if text_preset == "essay_head_tail":
        payload["head_words"] = int(DEFAULT_HEAD_WORDS)
        payload["tail_words"] = int(DEFAULT_TAIL_WORDS)
    if text_preset == "essay_head_mid_tail":
        payload["head_words"] = 120
        payload["mid_words"] = int(DEFAULT_MID_WORDS)
        payload["tail_words"] = int(DEFAULT_MID_WORDS)
    if text_preset == "essay_head_tail_ideas_compact":
        payload["head_words"] = int(DEFAULT_HEAD_WORDS)
        payload["tail_words"] = int(DEFAULT_TAIL_WORDS)
        payload["uses_compact_ideas_checklist"] = True
    if text_preset == "essay_plus_org_struct_v1":
        payload["org_struct_version"] = "v1"
    if text_preset == "essay_plus_org_struct_v2_short":
        payload["org_struct_version"] = "v2_short"
        payload["head_words"] = 72
        payload["tail_words"] = 48
        payload["paragraph_openings"] = 4
    if text_preset == "essay_plus_org_struct_v3_compact":
        payload["org_struct_version"] = "v3_compact"
        payload["head_words"] = 60
        payload["tail_words"] = 40
        payload["paragraph_openings"] = 4
        payload["uses_compact_org_checklist"] = True
    if text_preset == "essay_plus_org_flow_v1":
        payload["org_struct_version"] = "flow_v1"
        payload["head_words"] = 48
        payload["tail_words"] = 72
        payload["sentence_outline_head"] = 4
        payload["sentence_outline_tail"] = 3
    if text_preset == "essay_plus_conv_struct_v1":
        payload["conv_struct_version"] = "v1"
        payload["head_words"] = 72
        payload["tail_words"] = 48
        payload["max_evidence_items"] = 4
    return payload
