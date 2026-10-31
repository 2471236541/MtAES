#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt 7 mapper (Stage 3): reasoning text -> trait scores.
"""

from __future__ import annotations

import json
import re


class Prompt7Mapper:
    """
    Deterministic rule-based mapper for Prompt 7.
    No ML models are used.
    """

    PARTIAL_TOKENS = ("partial", "partially", "somewhat")

    # Calibration layer for trait distribution alignment.
    ENABLE_TRAIT_CALIBRATION = False
    STYLE_CALIBRATION_MAP = {0: 0, 1: 1, 2: 2, 3: 3}
    CONVENTIONS_CALIBRATION_MAP = {0: 0, 1: 1, 2: 2, 3: 3}

    # Style thresholds and hard-zero rule.
    STYLE_THRESHOLD_1 = 0.20
    STYLE_THRESHOLD_2 = 1.25
    STYLE_THRESHOLD_3 = 2.00
    STYLE_HARD_ZERO_MAX_MEAN = 0.65
    STYLE_HARD_ZERO_MAX_COMPONENT = 0
    STYLE_HARD_ZERO_MIN_LOW_COMPONENTS = 4
    STYLE_HARD_ZERO_MAX_PURPOSE_SCORE = 0

    # Conventions thresholds and hard-zero rule.
    CONVENTIONS_THRESHOLD_3 = 1.60
    CONVENTIONS_THRESHOLD_2 = 9.80
    CONVENTIONS_THRESHOLD_1 = 12.30
    CONVENTIONS_HARD_ZERO_MIN_DENSITY = 10.0
    CONVENTIONS_HARD_ZERO_MIN_CATEGORIES = 4
    CONVENTIONS_HARD_ZERO_SEVERITIES = {"frequent", "many"}

    # Optional main-path structured scoring for ideas/organization (A/B switch).
    # Keep disabled by default to preserve baseline behavior.
    ENABLE_IDEAS_ORG_STRUCTURED_MAIN = False
    IDEAS_STRUCTURED_THRESHOLD_1 = 0.70
    IDEAS_STRUCTURED_THRESHOLD_2 = 1.00
    IDEAS_STRUCTURED_THRESHOLD_3 = 2.10
    ORGANIZATION_STRUCTURED_THRESHOLD_1 = 0.30
    ORGANIZATION_STRUCTURED_THRESHOLD_2 = 0.90
    ORGANIZATION_STRUCTURED_THRESHOLD_3 = 1.80

    # Shadow-mode thresholds for ideas/organization (non-scoring side-channel).
    IDEAS_SHADOW_THRESHOLD_1 = 0.60
    IDEAS_SHADOW_THRESHOLD_2 = 1.40
    IDEAS_SHADOW_THRESHOLD_3 = 2.35
    ORGANIZATION_SHADOW_THRESHOLD_1 = 0.55
    ORGANIZATION_SHADOW_THRESHOLD_2 = 1.35
    ORGANIZATION_SHADOW_THRESHOLD_3 = 2.30
    ORGANIZATION_DIAGNOSTIC_V2_THRESHOLD_1 = 0.80
    ORGANIZATION_DIAGNOSTIC_V2_THRESHOLD_2 = 1.55
    ORGANIZATION_DIAGNOSTIC_V2_THRESHOLD_3 = 2.45

    @staticmethod
    def _extract_json_obj(reasoning: str) -> dict | None:
        if not reasoning:
            return None
        try:
            return json.loads(reasoning)
        except Exception:
            pass
        match = re.search(r"\{.*\}", reasoning, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    @staticmethod
    def _extract_answer_lines(reasoning: str) -> list[str]:
        answers = []
        for raw_line in (reasoning or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r"^\s*\d+\s*[\.\):：-]?\s*", "", line).strip()
            if not line:
                continue
            if "?" in line:
                tail = line.split("?", 1)[1].strip()
                if tail:
                    line = tail
            answers.append(line)
        return answers

    @classmethod
    def _detect_yes_no_partial(cls, text: str) -> float | None:
        lowered = (text or "").strip().lower()
        if any(tok in lowered for tok in cls.PARTIAL_TOKENS):
            return 0.5
        if re.match(r"^yes\b", lowered):
            return 1.0
        if re.match(r"^no\b", lowered):
            return 0.0
        return None

    @staticmethod
    def _to_int(value, default=0):
        try:
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _clamp_0_3(score: int) -> int:
        return max(0, min(3, int(score)))

    @staticmethod
    def _word_count(text: str | None) -> int:
        return len(re.findall(r"[A-Za-z0-9@']+", text or ""))

    @staticmethod
    def _bin_by_thresholds(value: float, t1: float, t2: float, t3: float) -> int:
        if value >= t3:
            return 3
        if value >= t2:
            return 2
        if value >= t1:
            return 1
        return 0

    @staticmethod
    def _map_shadow_band_token(text: str) -> int | None:
        lowered = (text or "").strip().lower()
        if lowered in {"0", "1", "2", "3"}:
            return int(lowered)
        match = re.search(r"\b([0-3])\b", lowered)
        if match:
            return int(match.group(1))
        if re.match(r"^(no|none)\b", lowered):
            return 0
        has_negation = bool(
            re.search(
                r"\b(does not|doesn't|did not|didn't|do not|don't|cannot|can't|lacks|lack|without|unclear)\b",
                lowered,
            )
            or re.search(
                r"\b(not clear|not clearly|no clear|not focused|not focus|not organized|not organised|"
                r"not coherent|not structured|not effective|not logical|not developed|not well developed)\b",
                lowered,
            )
        )
        has_partial = bool(re.search(r"\b(limited|partial|partially|weak|basic|somewhat|mixed|inconsistent)\b", lowered))
        has_positive = bool(
            re.search(
                r"\b(adequate|clear|mostly|sufficient|strong|effective|thorough|well-developed|cohesive)\b",
                lowered,
            )
        )
        if has_negation:
            if has_partial or has_positive or " but " in lowered:
                return 1
            return 0
        if re.search(r"\b(none|minimal|missing|absent)\b", lowered):
            return 0
        if re.search(r"\b(limited|partial|partially|weak|basic)\b", lowered):
            return 1
        if re.search(r"\b(adequate|clear|mostly|sufficient)\b", lowered):
            return 2
        if re.search(r"\b(strong|effective|thorough|well-developed|cohesive)\b", lowered):
            return 3
        return None

    @staticmethod
    def _normalize_structured_key(text: str) -> str:
        lowered = (text or "").strip().lower()
        lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
        return lowered.strip("_")

    @staticmethod
    def _contains_pattern(text: str, patterns: tuple[str, ...]) -> bool:
        lowered = (text or "").strip().lower()
        return any(re.search(pattern, lowered) for pattern in patterns)

    @classmethod
    def _map_yes_no_partial_to_band(cls, text: str) -> int | None:
        signal = cls._detect_yes_no_partial(text)
        if signal == 1.0:
            return 2
        if signal == 0.5:
            return 1
        if signal == 0.0:
            return 0
        return None

    @classmethod
    def _score_ideas_focus_answer(cls, text: str) -> int | None:
        direct = cls._map_yes_no_partial_to_band(text)
        if direct is not None:
            return direct
        if cls._contains_pattern(
            text,
            (
                r"\bnot clearly focused\b",
                r"\bnot focused\b",
                r"\bunclear focus\b",
                r"\blacks focus\b",
                r"\bno clear focus\b",
                r"\bdoes not mention .*topic\b",
                r"\bnot related to the topic\b",
                r"\boff[- ]topic\b",
            ),
        ):
            return 0
        if cls._contains_pattern(
            text,
            (
                r"\bmostly focused\b",
                r"\bsomewhat focused\b",
                r"\bpartially focused\b",
                r"\bgenerally focused\b",
                r"\bloosely focused\b",
            ),
        ):
            return 1
        if cls._contains_pattern(
            text,
            (
                r"\bclearly focused\b",
                r"\bwell focused\b",
                r"\bmaintains focus\b",
                r"\bfocused on the assigned topic\b",
                r"\bfocused on the topic\b",
                r"\bclearly on topic\b",
                r"\bdirectly related to the topic\b",
            ),
        ):
            return 2
        return None

    @classmethod
    def _score_ideas_central_answer(cls, text: str) -> int | None:
        direct = cls._map_yes_no_partial_to_band(text)
        if direct is not None:
            return direct
        if cls._contains_pattern(
            text,
            (
                r"\bdoes not present a clear central\b",
                r"\bdoes not present a central\b",
                r"\bno clear central\b",
                r"\bno controlling idea\b",
                r"\bno clear main idea\b",
                r"\blacks a central idea\b",
                r"\bjumps from one topic to another\b",
                r"\bwithout a consistent thread\b",
                r"\bwithout consistent thread\b",
                r"\bno consistent thread\b",
                r"\bno clear storyline\b",
            ),
        ):
            return 0
        if cls._contains_pattern(
            text,
            (
                r"\bsome central idea\b",
                r"\bbasic central idea\b",
                r"\bpartially clear\b",
                r"\bmostly clear\b",
                r"\bloose thread\b",
                r"\bweak central idea\b",
            ),
        ):
            return 1
        if cls._contains_pattern(
            text,
            (
                r"\bpresents a central storyline\b",
                r"\bcontrolling idea\b",
                r"\bcentral storyline\b",
                r"\bcentral idea\b",
                r"\bmain idea\b",
                r"\bclear storyline\b",
                r"\bconsistent thread\b",
                r"\bcoherent storyline\b",
            ),
        ):
            return 2
        return None

    @classmethod
    def _score_ideas_support_answer(cls, text: str) -> int | None:
        direct = cls._map_yes_no_partial_to_band(text)
        if direct is not None:
            return direct
        if cls._contains_pattern(
            text,
            (
                r"\bno specific details\b",
                r"\blacks support\b",
                r"\blacks evidence\b",
                r"\bnot supported\b",
                r"\bwithout supporting details\b",
                r"\bno real supporting details\b",
            ),
        ):
            return 0
        if cls._contains_pattern(
            text,
            (
                r"\bsome specific details\b",
                r"\bfew details\b",
                r"\blimited details\b",
                r"\bcould use more details\b",
                r"\bpartially supported\b",
                r"\bsome details supporting\b",
            ),
        ):
            return 1
        if cls._contains_pattern(
            text,
            (
                r"\bspecific details\b",
                r"\bsupporting details\b",
                r"\bprovides details\b",
                r"\bexamples?\b",
                r"\bevidence\b",
                r"\bsupport the main idea\b",
            ),
        ):
            return 2
        return None

    @classmethod
    def _score_ideas_development_answer(cls, text: str) -> int | None:
        direct = cls._map_yes_no_partial_to_band(text)
        if direct is not None:
            return direct
        if cls._contains_pattern(
            text,
            (
                r"\bbrief\b",
                r"\black[s]? depth\b",
                r"\bunderdeveloped\b",
                r"\bnot developed\b",
                r"\bnot well developed\b",
                r"\bvery limited development\b",
            ),
        ):
            return 0
        if cls._contains_pattern(
            text,
            (
                r"\bpartially developed\b",
                r"\bnot fully developed\b",
                r"\bcould be further developed\b",
                r"\bcould be more developed\b",
                r"\bsome aspects .* further developed\b",
                r"\bmore development\b",
            ),
        ):
            return 1
        if cls._contains_pattern(
            text,
            (
                r"\bfully developed\b",
                r"\bwell developed\b",
                r"\bdeveloped throughout\b",
                r"\bthoroughly developed\b",
                r"\brichly developed\b",
            ),
        ):
            return 2
        return None

    @classmethod
    def _score_ideas_offtopic_answer(cls, text: str) -> int | None:
        lowered = (text or "").strip().lower()
        signal = cls._detect_yes_no_partial(text)
        if signal == 0.0:
            return 0
        if signal == 0.5:
            return 1
        if signal == 1.0:
            if cls._contains_pattern(lowered, (r"\bsome\b", r"\bminor\b", r"\ba little\b", r"\ba few\b")):
                return 1
            return 2
        if cls._contains_pattern(
            lowered,
            (
                r"\bno irrelevant\b",
                r"\bno off[- ]topic\b",
                r"\ball of the events .* directly related\b",
                r"\bdirectly related to the topic\b",
            ),
        ):
            return 0
        if cls._contains_pattern(
            lowered,
            (
                r"\bsome irrelevant\b",
                r"\bsome off[- ]topic\b",
                r"\bminor irrelevant\b",
                r"\bminor off[- ]topic\b",
            ),
        ):
            return 1
        if cls._contains_pattern(
            lowered,
            (
                r"\birrelevant\b",
                r"\boff[- ]topic\b",
                r"\bunrelated content\b",
                r"\bunrelated thoughts\b",
            ),
        ):
            return 2
        return None

    @classmethod
    def score_ideas_field_aware_from_reasoning(
        cls,
        reasoning: str,
        thresholds: tuple[float, float, float] = (0.30, 0.50, 0.70),
    ) -> tuple[int, dict]:
        answers = cls._extract_answer_lines(reasoning)
        parsed_fields: dict[str, int] = {}
        parser_specs = (
            ("focus", cls._score_ideas_focus_answer),
            ("central_idea", cls._score_ideas_central_answer),
            ("support", cls._score_ideas_support_answer),
            ("development", cls._score_ideas_development_answer),
            ("off_topic", cls._score_ideas_offtopic_answer),
        )

        for idx, (field_name, parser_fn) in enumerate(parser_specs):
            if idx >= len(answers):
                break
            mapped = parser_fn(answers[idx])
            if mapped is not None:
                parsed_fields[field_name] = mapped

        coverage = len(parsed_fields)
        if coverage < 4:
            generic_score = cls._score_generic_yes_no(answers)
            return generic_score, {
                "mode": "field_aware_fallback_generic_yesno",
                "coverage": coverage,
                "parsed_fields": parsed_fields,
            }

        focus = parsed_fields.get("focus", 0) / 2.0
        central = parsed_fields.get("central_idea", 0) / 2.0
        support = parsed_fields.get("support", 0) / 2.0
        development = parsed_fields.get("development", 0) / 2.0
        # off_topic is a penalty signal: 0 means fully relevant, 2 means clearly off-topic.
        # Convert it into a positive relevance-like contribution before weighting.
        off_topic_raw = parsed_fields.get("off_topic", 0)
        off_topic = max(0.0, min(1.0, (2 - off_topic_raw) / 2.0))

        weights = {
            "focus": 1.0,
            "central_idea": 1.0,
            "support": 2.0,
            "development": 2.0,
            "off_topic": 1.0,
        }
        raw_num = (
            (focus * weights["focus"])
            + (central * weights["central_idea"])
            + (support * weights["support"])
            + (development * weights["development"])
            + (off_topic * weights["off_topic"])
        )
        raw = raw_num / sum(weights.values())

        score = cls._bin_by_thresholds(raw, thresholds[0], thresholds[1], thresholds[2])

        if parsed_fields.get("focus", 0) == 0 and parsed_fields.get("central_idea", 0) == 0:
            score = min(score, 1)
        if parsed_fields.get("support", 0) == 0 and parsed_fields.get("development", 0) == 0:
            score = min(score, 1)
        if parsed_fields.get("development", 0) == 0:
            score = min(score, 2)

        return score, {
            "mode": "field_aware_weighted",
            "coverage": coverage,
            "parsed_fields": parsed_fields,
            "raw": round(raw, 4),
            "thresholds": list(thresholds),
        }

    @classmethod
    def adjust_ideas_score_with_reasoning_caps(cls, score: int, reasoning: str) -> tuple[int, dict]:
        ideas_score = cls._clamp_0_3(score)
        answers = cls._extract_answer_lines(reasoning)
        meta = {
            "mode": "reasoning_caps",
            "changed": False,
            "reason": None,
        }
        if ideas_score != 3 or len(answers) < 5:
            return ideas_score, meta

        q1 = cls._score_ideas_focus_answer(answers[0])
        q2 = cls._score_ideas_central_answer(answers[1])
        q4 = cls._score_ideas_development_answer(answers[3])
        q5 = cls._score_ideas_offtopic_answer(answers[4])

        if q4 is not None and q4 < 2:
            return 2, {
                "mode": "reasoning_caps",
                "changed": True,
                "reason": "development_not_strong_enough_for_3",
            }
        if q1 == 0:
            return 2, {
                "mode": "reasoning_caps",
                "changed": True,
                "reason": "focus_failure_blocks_3",
            }
        if q2 == 0:
            return 2, {
                "mode": "reasoning_caps",
                "changed": True,
                "reason": "central_idea_failure_blocks_3",
            }
        if q5 is not None and q5 > 0:
            return 2, {
                "mode": "reasoning_caps",
                "changed": True,
                "reason": "off_topic_content_blocks_3",
            }
        return ideas_score, meta

    @classmethod
    def _extract_ideas_org_fields_from_json(cls, trait_key: str, payload: dict) -> dict[str, int]:
        if trait_key == "ideas":
            field_key_aliases = {
                "focus": {"focus", "topic_focus", "focus_control"},
                "central_idea": {
                    "central_idea",
                    "main_idea",
                    "claim",
                    "controlling_idea",
                    "thesis",
                    "storyline",
                },
                "evidence": {"evidence", "support", "supporting_evidence", "support_details"},
                "elaboration": {"elaboration", "development", "detail", "details"},
                "relevance": {
                    "relevance",
                    "on_topic",
                    "topic_relevance",
                    "consistency",
                    "off_topic_control",
                },
            }
        elif trait_key == "organization":
            field_key_aliases = {
                "lead_closure": {
                    "lead_closure",
                    "lead",
                    "intro",
                    "introduction",
                    "conclusion",
                    "closure",
                    "lead_and_closure",
                },
                "progression": {"progression", "sequence", "logical_progression", "logical_order"},
                "transitions": {"transitions", "transition", "linking", "connectors"},
                "coherence": {"coherence", "paragraphing", "paragraph", "unity"},
            }
        else:
            return {}

        def resolve_field_name(key_norm: str) -> str | None:
            suffixes = ("_score", "_band", "_level", "_label", "_value")
            for candidate, aliases in field_key_aliases.items():
                if key_norm in aliases:
                    return candidate
                for suffix in suffixes:
                    if key_norm.endswith(suffix):
                        base = key_norm[: -len(suffix)]
                        if base in aliases:
                            return candidate
            return None

        parsed_fields: dict[str, int] = {}
        for raw_key, raw_value in payload.items():
            key_norm = cls._normalize_structured_key(str(raw_key))
            field_name = resolve_field_name(key_norm)
            if field_name is None or field_name in parsed_fields:
                continue

            value = raw_value
            if isinstance(value, dict):
                for nested_key in ("score", "band", "level", "label", "value"):
                    if nested_key in value:
                        value = value[nested_key]
                        break
            if isinstance(value, (int, float)):
                maybe = int(round(value))
                if 0 <= maybe <= 3:
                    parsed_fields[field_name] = maybe
                    continue
                value = str(value)
            mapped = cls._map_shadow_band_token(str(value))
            if mapped is not None:
                parsed_fields[field_name] = mapped
        return parsed_fields

    @classmethod
    def _score_ideas_org_banded_reasoning(
        cls,
        trait: str,
        reasoning: str,
        thresholds: tuple[float, float, float],
        apply_main_adjustments: bool = False,
    ) -> tuple[int | None, dict]:
        trait_key = (trait or "").strip().lower()
        if trait_key == "ideas":
            field_aliases = {
                "focus": (("focus", "focused", "topic focus"), 1.00),
                "central_idea": (
                    ("central idea", "central_idea", "main idea", "controlling idea", "storyline", "thesis"),
                    1.00,
                ),
                "evidence": (("evidence", "support", "supporting detail", "details"), 2.00),
                "elaboration": (("elaboration", "development", "depth"), 2.00),
                "relevance": (("relevance", "on topic", "on_topic", "off topic", "off_topic", "consistency"), 1.00),
            }
            bonus = 0.05
            min_required_fields = 4
        elif trait_key == "organization":
            field_aliases = {
                "lead_closure": (("lead", "lead_closure", "introduction", "conclusion", "closure"), 1.00),
                "progression": (("progression", "sequence", "logical"), 1.20),
                "transitions": (("transition", "transitions", "linking"), 1.00),
                "coherence": (("coherence", "paragraph"), 1.15),
            }
            bonus = 0.04
            min_required_fields = 3
        else:
            return None, {"parsed_fields": {}, "coverage": 0}

        parsed_fields: dict[str, int] = {}
        weighted_sum = 0.0
        weight_total = 0.0
        payload = cls._extract_json_obj(reasoning)
        if isinstance(payload, dict):
            parsed_fields.update(cls._extract_ideas_org_fields_from_json(trait_key, payload))
            for field, mapped in parsed_fields.items():
                weight = field_aliases[field][1]
                weighted_sum += mapped * weight
                weight_total += weight

        answers = cls._extract_answer_lines(reasoning)

        for answer in answers:
            lowered = answer.strip().lower()
            if not lowered:
                continue
            value_part = lowered.split(":", 1)[1].strip() if ":" in lowered else lowered
            mapped = cls._map_shadow_band_token(value_part)
            if mapped is None:
                mapped = cls._map_shadow_band_token(lowered)
            if mapped is None:
                continue
            lowered_norm = cls._normalize_structured_key(lowered)
            for field, (aliases, weight) in field_aliases.items():
                if field in parsed_fields:
                    continue
                alias_norms = [cls._normalize_structured_key(alias) for alias in aliases]
                if any(alias in lowered for alias in aliases) or any(
                    alias_norm in lowered_norm for alias_norm in alias_norms if alias_norm
                ):
                    parsed_fields[field] = mapped
                    weighted_sum += mapped * weight
                    weight_total += weight
                    break

        if len(parsed_fields) < min_required_fields and len(answers) >= 4:
            field_order = (
                ["focus", "central_idea", "evidence", "elaboration", "relevance"]
                if trait_key == "ideas"
                else ["lead_closure", "progression", "transitions", "coherence"]
            )
            for idx, field in enumerate(field_order):
                if idx >= len(answers):
                    break
                if field in parsed_fields:
                    continue
                answer = answers[idx].strip().lower()
                value_part = answer.split(":", 1)[1].strip() if ":" in answer else answer
                mapped = cls._map_shadow_band_token(value_part)
                if mapped is None:
                    continue
                parsed_fields[field] = mapped
                weight = field_aliases[field][1]
                weighted_sum += mapped * weight
                weight_total += weight

        if len(parsed_fields) >= min_required_fields and weight_total > 0:
            mean_val = weighted_sum / weight_total
            if len(parsed_fields) == len(field_aliases) and min(parsed_fields.values()) >= 2:
                mean_val += bonus
            if apply_main_adjustments and len(parsed_fields) >= min_required_fields:
                if trait_key == "ideas":
                    if parsed_fields.get("evidence", 2) <= 1:
                        mean_val -= 0.15
                    if parsed_fields.get("elaboration", 2) <= 1:
                        mean_val -= 0.20
                    if parsed_fields.get("central_idea", 2) <= 1:
                        mean_val -= 0.10
                    if parsed_fields.get("relevance", 2) <= 1:
                        mean_val -= 0.10
                elif trait_key == "organization":
                    if parsed_fields.get("transitions", 2) <= 1:
                        mean_val -= 0.20
                    if parsed_fields.get("coherence", 2) <= 1:
                        mean_val -= 0.20
                    if parsed_fields.get("lead_closure", 2) == 0:
                        mean_val -= 0.08
            score = cls._bin_by_thresholds(mean_val, thresholds[0], thresholds[1], thresholds[2])
            if apply_main_adjustments and trait_key == "ideas":
                if parsed_fields.get("focus", 2) == 0 or parsed_fields.get("central_idea", 2) == 0:
                    score = min(score, 1)
                if parsed_fields.get("relevance", 2) == 0:
                    score = min(score, 1)
                elif parsed_fields.get("relevance", 2) == 1:
                    score = min(score, 2)
                if parsed_fields.get("evidence", 2) <= 1 and parsed_fields.get("elaboration", 2) <= 1:
                    score = min(score, 1)
                elif parsed_fields.get("elaboration", 2) <= 1:
                    score = min(score, 2)
            return score, {
                "parsed_fields": parsed_fields,
                "coverage": len(parsed_fields),
                "mean": round(mean_val, 4),
            }

        token_scores = []
        for answer in answers:
            mapped = cls._map_shadow_band_token(answer)
            if mapped is not None:
                token_scores.append(mapped)
        if len(token_scores) >= min_required_fields:
            mean_val = sum(token_scores) / len(token_scores)
            score = cls._bin_by_thresholds(mean_val, thresholds[0], thresholds[1], thresholds[2])
            return score, {
                "parsed_fields": parsed_fields,
                "coverage": len(parsed_fields),
                "mean": round(mean_val, 4),
            }

        return None, {"parsed_fields": parsed_fields, "coverage": len(parsed_fields)}

    @classmethod
    def score_organization_diagnostic_v2_from_reasoning(cls, reasoning: str) -> tuple[int, dict]:
        payload = cls._extract_json_obj(reasoning)
        if not isinstance(payload, dict):
            fallback = cls.score_trait_from_reasoning("organization", reasoning)
            return int(fallback), {"mode": "organization_diagnostic_v2_fallback"}

        band_to_score = {"none": 0, "limited": 1, "adequate": 2, "strong": 3}
        pattern_floor = {
            "scatter": 0,
            "list_like": 0,
            "partly_ordered": 1,
            "clear_arc": 2,
            "cohesive_arc": 2,
        }
        pattern_cap = {
            "scatter": 1,
            "list_like": 1,
            "partly_ordered": 2,
            "clear_arc": 3,
            "cohesive_arc": 3,
        }

        def read_component(key: str) -> int:
            raw_value = payload.get(key, None)
            if isinstance(raw_value, dict):
                score = cls._to_int(raw_value.get("score"), -1)
                if 0 <= score <= 3:
                    return int(score)
                band = str(raw_value.get("band", "")).strip().lower()
                if band in band_to_score:
                    return int(band_to_score[band])
            score = cls._to_int(raw_value, -1)
            if 0 <= score <= 3:
                return int(score)
            band = str(raw_value).strip().lower()
            if band in band_to_score:
                return int(band_to_score[band])
            return 0

        story_frame = read_component("story_frame")
        paragraph_plan = read_component("paragraph_plan")
        progression_control = read_component("progression_control")
        linking = read_component("linking")
        unity = read_component("unity")

        dominant_pattern = str(payload.get("dominant_pattern", "")).strip().lower()
        primary_limiter = str(payload.get("primary_limiter", "")).strip().lower()
        secondary_limiter = str(payload.get("secondary_limiter", "")).strip().lower()
        limiter_values = {primary_limiter, secondary_limiter}

        overall_score = cls._to_int(payload.get("overall_score"), -1)

        weighted_mean = (
            (story_frame * 0.90)
            + (paragraph_plan * 1.10)
            + (progression_control * 1.35)
            + (linking * 0.85)
            + (unity * 1.25)
        ) / 5.45
        fallback_score = cls._bin_by_thresholds(
            weighted_mean,
            cls.ORGANIZATION_DIAGNOSTIC_V2_THRESHOLD_1,
            cls.ORGANIZATION_DIAGNOSTIC_V2_THRESHOLD_2,
            cls.ORGANIZATION_DIAGNOSTIC_V2_THRESHOLD_3,
        )

        score = int(overall_score if 0 <= overall_score <= 3 else fallback_score)

        if dominant_pattern in pattern_cap:
            score = min(score, int(pattern_cap[dominant_pattern]))

        if progression_control == 0 or unity == 0:
            score = min(score, 1)
        if paragraph_plan == 0 and story_frame <= 1:
            score = min(score, 1)
        if linking == 0 and progression_control <= 1:
            score = min(score, 1)
        if dominant_pattern in {"scatter", "list_like"}:
            score = min(score, 1)
        if limiter_values & {"multiple", "progression_control", "unity"} and min(progression_control, unity) <= 1:
            score = min(score, 1)

        if (
            score == 0
            and progression_control >= 1
            and unity >= 1
            and dominant_pattern in {"partly_ordered", "clear_arc", "cohesive_arc"}
        ):
            score = 1

        if dominant_pattern in pattern_floor and score < int(pattern_floor[dominant_pattern]):
            score = int(pattern_floor[dominant_pattern])

        if (
            score <= 1
            and progression_control >= 2
            and unity >= 2
            and paragraph_plan >= 1
            and dominant_pattern in {"partly_ordered", "clear_arc", "cohesive_arc"}
            and "progression_control" not in limiter_values
            and "unity" not in limiter_values
        ):
            score = 2

        if (
            score >= 3
            and (
                min(story_frame, paragraph_plan, progression_control, linking, unity) < 2
                or dominant_pattern not in {"clear_arc", "cohesive_arc"}
                or limiter_values & {"story_frame", "paragraph_plan", "progression_control", "linking", "unity", "multiple"}
            )
        ):
            score = 2

        if (
            score == 2
            and story_frame >= 2
            and paragraph_plan >= 2
            and progression_control >= 2
            and linking >= 2
            and unity >= 2
            and dominant_pattern == "cohesive_arc"
            and "none" in limiter_values
        ):
            score = 3

        return cls._clamp_0_3(score), {
            "mode": "organization_diagnostic_v2",
            "overall_score": int(overall_score) if 0 <= overall_score <= 3 else None,
            "weighted_mean": round(weighted_mean, 4),
            "fallback_score": int(fallback_score),
            "dominant_pattern": dominant_pattern,
            "parsed_fields": {
                "story_frame": int(story_frame),
                "paragraph_plan": int(paragraph_plan),
                "progression_control": int(progression_control),
                "linking": int(linking),
                "unity": int(unity),
            },
            "limiters": sorted(x for x in limiter_values if x),
        }

    @classmethod
    def score_organization_readerpath_v1_from_reasoning(cls, reasoning: str) -> tuple[int, dict]:
        payload = cls._extract_json_obj(reasoning)
        if not isinstance(payload, dict):
            fallback = cls.score_trait_from_reasoning("organization", reasoning)
            return int(fallback), {"mode": "organization_readerpath_v1_fallback"}

        token_map = {
            "reader_path": {"lost": 0, "repair_needed": 1, "mostly_clear": 2, "easy": 3},
            "begin_end_control": {"missing": 0, "weak": 1, "clear": 2, "strong": 3},
            "event_order": {"broken": 0, "uneven": 1, "mostly_clear": 2, "smooth": 3},
            "grouping": {"missing": 0, "weak": 1, "clear": 2, "strong": 3},
            "jumps": {"many": 0, "some": 1, "minor": 2, "none": 3},
            "connections": {"missing": 0, "basic": 1, "useful": 2, "strong": 3},
        }

        def read_token_score(key: str) -> int:
            mapping = token_map[key]
            raw = str(payload.get(key, "")).strip().lower()
            return int(mapping.get(raw, 0))

        reader_path = read_token_score("reader_path")
        begin_end_control = read_token_score("begin_end_control")
        event_order = read_token_score("event_order")
        grouping = read_token_score("grouping")
        jumps = read_token_score("jumps")
        connections = read_token_score("connections")
        forced_score = cls._to_int(payload.get("forced_score"), -1)
        main_blocker = str(payload.get("main_blocker", "")).strip().lower()

        weighted_mean = (
            (reader_path * 1.35)
            + (begin_end_control * 0.80)
            + (event_order * 1.25)
            + (grouping * 1.10)
            + (jumps * 1.15)
            + (connections * 0.75)
        ) / 6.40
        fallback_score = cls._bin_by_thresholds(weighted_mean, 0.80, 1.55, 2.45)
        score = int(forced_score if 0 <= forced_score <= 3 else fallback_score)

        if reader_path == 0 or event_order == 0:
            score = 0
        if reader_path == 1:
            score = min(score, 1)
        if jumps == 0 and grouping <= 1:
            score = min(score, 1)

        if score == 0 and reader_path >= 1 and event_order >= 1:
            score = 1

        if score >= 2 and (reader_path < 2 or event_order < 2 or jumps < 1):
            score = 1

        if (
            score <= 1
            and reader_path >= 2
            and event_order >= 2
            and grouping >= 1
            and jumps >= 1
            and main_blocker not in {"reader_path", "event_order"}
        ):
            score = 2

        if (
            score >= 3
            and (
                reader_path < 3
                or event_order < 3
                or grouping < 2
                or jumps < 2
                or begin_end_control < 2
                or connections < 2
                or main_blocker != "none"
            )
        ):
            score = 2

        if (
            score == 2
            and reader_path == 3
            and event_order >= 2
            and grouping >= 2
            and jumps >= 2
            and begin_end_control >= 2
            and connections >= 2
            and main_blocker == "none"
        ):
            score = 3

        return cls._clamp_0_3(score), {
            "mode": "organization_readerpath_v1",
            "forced_score": int(forced_score) if 0 <= forced_score <= 3 else None,
            "weighted_mean": round(weighted_mean, 4),
            "fallback_score": int(fallback_score),
            "main_blocker": main_blocker,
            "parsed_fields": {
                "reader_path": int(reader_path),
                "begin_end_control": int(begin_end_control),
                "event_order": int(event_order),
                "grouping": int(grouping),
                "jumps": int(jumps),
                "connections": int(connections),
            },
        }

    @classmethod
    def _score_ideas_org_structured_from_reasoning(cls, trait: str, reasoning: str) -> tuple[int | None, dict]:
        trait_key = (trait or "").strip().lower()
        if trait_key == "ideas":
            thresholds = (
                cls.IDEAS_STRUCTURED_THRESHOLD_1,
                cls.IDEAS_STRUCTURED_THRESHOLD_2,
                cls.IDEAS_STRUCTURED_THRESHOLD_3,
            )
        elif trait_key == "organization":
            thresholds = (
                cls.ORGANIZATION_STRUCTURED_THRESHOLD_1,
                cls.ORGANIZATION_STRUCTURED_THRESHOLD_2,
                cls.ORGANIZATION_STRUCTURED_THRESHOLD_3,
            )
        else:
            return None, {"parsed_fields": {}, "coverage": 0}
        return cls._score_ideas_org_banded_reasoning(
            trait=trait_key,
            reasoning=reasoning,
            thresholds=thresholds,
            apply_main_adjustments=True,
        )

    @classmethod
    def score_ideas_org_shadow_from_reasoning(cls, trait: str, reasoning: str) -> tuple[int | None, dict]:
        trait_key = (trait or "").strip().lower()
        if trait_key == "ideas":
            thresholds = (
                cls.IDEAS_SHADOW_THRESHOLD_1,
                cls.IDEAS_SHADOW_THRESHOLD_2,
                cls.IDEAS_SHADOW_THRESHOLD_3,
            )
        elif trait_key == "organization":
            thresholds = (
                cls.ORGANIZATION_SHADOW_THRESHOLD_1,
                cls.ORGANIZATION_SHADOW_THRESHOLD_2,
                cls.ORGANIZATION_SHADOW_THRESHOLD_3,
            )
        else:
            return None, {"parsed_fields": {}, "coverage": 0}
        return cls._score_ideas_org_banded_reasoning(
            trait=trait_key,
            reasoning=reasoning,
            thresholds=thresholds,
            apply_main_adjustments=False,
        )

    @classmethod
    def _map_style_token(cls, token: str) -> int | None:
        t = (token or "").strip().lower()
        if t in {"none", "ineffective", "not", "off_target", "hinders", "unclear"}:
            return 0
        if t in {"limited", "partially", "low", "mixed", "inconsistent", "weak"}:
            return 1
        if t in {"adequate", "mostly", "mostly_consistent", "medium", "clear", "supportive"}:
            return 2
        if t in {
            "strong",
            "compelling",
            "clearly",
            "consistent",
            "high",
            "effective",
            "distinctive",
            "purpose_aligned",
            "audience_aware",
        }:
            return 3
        return None

    @classmethod
    def _map_style_text(cls, text: str) -> int | None:
        lowered = (text or "").strip().lower().replace("-", "_")
        lowered = lowered.replace("mostly consistent", "mostly_consistent")
        lowered = lowered.replace("supports purpose and audience", "purpose_aligned")
        lowered = lowered.replace("supports purpose", "purpose_aligned")
        lowered = lowered.replace("supports the audience", "audience_aware")

        if re.search(
            r"\b(strong|compelling|effective|high|consistent|clearly|distinctive|purpose_aligned|audience_aware)\b",
            lowered,
        ):
            return 3
        if re.search(r"\b(adequate|mostly|mostly_consistent|medium|clear|supportive)\b", lowered):
            return 2
        if re.search(r"\b(limited|partially|low|mixed|inconsistent|weak)\b", lowered):
            return 1
        if re.search(r"\b(none|ineffective|off_target|hinder|hinders|unclear)\b", lowered):
            return 0
        return None

    @classmethod
    def _style_bin(
        cls,
        style_mean: float,
        core_scores: list[int] | None = None,
        purpose_score: int | None = None,
    ) -> int:
        if core_scores:
            low_count = sum(1 for x in core_scores if x <= cls.STYLE_HARD_ZERO_MAX_COMPONENT)
            purpose_is_low = (
                purpose_score is not None and purpose_score <= cls.STYLE_HARD_ZERO_MAX_PURPOSE_SCORE
            )
            if (
                low_count >= cls.STYLE_HARD_ZERO_MIN_LOW_COMPONENTS
                and (purpose_is_low or style_mean < cls.STYLE_HARD_ZERO_MAX_MEAN)
            ):
                return 0

        if style_mean >= cls.STYLE_THRESHOLD_3:
            return 3
        if style_mean >= cls.STYLE_THRESHOLD_2:
            return 2
        if style_mean >= cls.STYLE_THRESHOLD_1:
            return 1
        return 0

    @classmethod
    def _score_style_from_json(cls, payload: dict) -> int | None:
        # Core Prompt 7 style dimensions: purpose/audience support, word choice,
        # sentence variety, and voice. Legacy fields are mapped as fallback.
        numeric_aliases = {
            "purpose_audience_score": (("purpose_audience_score", 1.35), ("tone_score", 1.10)),
            "word_choice_score": (("word_choice_score", 1.00),),
            "sentence_variety_score": (("sentence_variety_score", 1.00),),
            "voice_score": (("voice_score", 1.00), ("engagement_score", 0.90)),
        }
        weighted_sum = 0.0
        weight_total = 0.0
        core_numeric_scores = []
        purpose_score: int | None = None
        for aliases in numeric_aliases.values():
            mapped_score = None
            mapped_weight = 0.0
            mapped_field = None
            for key, weight in aliases:
                score = cls._to_int(payload.get(key), -1)
                if 0 <= score <= 3:
                    mapped_score = float(score)
                    mapped_weight = weight
                    mapped_field = key
                    break
            if mapped_score is not None:
                weighted_sum += mapped_score * mapped_weight
                weight_total += mapped_weight
                core_numeric_scores.append(int(mapped_score))
                if mapped_field in {"purpose_audience_score", "tone_score"}:
                    purpose_score = int(mapped_score)
        if weight_total > 0:
            style_mean = weighted_sum / weight_total
            fig_score = cls._to_int(payload.get("figurative_score"), -1)
            if fig_score == 1:
                style_mean += 0.04
            elif fig_score == 2:
                style_mean += 0.10
            elif fig_score == 3:
                style_mean += 0.16
            return cls._style_bin(
                style_mean=style_mean,
                core_scores=core_numeric_scores,
                purpose_score=purpose_score,
            )

        text_aliases = {
            "purpose_audience_support": ("purpose_audience_support", "purpose_audience", "tone_consistency"),
            "word_choice": ("word_choice",),
            "sentence_variety": ("sentence_variety",),
            "voice_presence": ("voice_presence", "engagement"),
        }
        text_weights = {
            "purpose_audience_support": 1.35,
            "word_choice": 1.00,
            "sentence_variety": 1.00,
            "voice_presence": 1.00,
        }
        ordered_core_fields = (
            "purpose_audience_support",
            "word_choice",
            "sentence_variety",
            "voice_presence",
        )
        weighted_sum = 0.0
        weight_total = 0.0
        field_scores: dict[str, int] = {}
        for canonical, aliases in text_aliases.items():
            mapped = None
            for key in aliases:
                raw_value = str(payload.get(key, ""))
                if not raw_value.strip():
                    continue
                mapped = cls._map_style_token(raw_value)
                if mapped is None:
                    mapped = cls._map_style_text(raw_value)
                if mapped is not None:
                    break
            if mapped is not None:
                weight = text_weights[canonical]
                weighted_sum += mapped * weight
                weight_total += weight
                field_scores[canonical] = mapped
        if weight_total == 0:
            return None

        # Figurative language is supportive, not mandatory in Prompt 7 style.
        fig = str(payload.get("figurative_language", "")).strip().lower()
        fig_mapped = cls._map_style_token(fig)
        if fig_mapped is None:
            fig_mapped = cls._map_style_text(fig)
        fig_bonus = 0.0
        if fig_mapped == 1:
            fig_bonus = 0.04
        elif fig_mapped == 2:
            fig_bonus = 0.10
        elif fig_mapped == 3:
            fig_bonus = 0.16

        style_mean = (weighted_sum / weight_total) + fig_bonus
        core_scores = [field_scores[k] for k in ordered_core_fields if k in field_scores]
        purpose_score = field_scores.get("purpose_audience_support")
        return cls._style_bin(
            style_mean=style_mean,
            core_scores=core_scores,
            purpose_score=purpose_score,
        )

    @classmethod
    def _score_style_from_answer_lines(cls, answers: list[str]) -> int | None:
        if not answers:
            return None

        core_aliases = {
            "purpose_audience_support": (
                "purpose_audience_support",
                "purpose audience",
                "purpose/audience",
                "tone_consistency",
                "tone consistency",
                "tone",
            ),
            "word_choice": ("word_choice", "word choice"),
            "sentence_variety": ("sentence_variety", "sentence variety"),
            "voice_presence": ("voice_presence", "voice", "engagement"),
        }
        core_weights = {
            "purpose_audience_support": 1.35,
            "word_choice": 1.00,
            "sentence_variety": 1.00,
            "voice_presence": 1.00,
        }
        fig_aliases = ("figurative_language", "figurative language", "figurative")

        field_scores: dict[str, int] = {}
        fig_score: int | None = None

        for answer in answers:
            lowered = answer.strip().lower()
            if not lowered:
                continue
            value_part = lowered.split(":", 1)[1].strip() if ":" in lowered else lowered

            for field, aliases in core_aliases.items():
                if field in field_scores:
                    continue
                if any(alias in lowered for alias in aliases):
                    mapped = cls._map_style_text(value_part)
                    if mapped is not None:
                        field_scores[field] = mapped
                    break

            if fig_score is None and any(alias in lowered for alias in fig_aliases):
                mapped = cls._map_style_text(value_part)
                if mapped is not None:
                    fig_score = mapped

        if field_scores:
            weighted_sum = 0.0
            weight_total = 0.0
            for field, score in field_scores.items():
                weight = core_weights.get(field, 1.0)
                weighted_sum += score * weight
                weight_total += weight
            style_mean = (weighted_sum / weight_total) if weight_total > 0 else 0.0
            if fig_score == 1:
                style_mean += 0.04
            elif fig_score == 2:
                style_mean += 0.10
            elif fig_score == 3:
                style_mean += 0.16
            ordered_core_fields = (
                "purpose_audience_support",
                "word_choice",
                "sentence_variety",
                "voice_presence",
            )
            core_scores = [field_scores[k] for k in ordered_core_fields if k in field_scores]
            purpose_score = field_scores.get("purpose_audience_support")
            return cls._style_bin(
                style_mean=style_mean,
                core_scores=core_scores,
                purpose_score=purpose_score,
            )

        token_scores = []
        for answer in answers:
            mapped = cls._map_style_text(answer)
            if mapped is not None:
                token_scores.append(mapped)
        if len(token_scores) >= 2:
            return cls._style_bin(style_mean=(sum(token_scores) / len(token_scores)))

        return None

    @classmethod
    def _score_conventions_from_json(cls, payload: dict, essay_word_count: int | None = None) -> int | None:
        error_keys = (
            "grammar_errors",
            "spelling_errors",
            "punctuation_errors",
            "capitalization_errors",
            "sentence_boundary_errors",
        )
        error_counts = {
            key: max(0, cls._to_int(payload.get(key, 0), 0))
            for key in error_keys
        }
        total_errors = sum(error_counts.values())
        error_categories = sum(1 for v in error_counts.values() if v > 0)
        severity = str(payload.get("error_severity", "")).strip().lower()

        severity_weight = {
            "none": 0,
            "minor": 0,
            "few": 0.5,
            "moderate": 1.5,
            "some": 1.5,
            "frequent": 3.0,
            "many": 3.0,
        }.get(severity, 0)

        evidence = payload.get("evidence", [])
        if isinstance(evidence, list):
            evidence_text = " ".join(str(x) for x in evidence).lower()
        else:
            evidence_text = str(evidence).lower()

        inconsistency_penalty = 0
        # Consistency check: all-zero counts should not co-exist with non-minor severity or error evidence.
        if total_errors == 0:
            if severity in {"moderate", "some", "frequent", "many"}:
                inconsistency_penalty += 2
            elif severity == "few":
                inconsistency_penalty += 1
            if "error" in evidence_text:
                inconsistency_penalty += 1

        # Another consistency check: high counts with "minor/none" severity are suspicious.
        if total_errors >= 6 and severity in {"none", "minor"}:
            inconsistency_penalty += 1

        # Use error density so long essays are not over-penalized by raw counts.
        if essay_word_count is not None and essay_word_count > 0:
            errors_per_100_words = (total_errors * 100.0) / max(essay_word_count, 40)
        else:
            errors_per_100_words = float(total_errors)
        density_component = min(errors_per_100_words, 12.0) * 0.75
        spread_component = error_categories * 0.45
        error_signal = density_component + spread_component + severity_weight + inconsistency_penalty

        is_hard_zero = (
            errors_per_100_words >= cls.CONVENTIONS_HARD_ZERO_MIN_DENSITY
            and error_categories >= cls.CONVENTIONS_HARD_ZERO_MIN_CATEGORIES
            and severity in cls.CONVENTIONS_HARD_ZERO_SEVERITIES
        )
        if is_hard_zero:
            return 0

        if error_signal <= cls.CONVENTIONS_THRESHOLD_3:
            return 3
        if error_signal <= cls.CONVENTIONS_THRESHOLD_2:
            return 2
        if error_signal <= cls.CONVENTIONS_THRESHOLD_1:
            return 1
        return 0

    @classmethod
    def _apply_trait_calibration(cls, trait_key: str, score: int) -> int:
        score = cls._clamp_0_3(score)
        if not cls.ENABLE_TRAIT_CALIBRATION:
            return score
        if trait_key == "style":
            return cls._clamp_0_3(cls.STYLE_CALIBRATION_MAP.get(score, score))
        if trait_key == "conventions":
            return cls._clamp_0_3(cls.CONVENTIONS_CALIBRATION_MAP.get(score, score))
        return score

    @classmethod
    def _score_conventions_from_yes_no(cls, answers: list[str]) -> int:
        if not answers:
            return 0

        deductions = 0.0
        for idx, answer in enumerate(answers[:5]):
            signal = cls._detect_yes_no_partial(answer)
            lowered = answer.lower()

            if signal is None:
                if idx <= 1:
                    if re.search(r"\b(no|none|not|without)\b.*\berror", lowered):
                        signal = 0.0
                    elif "error" in lowered:
                        signal = 1.0
                else:
                    if re.search(r"\b(not|incorrect|wrong|missing|unclear)\b", lowered):
                        signal = 0.0
                    elif re.search(r"\b(correct|proper|followed|appropriate|well-formed)\b", lowered):
                        signal = 1.0

            if idx <= 1:
                if signal == 1.0:
                    deductions += 1.0
                elif signal == 0.5:
                    deductions += 0.5
            else:
                if signal == 0.0:
                    deductions += 1.0
                elif signal == 0.5:
                    deductions += 0.5

        if deductions <= 0.5:
            return 3
        if deductions <= 1.5:
            return 2
        if deductions <= 2.5:
            return 1
        return 0

    @classmethod
    def _score_generic_yes_no(cls, answers: list[str]) -> int:
        answer_scores = []
        for answer in answers:
            signal = cls._detect_yes_no_partial(answer)
            if signal is not None:
                answer_scores.append(signal)

        if not answer_scores:
            return 0

        trait_norm = sum(answer_scores) / len(answer_scores)
        if trait_norm >= 0.8:
            return 3
        if trait_norm >= 0.6:
            return 2
        if trait_norm >= 0.4:
            return 1
        return 0

    @classmethod
    def score_trait_from_reasoning(cls, trait: str, reasoning: str, essay_text: str | None = None) -> int:
        trait_key = (trait or "").strip().lower()
        payload = cls._extract_json_obj(reasoning)

        if trait_key in {"ideas", "organization"}:
            if trait_key == "organization" and isinstance(payload, dict):
                readerpath_keys = {"reader_path", "begin_end_control", "event_order", "grouping", "jumps", "connections"}
                if any(key in payload for key in readerpath_keys):
                    score, _ = cls.score_organization_readerpath_v1_from_reasoning(reasoning)
                    return score
                org_diag_keys = {"story_frame", "paragraph_plan", "progression_control", "linking", "unity"}
                if any(key in payload for key in org_diag_keys):
                    score, _ = cls.score_organization_diagnostic_v2_from_reasoning(reasoning)
                    return score
            if cls.ENABLE_IDEAS_ORG_STRUCTURED_MAIN:
                score, _ = cls._score_ideas_org_structured_from_reasoning(trait_key, reasoning)
                if score is not None:
                    return score
            answers = cls._extract_answer_lines(reasoning)
            return cls._score_generic_yes_no(answers)

        if trait_key == "style":
            if isinstance(payload, dict):
                score = cls._score_style_from_json(payload)
                if score is not None:
                    return cls._apply_trait_calibration(trait_key, score)
            answers = cls._extract_answer_lines(reasoning)
            score_fallback = cls._score_style_from_answer_lines(answers)
            if score_fallback is not None:
                return cls._apply_trait_calibration(trait_key, score_fallback)
            return cls._apply_trait_calibration(trait_key, cls._score_generic_yes_no(answers))

        if trait_key == "conventions":
            essay_word_count = cls._word_count(essay_text)
            if isinstance(payload, dict):
                score = cls._score_conventions_from_json(payload, essay_word_count=essay_word_count)
                if score is not None:
                    return cls._apply_trait_calibration(trait_key, score)
            answers = cls._extract_answer_lines(reasoning)
            return cls._apply_trait_calibration(trait_key, cls._score_conventions_from_yes_no(answers))

        answers = cls._extract_answer_lines(reasoning)
        return cls._score_generic_yes_no(answers)

    @staticmethod
    def compute_overall_score(ideas: int, organization: int, style: int, conventions: int) -> int:
        # Calibrated mapping from trait composite (0-15) to Prompt 7 domain1 scale (2-24).
        single_rater = (ideas * 2) + organization + style + conventions
        calibrated = round((1.32 * single_rater) + 3.04)
        return max(2, min(24, int(calibrated)))
