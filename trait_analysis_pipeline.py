#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Prompt 7 pipeline:
Stage 1: load fixed trait questions from trait_questions_prompt7/*.json
Stage 2: call LLM to produce reasoning answers
Stage 3: convert reasoning into trait scores
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt7_mapper import Prompt7Mapper


class TraitAnalysisPipeline:
    def __init__(
        self,
        model_path: str,
        questions_dir: str = "trait_questions_prompt7",
        max_new_tokens: int = 320,
        device: str | None = None,
        enable_json_retry: bool = True,
    ):
        self.questions_dir = Path(questions_dir)
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_json_retry = bool(enable_json_retry)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()
        self.mapper = Prompt7Mapper()

    def load_prompt7_questions(self) -> dict[str, list[str]]:
        trait_files = {
            "ideas": "ideas.json",
            "organization": "organization.json",
            "style": "style.json",
            "conventions": "conventions.json",
        }

        if not self.questions_dir.exists():
            raise FileNotFoundError(f"Questions directory not found: {self.questions_dir}")

        questions = {}
        for trait, filename in trait_files.items():
            file_path = self.questions_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Missing question file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            questions[trait] = data["questions"]
        return questions

    def _build_reasoning_prompt(self, essay: str, trait: str, questions: list[str]) -> str:
        if trait == "style":
            return (
                "You are evaluating STYLE only for an ASAP Prompt 7 essay.\n"
                "Do NOT restate questions.\n"
                "Do NOT give final trait score.\n"
                "Each style component must be judged independently.\n"
                "Style means command of language supporting purpose and audience.\n"
                "Return STRICT JSON only:\n"
                "{\n"
                '  "purpose_audience_support": "ineffective|limited|adequate|strong",\n'
                '  "voice_presence": "none|limited|adequate|distinctive",\n'
                '  "sentence_variety": "none|limited|adequate|strong",\n'
                '  "word_choice": "none|limited|adequate|strong",\n'
                '  "figurative_language": "none|limited|effective",\n'
                '  "purpose_audience_score": <0-3 integer>,\n'
                '  "voice_score": <0-3 integer>,\n'
                '  "sentence_variety_score": <0-3 integer>,\n'
                '  "word_choice_score": <0-3 integer>,\n'
                '  "figurative_score": <0-3 integer>,\n'
                '  "evidence": ["short quote 1", "short quote 2"]\n'
                "}\n\n"
                "Scoring guidance:\n"
                "- 0 = ineffective language support for purpose and audience.\n"
                "- 1 = limited language control; support is weak.\n"
                "- 2 = adequate language control; support is clear.\n"
                "- 3 = compelling language control that clearly supports purpose and audience.\n"
                "- Figurative language is optional support, not required for high score.\n"
                "- Provide exactly 2 short direct quotes from the essay as evidence.\n"
                f"Essay:\n{essay}\n"
            )

        if trait == "conventions":
            return (
                "You are evaluating CONVENTIONS only for an ASAP Prompt 7 essay.\n"
                "Important:\n"
                "- Ignore anonymization tokens like @CAPS, @PERSON, @LOCATION, @NUM.\n"
                "- Count observable errors in the essay text only.\n"
                "- Do NOT give final trait score.\n"
                "- Do NOT restate questions.\n"
                "Return STRICT JSON only:\n"
                "{\n"
                '  "grammar_errors": <int>,\n'
                '  "spelling_errors": <int>,\n'
                '  "punctuation_errors": <int>,\n'
                '  "capitalization_errors": <int>,\n'
                '  "sentence_boundary_errors": <int>,\n'
                '  "error_severity": "none|minor|few|moderate|some|frequent|many",\n'
                '  "evidence": ["error example 1", "error example 2"]\n'
                "}\n\n"
                f"Essay:\n{essay}\n"
            )

        if trait in {"ideas", "organization"} and self.mapper.ENABLE_IDEAS_ORG_STRUCTURED_MAIN:
            return self._build_ideas_org_structured_prompt(essay, trait, for_shadow=False)

        numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
        return (
            f"You are analyzing an ASAP Prompt 7 student essay for trait: {trait}.\n"
            "Answer each question with short evidence-based reasoning.\n"
            "Do not give numeric scores.\n\n"
            f"Essay:\n{essay}\n\n"
            f"Questions:\n{numbered}\n\n"
            "Return answers as a numbered list matching the questions."
        )

    def _build_ideas_org_structured_prompt(self, essay: str, trait: str, for_shadow: bool) -> str:
        mode_note = "shadow analysis" if for_shadow else "main analysis"
        if trait == "ideas":
            return (
                "You are evaluating IDEAS only for an ASAP Prompt 7 essay.\n"
                f"This is for {mode_note}; do NOT output final trait score.\n"
                "Return STRICT JSON only:\n"
                "{\n"
                '  "focus": {"band": "none|limited|adequate|strong", "score": 0},\n'
                '  "central_idea": {"band": "none|limited|adequate|strong", "score": 0},\n'
                '  "evidence": {"band": "none|limited|adequate|strong", "score": 0},\n'
                '  "elaboration": {"band": "none|limited|adequate|strong", "score": 0},\n'
                '  "relevance": {"band": "none|limited|adequate|strong", "score": 0},\n'
                '  "evidence_quotes": ["short quote 1", "short quote 2"],\n'
                '  "confidence": "low|medium|high"\n'
                "}\n"
                "Band anchors for each field:\n"
                "- focus: none=not focused on the assigned topic; limited=partly focused or drifting; adequate=generally focused; strong=sharply and consistently focused.\n"
                "- central_idea: none=no clear controlling idea; limited=weak/basic controlling idea; adequate=clear main idea; strong=clear and sustained controlling idea.\n"
                "- evidence: none=no specific support; limited=few/vague details; adequate=some specific supporting details; strong=multiple concrete supporting details.\n"
                "- elaboration: none=undeveloped; limited=thin/basic development; adequate=clear development of ideas/events; strong=sustained, detailed development.\n"
                "- relevance: none=mostly off-topic or irrelevant; limited=some clear drift/irrelevance; adequate=generally relevant with minor drift; strong=fully relevant with no meaningful drift.\n"
                "Rules:\n"
                "- For each component, score must match band exactly: none=0, limited=1, adequate=2, strong=3.\n"
                "- Choose the best-fitting band from the essay evidence; do not force all components to one band.\n"
                "- Use strong only when performance is clearly sustained, not just briefly present.\n"
                "- Provide exactly 2 short direct quotes copied from the essay (each <= 12 words).\n"
                "- Quotes must be literal text from the essay, not placeholders.\n"
                "- No markdown. No explanation. JSON object only.\n\n"
                f"Essay:\n{essay}\n"
            )
        return (
            "You are evaluating ORGANIZATION only for an ASAP Prompt 7 essay.\n"
            f"This is for {mode_note}; do NOT output final trait score.\n"
            "Return STRICT JSON only:\n"
            "{\n"
            '  "lead_closure": {"band": "none|limited|adequate|strong", "score": 0},\n'
            '  "progression": {"band": "none|limited|adequate|strong", "score": 0},\n'
            '  "transitions": {"band": "none|limited|adequate|strong", "score": 0},\n'
            '  "coherence": {"band": "none|limited|adequate|strong", "score": 0},\n'
            '  "evidence_quotes": ["short quote 1", "short quote 2"],\n'
            '  "confidence": "low|medium|high"\n'
            "}\n"
            "Band anchors:\n"
            "- none: no clear lead/closure or logical structure.\n"
            "- limited: some structure, but weak progression/linking.\n"
            "- adequate: generally clear progression and coherence with recognizable structure.\n"
            "- strong: deliberate, cohesive, and effective structure throughout.\n"
            "Rules:\n"
            "- For each component, score must match band exactly: none=0, limited=1, adequate=2, strong=3.\n"
            "- Choose the best-fitting band from the essay evidence; do not force all components to one band.\n"
            "- Use strong only when organization is clearly effective across the essay.\n"
            "- Provide exactly 2 short direct quotes copied from the essay (each <= 12 words).\n"
            "- Quotes must be literal text from the essay, not placeholders.\n"
            "- No markdown. No explanation. JSON object only.\n\n"
            f"Essay:\n{essay}\n"
        )

    def _build_ideas_org_shadow_prompt(self, essay: str, trait: str) -> str:
        return self._build_ideas_org_structured_prompt(essay, trait, for_shadow=True)

    def _build_organization_diagnostic_v2_prompt(self, essay: str) -> str:
        return (
            "You are evaluating ORGANIZATION only for an ASAP Prompt 7 essay.\n"
            "This prompt is for a diagnostic organization analysis; do NOT output markdown.\n"
            "Return STRICT JSON only:\n"
            "{\n"
            '  "overall_score": 0,\n'
            '  "story_frame": {"band": "none|limited|adequate|strong", "score": 0},\n'
            '  "paragraph_plan": {"band": "none|limited|adequate|strong", "score": 0},\n'
            '  "progression_control": {"band": "none|limited|adequate|strong", "score": 0},\n'
            '  "linking": {"band": "none|limited|adequate|strong", "score": 0},\n'
            '  "unity": {"band": "none|limited|adequate|strong", "score": 0},\n'
            '  "dominant_pattern": "scatter|list_like|partly_ordered|clear_arc|cohesive_arc",\n'
            '  "primary_limiter": "none|story_frame|paragraph_plan|progression_control|linking|unity|multiple",\n'
            '  "secondary_limiter": "none|story_frame|paragraph_plan|progression_control|linking|unity|multiple",\n'
            '  "confidence": "low|medium|high",\n'
            '  "evidence_quotes": ["short quote 1", "short quote 2"]\n'
            "}\n"
            "Overall score anchors:\n"
            "- 0: no stable beginning-middle-end, hard to follow, major jumps or fragments.\n"
            "- 1: some order exists, but grouping or progression is shaky and the reader must repair gaps.\n"
            "- 2: clear enough structure and order overall, but control is uneven or not sustained.\n"
            "- 3: deliberate, cohesive, easy-to-follow structure across the whole essay.\n"
            "Field anchors:\n"
            "- story_frame: none=no clear setup or ending; limited=weak setup or ending only; adequate=recognizable beginning and ending; strong=effective framing around the whole story.\n"
            "- paragraph_plan: none=no clear grouping; limited=some breaks but weak grouping; adequate=body parts are meaningfully grouped; strong=paragraphing clearly helps the story unfold.\n"
            "- progression_control: none=events or ideas feel out of order; limited=some order but jumps or restarts; adequate=generally easy to follow; strong=purposeful and smooth progression.\n"
            "- linking: none=missing or confusing links; limited=few/basic links; adequate=useful links or references; strong=links consistently guide the reader.\n"
            "- unity: none=essay keeps jumping or restarting; limited=one main line exists but drift or jumps are common; adequate=mostly stays on one line of development; strong=stays tightly unified throughout.\n"
            "Pattern anchors:\n"
            "- scatter=no stable line of movement.\n"
            "- list_like=ordered fragments or event list with weak glue.\n"
            "- partly_ordered=recognizable line but uneven control.\n"
            "- clear_arc=clear beginning-middle-end.\n"
            "- cohesive_arc=clear arc with strong control throughout.\n"
            "Rules:\n"
            "- Choose overall_score FIRST, then use the fields to justify it.\n"
            "- Do NOT average the fields mechanically.\n"
            "- If progression_control or unity is none, overall_score 2 or 3 is usually wrong.\n"
            "- If paragraph_plan is none and story_frame is none or limited, overall_score 2 or 3 is usually wrong.\n"
            "- Use score 3 only when control is sustained across the essay, not just present briefly.\n"
            "- Provide exactly 2 short direct quotes copied from the essay (each <= 12 words).\n"
            "- Quotes must be literal text from the essay, not placeholders.\n"
            "- No explanation outside JSON.\n\n"
            f"Essay:\n{essay}\n"
        )

    def _generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text_input = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                top_p=1.0,
                top_k=50,
            )
        return self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

    @staticmethod
    def _extract_json_obj(text: str) -> dict | None:
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    def _generate_json_with_retry(self, base_prompt: str) -> tuple[str, dict | None]:
        raw = self._generate(base_prompt)
        parsed = self._extract_json_obj(raw)
        if not self.enable_json_retry:
            return raw, parsed
        if parsed is not None:
            return raw, parsed

        retry_prompt = (
            base_prompt
            + "\n\nYour previous answer was not valid JSON.\n"
            + "Reply with ONE valid JSON object only. No markdown. No explanation."
        )
        raw_retry = self._generate(retry_prompt)
        parsed_retry = self._extract_json_obj(raw_retry)
        return raw_retry, parsed_retry

    @staticmethod
    def _normalize_text_for_match(text: str) -> str:
        lowered = text.lower()
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered

    def _validate_style_json(self, payload: dict, essay: str) -> tuple[bool, list[str]]:
        reasons = []
        required_keys = [
            "purpose_audience_support",
            "voice_presence",
            "sentence_variety",
            "word_choice",
            "figurative_language",
            "purpose_audience_score",
            "voice_score",
            "sentence_variety_score",
            "word_choice_score",
            "figurative_score",
            "evidence",
        ]
        for key in required_keys:
            if key not in payload:
                reasons.append(f"missing_{key}")

        score_keys = [
            "purpose_audience_score",
            "voice_score",
            "sentence_variety_score",
            "word_choice_score",
            "figurative_score",
        ]
        for key in score_keys:
            value = payload.get(key, None)
            if not isinstance(value, int) or value < 0 or value > 3:
                reasons.append(f"invalid_{key}")

        evidence = payload.get("evidence", [])
        if not isinstance(evidence, list):
            reasons.append("evidence_not_list")
            evidence = []
        non_empty_evidence = [str(x).strip() for x in evidence if str(x).strip()]
        if len(non_empty_evidence) < 2:
            reasons.append("evidence_less_than_2")
        else:
            essay_norm = self._normalize_text_for_match(essay)
            missing_quote_count = 0
            for ev in non_empty_evidence[:2]:
                ev_norm = self._normalize_text_for_match(ev.strip(" \"'"))
                if len(ev_norm) < 5 or ev_norm not in essay_norm:
                    missing_quote_count += 1
            if missing_quote_count > 0:
                reasons.append("evidence_not_verbatim_quote")

        return len(reasons) == 0, reasons

    def _build_style_retry_prompt(self, base_prompt: str, reasons: list[str]) -> str:
        reason_text = ", ".join(reasons) if reasons else "unknown_validation_error"
        return (
            base_prompt
            + "\n\nYour previous JSON failed validation for these reasons: "
            + reason_text
            + ".\n"
            + "Fix ALL issues and return ONE valid JSON object only.\n"
            + "Hard constraints:\n"
            + "- Provide exactly 2 short direct quotes copied verbatim from the essay.\n"
            + "- Each component score must be an integer 0-3.\n"
            + "- No markdown, no explanation, JSON only."
        )

    @staticmethod
    def _is_placeholder_quote(text: str) -> bool:
        lowered = (text or "").strip().lower()
        if not lowered:
            return True
        placeholder_patterns = (
            r"^short quote\s*\d*$",
            r"^quote\s*\d*$",
            r"^example\s*\d*$",
            r"^placeholder$",
            r"^n/?a$",
            r"^none$",
            r"^todo$",
        )
        return any(re.search(p, lowered) for p in placeholder_patterns)

    def _generate_style_json_with_retry(self, essay: str, base_prompt: str) -> tuple[str, dict | None]:
        raw = self._generate(base_prompt)
        parsed = self._extract_json_obj(raw)
        if not self.enable_json_retry:
            return raw, parsed if isinstance(parsed, dict) else None
        if isinstance(parsed, dict):
            ok, reasons = self._validate_style_json(parsed, essay)
            if ok:
                return raw, parsed
        else:
            reasons = ["invalid_json"]

        retry_prompt = self._build_style_retry_prompt(base_prompt, reasons)
        raw_retry = self._generate(retry_prompt)
        parsed_retry = self._extract_json_obj(raw_retry)
        if isinstance(parsed_retry, dict):
            ok_retry, _ = self._validate_style_json(parsed_retry, essay)
            if ok_retry:
                return raw_retry, parsed_retry
        return raw_retry, None

    def _validate_ideas_org_json(self, trait: str, payload: dict, essay: str) -> tuple[bool, list[str]]:
        reasons = []
        bands = {"none", "limited", "adequate", "strong"}
        score_by_band = {"none": 0, "limited": 1, "adequate": 2, "strong": 3}
        if trait == "ideas":
            if "central_idea" in payload or "relevance" in payload:
                required_keys = ["focus", "central_idea", "evidence", "elaboration", "relevance", "evidence_quotes"]
            else:
                required_keys = ["focus", "evidence", "elaboration", "consistency", "evidence_quotes"]
        else:
            required_keys = ["lead_closure", "progression", "transitions", "coherence", "evidence_quotes"]
        component_keys = [k for k in required_keys if k != "evidence_quotes"]

        for key in required_keys:
            if key not in payload:
                reasons.append(f"missing_{key}")

        band_values = {}
        for key in required_keys:
            if key == "evidence_quotes":
                continue
            raw_value = payload.get(key, "")
            if isinstance(raw_value, dict):
                band = str(raw_value.get("band", "")).strip().lower()
                score = raw_value.get("score", None)
                if band not in bands:
                    reasons.append(f"invalid_{key}_band")
                    continue
                if not isinstance(score, int) or score < 0 or score > 3:
                    reasons.append(f"invalid_{key}_score")
                    continue
                if score_by_band[band] != score:
                    reasons.append(f"inconsistent_{key}_band_score")
                band_values[key] = band
                continue

            band = str(raw_value).strip().lower()
            if band not in bands:
                reasons.append(f"invalid_{key}")
                continue
            band_values[key] = band

        if len(component_keys) > 0 and all(band_values.get(k) == "none" for k in component_keys):
            reasons.append("all_components_none")

        quotes = payload.get("evidence_quotes", [])
        if not isinstance(quotes, list):
            reasons.append("evidence_quotes_not_list")
            quotes = []
        quotes = [str(x).strip() for x in quotes if str(x).strip()]
        if len(quotes) != 2:
            reasons.append("evidence_quotes_not_exactly_2")
        else:
            essay_norm = self._normalize_text_for_match(essay)
            for q in quotes:
                if self._is_placeholder_quote(q):
                    reasons.append("evidence_quote_placeholder")
                    break
                if len(q.split()) > 12:
                    reasons.append("evidence_quote_too_long")
                    break
                q_norm = self._normalize_text_for_match(q.strip(" \"'"))
                if len(q_norm) < 5 or q_norm not in essay_norm:
                    reasons.append("evidence_quote_not_verbatim")
                    break
            if self._normalize_text_for_match(quotes[0]) == self._normalize_text_for_match(quotes[1]):
                reasons.append("evidence_quotes_duplicate")
        return len(reasons) == 0, reasons

    def _build_ideas_org_retry_prompt(self, base_prompt: str, reasons: list[str]) -> str:
        reason_text = ", ".join(reasons) if reasons else "unknown_validation_error"
        return (
            base_prompt
            + "\n\nYour previous JSON failed validation for these reasons: "
            + reason_text
            + ".\n"
            + "Fix ALL issues and return ONE valid JSON object only.\n"
            + "Hard constraints:\n"
            + "- Each component must be an object: {\"band\":..., \"score\":...}.\n"
            + "- Band must be one of none|limited|adequate|strong and score must match (0|1|2|3).\n"
            + "- Include exactly 2 short entries in evidence_quotes.\n"
            + "- Do not use placeholder text like \"short quote 1\".\n"
            + "- evidence_quotes must be literal text copied from the essay.\n"
            + "- Not all components can be 'none'.\n"
            + "- No markdown, no explanation, JSON only."
        )

    def _validate_organization_diagnostic_v2_json(self, payload: dict, essay: str) -> tuple[bool, list[str]]:
        reasons = []
        bands = {"none", "limited", "adequate", "strong"}
        score_by_band = {"none": 0, "limited": 1, "adequate": 2, "strong": 3}
        required_keys = [
            "overall_score",
            "story_frame",
            "paragraph_plan",
            "progression_control",
            "linking",
            "unity",
            "dominant_pattern",
            "primary_limiter",
            "secondary_limiter",
            "confidence",
            "evidence_quotes",
        ]
        component_keys = ["story_frame", "paragraph_plan", "progression_control", "linking", "unity"]
        valid_patterns = {"scatter", "list_like", "partly_ordered", "clear_arc", "cohesive_arc"}
        valid_limiters = {"none", "story_frame", "paragraph_plan", "progression_control", "linking", "unity", "multiple"}
        valid_confidence = {"low", "medium", "high"}

        for key in required_keys:
            if key not in payload:
                reasons.append(f"missing_{key}")

        overall_score = payload.get("overall_score", None)
        if not isinstance(overall_score, int) or overall_score < 0 or overall_score > 3:
            reasons.append("invalid_overall_score")

        band_values = {}
        for key in component_keys:
            raw_value = payload.get(key, None)
            if not isinstance(raw_value, dict):
                reasons.append(f"invalid_{key}_object")
                continue
            band = str(raw_value.get("band", "")).strip().lower()
            score = raw_value.get("score", None)
            if band not in bands:
                reasons.append(f"invalid_{key}_band")
                continue
            if not isinstance(score, int) or score < 0 or score > 3:
                reasons.append(f"invalid_{key}_score")
                continue
            if score_by_band[band] != score:
                reasons.append(f"inconsistent_{key}_band_score")
            band_values[key] = band

        if len(component_keys) > 0 and all(band_values.get(k) == "none" for k in component_keys):
            reasons.append("all_components_none")

        dominant_pattern = str(payload.get("dominant_pattern", "")).strip().lower()
        if dominant_pattern not in valid_patterns:
            reasons.append("invalid_dominant_pattern")

        for limiter_key in ("primary_limiter", "secondary_limiter"):
            limiter_value = str(payload.get(limiter_key, "")).strip().lower()
            if limiter_value not in valid_limiters:
                reasons.append(f"invalid_{limiter_key}")

        confidence = str(payload.get("confidence", "")).strip().lower()
        if confidence not in valid_confidence:
            reasons.append("invalid_confidence")

        quotes = payload.get("evidence_quotes", [])
        if not isinstance(quotes, list):
            reasons.append("evidence_quotes_not_list")
            quotes = []
        quotes = [str(x).strip() for x in quotes if str(x).strip()]
        if len(quotes) != 2:
            reasons.append("evidence_quotes_not_exactly_2")
        else:
            essay_norm = self._normalize_text_for_match(essay)
            for q in quotes:
                if self._is_placeholder_quote(q):
                    reasons.append("evidence_quote_placeholder")
                    break
                if len(q.split()) > 12:
                    reasons.append("evidence_quote_too_long")
                    break
                q_norm = self._normalize_text_for_match(q.strip(" \"'"))
                if len(q_norm) < 5 or q_norm not in essay_norm:
                    reasons.append("evidence_quote_not_verbatim")
                    break
            if self._normalize_text_for_match(quotes[0]) == self._normalize_text_for_match(quotes[1]):
                reasons.append("evidence_quotes_duplicate")
        return len(reasons) == 0, reasons

    def _build_organization_diagnostic_v2_retry_prompt(self, base_prompt: str, reasons: list[str]) -> str:
        reason_text = ", ".join(reasons) if reasons else "unknown_validation_error"
        return (
            base_prompt
            + "\n\nYour previous JSON failed validation for these reasons: "
            + reason_text
            + ".\n"
            + "Fix ALL issues and return ONE valid JSON object only.\n"
            + "Hard constraints:\n"
            + "- overall_score must be an integer 0-3.\n"
            + "- Each diagnostic field must be an object: {\"band\":..., \"score\":...}.\n"
            + "- Band must be one of none|limited|adequate|strong and score must match 0|1|2|3.\n"
            + "- dominant_pattern must be one of scatter|list_like|partly_ordered|clear_arc|cohesive_arc.\n"
            + "- primary_limiter and secondary_limiter must be valid labels.\n"
            + "- Include exactly 2 short entries in evidence_quotes copied literally from the essay.\n"
            + "- Do not use placeholder text.\n"
            + "- Not all components can be 'none'.\n"
            + "- No markdown, no explanation, JSON only."
        )

    def _generate_ideas_org_json_with_retry(self, essay: str, trait: str, base_prompt: str) -> tuple[str, dict | None]:
        raw = self._generate(base_prompt)
        parsed = self._extract_json_obj(raw)
        if not self.enable_json_retry:
            return raw, parsed if isinstance(parsed, dict) else None
        if isinstance(parsed, dict):
            ok, reasons = self._validate_ideas_org_json(trait, parsed, essay)
            if ok:
                return raw, parsed
        else:
            reasons = ["invalid_json"]

        retry_prompt = self._build_ideas_org_retry_prompt(base_prompt, reasons)
        raw_retry = self._generate(retry_prompt)
        parsed_retry = self._extract_json_obj(raw_retry)
        if isinstance(parsed_retry, dict):
            ok_retry, _ = self._validate_ideas_org_json(trait, parsed_retry, essay)
            if ok_retry:
                return raw_retry, parsed_retry
        return raw_retry, None

    def _generate_organization_diagnostic_v2_json_with_retry(self, essay: str, base_prompt: str) -> tuple[str, dict | None]:
        raw = self._generate(base_prompt)
        parsed = self._extract_json_obj(raw)
        if not self.enable_json_retry:
            return raw, parsed if isinstance(parsed, dict) else None
        if isinstance(parsed, dict):
            ok, reasons = self._validate_organization_diagnostic_v2_json(parsed, essay)
            if ok:
                return raw, parsed
        else:
            reasons = ["invalid_json"]

        retry_prompt = self._build_organization_diagnostic_v2_retry_prompt(base_prompt, reasons)
        raw_retry = self._generate(retry_prompt)
        parsed_retry = self._extract_json_obj(raw_retry)
        if isinstance(parsed_retry, dict):
            ok_retry, _ = self._validate_organization_diagnostic_v2_json(parsed_retry, essay)
            if ok_retry:
                return raw_retry, parsed_retry
        return raw_retry, None

    @staticmethod
    def _json_to_answer_lines(trait: str, parsed_json: dict) -> list[str]:
        if trait == "ideas":
            if "central_idea" in parsed_json or "relevance" in parsed_json:
                keys = ["focus", "central_idea", "evidence", "elaboration", "relevance"]
            else:
                keys = ["focus", "evidence", "elaboration", "consistency"]
            lines = []
            for key in keys:
                value = parsed_json.get(key, "")
                if isinstance(value, dict):
                    band = value.get("band", "")
                    score = value.get("score", "")
                    lines.append(f"{key}: band={band} score={score}")
                else:
                    lines.append(f"{key}: {value}")
            return lines

        if trait == "organization":
            keys = ["lead_closure", "progression", "transitions", "coherence"]
            lines = []
            for key in keys:
                value = parsed_json.get(key, "")
                if isinstance(value, dict):
                    band = value.get("band", "")
                    score = value.get("score", "")
                    lines.append(f"{key}: band={band} score={score}")
                else:
                    lines.append(f"{key}: {value}")
            return lines

        if trait == "style":
            keys = [
                "purpose_audience_support",
                "voice_presence",
                "sentence_variety",
                "word_choice",
                "figurative_language",
            ]
            return [f"{k}: {parsed_json.get(k, '')}" for k in keys]

        if trait == "conventions":
            keys = [
                "grammar_errors",
                "spelling_errors",
                "punctuation_errors",
                "capitalization_errors",
                "sentence_boundary_errors",
                "error_severity",
            ]
            return [f"{k}: {parsed_json.get(k, '')}" for k in keys]

        return []

    @staticmethod
    def _split_numbered_answers(text: str, expected: int) -> list[str]:
        pattern = r"(?:^|\n)\s*\d+[.)、:：-]?\s*(.+?)(?=(?:\n\s*\d+[.)、:：-]?\s)|$)"
        matches = re.findall(pattern, text, flags=re.DOTALL)
        answers = [m.strip() for m in matches if m.strip()]
        if not answers:
            answers = [text.strip()] if text.strip() else []
        while len(answers) < expected:
            answers.append("No clear answer generated.")
        return answers[:expected]

    def analyze_trait_reasoning(self, essay: str, trait: str, questions: list[str]) -> dict:
        prompt = self._build_reasoning_prompt(essay, trait, questions)
        if trait in {"ideas", "organization"} and self.mapper.ENABLE_IDEAS_ORG_STRUCTURED_MAIN:
            raw_text, parsed_json = self._generate_ideas_org_json_with_retry(essay, trait, prompt)
            if parsed_json is not None:
                answers = self._json_to_answer_lines(trait, parsed_json)
            else:
                answers = self._split_numbered_answers(raw_text, len(questions))
            return {
                "trait": trait,
                "questions": questions,
                "answers": answers,
                "raw_reasoning": raw_text,
                "structured_json": parsed_json,
            }

        if trait == "style":
            raw_text, parsed_json = self._generate_style_json_with_retry(essay, prompt)
            if parsed_json is not None:
                answers = self._json_to_answer_lines(trait, parsed_json)
            else:
                answers = self._split_numbered_answers(raw_text, len(questions))
            return {
                "trait": trait,
                "questions": questions,
                "answers": answers,
                "raw_reasoning": raw_text,
                "structured_json": parsed_json,
            }

        if trait == "conventions":
            raw_text, parsed_json = self._generate_json_with_retry(prompt)
            if parsed_json is not None:
                answers = self._json_to_answer_lines(trait, parsed_json)
            else:
                answers = self._split_numbered_answers(raw_text, len(questions))
            return {
                "trait": trait,
                "questions": questions,
                "answers": answers,
                "raw_reasoning": raw_text,
                "structured_json": parsed_json,
            }

        raw_text = self._generate(prompt)
        answers = self._split_numbered_answers(raw_text, len(questions))
        return {
            "trait": trait,
            "questions": questions,
            "answers": answers,
            "raw_reasoning": raw_text,
            "structured_json": None,
        }

    def _keyword_hits(self, trait: str, reasoning: str) -> tuple[list[str], list[str]]:
        _ = trait
        _ = reasoning
        return [], []

    def score_prompt7_essay(
        self,
        text: str,
        essay_id: int | None = None,
        debug: bool = False,
        enable_ideas_org_shadow: bool = False,
    ) -> dict:
        # Stage 1: fixed question loading from trait_questions_prompt7/
        trait_questions = self.load_prompt7_questions()

        # Stage 2: LLM reasoning answers for each trait
        reasoning = {}
        for trait in ("ideas", "organization", "style", "conventions"):
            reasoning[trait] = self.analyze_trait_reasoning(text, trait, trait_questions[trait])

        # Stage 3: reasoning -> trait scores
        scores = {}
        for trait in ("ideas", "organization", "style", "conventions"):
            scores[trait] = self.mapper.score_trait_from_reasoning(
                trait=trait,
                reasoning=reasoning[trait]["raw_reasoning"],
                essay_text=text,
            )
        overall = self.mapper.compute_overall_score(
            scores["ideas"], scores["organization"], scores["style"], scores["conventions"]
        )
        shadow_outputs = {}
        if enable_ideas_org_shadow:
            for shadow_trait in ("ideas", "organization"):
                shadow_prompt = self._build_ideas_org_shadow_prompt(text, shadow_trait)
                shadow_raw, shadow_json = self._generate_ideas_org_json_with_retry(text, shadow_trait, shadow_prompt)
                shadow_reasoning = (
                    json.dumps(shadow_json, ensure_ascii=False)
                    if isinstance(shadow_json, dict)
                    else shadow_raw
                )
                shadow_score, shadow_meta = self.mapper.score_ideas_org_shadow_from_reasoning(
                    shadow_trait,
                    shadow_reasoning,
                )
                shadow_outputs[shadow_trait] = {
                    "raw_reasoning": shadow_reasoning,
                    "raw_model_output": shadow_raw,
                    "structured_json": shadow_json,
                    "score": shadow_score,
                    "meta": shadow_meta,
                }

        if debug:
            trait_order = ("ideas", "organization", "style", "conventions")
            trait_title = {
                "ideas": "Ideas",
                "organization": "Organization",
                "style": "Style",
                "conventions": "Conventions",
            }

            print("\n" + "=" * 48)
            print("ESSAY ID")
            print("=" * 48)
            print(essay_id if essay_id is not None else "N/A")
            print()
            print(f"Essay length: {len(text)}")
            print()
            print("Stage2 Reasoning Output (per trait)")
            print()
            for trait in trait_order:
                print(f"{trait_title[trait]} reasoning:")
                print(reasoning[trait]["raw_reasoning"])
                print()

            print("Stage3 Scoring:")
            print()
            print(f"Ideas score = {scores['ideas']}")
            print(f"Organization score = {scores['organization']}")
            print(f"Style score = {scores['style']}")
            print(f"Conventions score = {scores['conventions']}")
            if enable_ideas_org_shadow:
                print(f"Shadow ideas score = {shadow_outputs.get('ideas', {}).get('score')}")
                print(f"Shadow organization score = {shadow_outputs.get('organization', {}).get('score')}")
            print()
            print("Keyword hits:")
            print()
            for trait in trait_order:
                pos_hits, neg_hits = self._keyword_hits(trait, "\n".join(reasoning[trait]["answers"]))
                print(f"{trait_title[trait]} positive hits: {pos_hits}")
                print(f"{trait_title[trait]} negative hits: {neg_hits}")
                print()
            print("=" * 48)

        return {
            "ideas": scores["ideas"],
            "organization": scores["organization"],
            "style": scores["style"],
            "conventions": scores["conventions"],
            "overall": overall,
            "reasoning": reasoning,
            "shadow": shadow_outputs,
        }
