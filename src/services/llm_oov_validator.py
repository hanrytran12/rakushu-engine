"""AI OOV Validator & Enhancer: Evaluates morphological validity and generates definitions."""
import re
import logging
from typing import Optional, Dict, Any, Tuple
from src.models import TokenModel, OovCandidate

logger = logging.getLogger("OovValidator")

JAPANESE_CHAR_PATTERN = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")
DISCARD_PUNCT_PATTERN = re.compile(r"^[0-9\s\-_.,!?:;\"'()\[\]{}、。！？・…〜~]+$")
DISCARD_PARTICLES = {"は", "が", "を", "に", "へ", "で", "と", "から", "より", "まで", "て", "で", "ね", "よ"}


class LlmOovValidator:
    """Validates candidate tokens against noise/glitches and enriches valid novel vocabulary."""

    def __init__(self, llm_service=None):
        self.llm_service = llm_service

    def is_morphologically_plausible(self, term: str) -> bool:
        """Fast linguistic pre-filter: rejects non-Japanese characters, single particles, and punctuation."""
        if not term or len(term.strip()) == 0:
            return False
        clean = term.strip()
        if DISCARD_PUNCT_PATTERN.match(clean):
            return False
        if not JAPANESE_CHAR_PATTERN.search(clean):
            return False
        if clean in DISCARD_PARTICLES:
            return False
        if len(clean) == 1 and "\u3040" <= clean <= "\u309F":
            return False
        return True

    def validate_and_enhance(
        self,
        token: TokenModel,
        context_text: str = ""
    ) -> Tuple[bool, Optional[OovCandidate]]:
        """Flow 4 AI Step: Validates candidate. If valid, enhances with tentative definition."""
        term = token.surface or token.lemma
        if not self.is_morphologically_plausible(term):
            logger.info(f"[OOV Filtered] '{term}' rejected as morphological noise/particle.")
            return False, None

        # 1. Attempt LLM contextual validation & enhancement
        if self.llm_service and hasattr(self.llm_service, "_call_llm"):
            llm_result = self._query_llm_validation(term, token.pos, context_text)
            if llm_result:
                is_valid = llm_result.get("is_valid", False)
                confidence = float(llm_result.get("confidence_score", 0.0))
                if is_valid and confidence >= 0.5:
                    return True, OovCandidate(
                        token_id=token.token_id,
                        term=term,
                        tentative_reading=llm_result.get("tentative_reading") or token.reading or term,
                        suggested_pos=llm_result.get("tentative_pos") or token.pos or "NOUN",
                        suggested_meaning=llm_result.get("suggested_meaning") or f"Từ mới/tiếng lóng: {term}",
                        context_snippet=context_text,
                        confidence_score=confidence,
                        status="PENDING_CURATOR_REVIEW"
                    )
                else:
                    logger.info(f"[OOV Discarded] AI judged '{term}' invalid (confidence: {confidence:.2f}).")
                    return False, None

        # 2. Rule-based / Fallback validation when LLM offline
        return self._heuristic_fallback_validation(token, context_text)

    def _query_llm_validation(self, term: str, pos: str, context: str) -> Optional[Dict[str, Any]]:
        """Asks LLM to evaluate if the term is genuine Japanese slang/term or transcription hallucination."""
        prompt = f"""Bạn là chuyên gia thẩm định ngôn ngữ Nhật Bản.
Hãy thẩm định xem từ vựng sau có phải là một từ vựng tiếng Nhật hợp lệ (từ vựng thật, tiếng lóng giới trẻ, thuật ngữ, hoặc từ ghép mới) hay là lỗi nhận diện âm thanh (transcription noise / hallucination):

Từ cần thẩm định: "{term}" (POS gợi ý: {pos})
Ngữ cảnh câu: "{context}"

Trả về đúng định dạng JSON:
{{
  "is_valid": true,
  "confidence_score": 0.90,
  "tentative_reading": "hiragana",
  "tentative_pos": "NOUN/VERB/SLANG/ADJECTIVE",
  "suggested_meaning": "định nghĩa tiếng Việt súc tích",
  "nuance": "sắc thái sử dụng hoặc nguồn gốc tiếng lóng"
}}"""
        try:
            return self.llm_service._call_llm(prompt, request_purpose="oov_validation")
        except Exception as exc:
            logger.warning(f"LLM validation query failed for '{term}': {exc}")
            return None

    def _heuristic_fallback_validation(
        self,
        token: TokenModel,
        context_text: str
    ) -> Tuple[bool, Optional[OovCandidate]]:
        """Heuristic evaluation when LLM service is offline."""
        term = token.surface or token.lemma
        reading = token.reading if token.reading and token.reading != "*" else term
        pos = token.pos if token.pos not in ("OTHER", "") else "NOUN"
        score = 0.85 if len(term) >= 2 else 0.60

        candidate = OovCandidate(
            token_id=token.token_id,
            term=term,
            tentative_reading=reading,
            suggested_pos=pos,
            suggested_meaning=f"Thuật ngữ mới: {term}",
            context_snippet=context_text,
            confidence_score=score,
            status="PENDING_CURATOR_REVIEW"
        )
        return True, candidate
