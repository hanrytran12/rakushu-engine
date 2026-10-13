"""LLM Service: Contextual translation, linguistic explanation, and structured OOV learning."""
import os
import json
import re
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from src.models import TokenModel, DictionaryEntry, OovCandidate

logger = logging.getLogger("RakushuPipeline")


class LlmEnrichmentService:
    """Enriches transcription with contextual translation, retries, and OOV candidate extraction."""

    def __init__(self):
        self.local_llm_url = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434")
        self.local_llm_model = os.environ.get("LOCAL_LLM_MODEL", "qwen3:4b")
        self.timeout = float(os.environ.get("LLM_TIMEOUT", "3600.0"))
        self.max_retries = int(os.environ.get("LLM_MAX_RETRIES", "3"))
        self.last_sentence_translation_failed = False

    def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_attempts: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Invokes local LLM with bounded retries and returns parsed JSON or None."""
        base = self.local_llm_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        sys_content = system_prompt or "You are Rakushu AI Linguistic Engine. Return JSON only."
        default_num_ctx = max(1, int(os.environ.get("LLM_NUM_CTX", "4096")))
        long_prompt_num_ctx = max(1, int(os.environ.get("LLM_LONG_PROMPT_NUM_CTX", "8192")))
        long_prompt_threshold = max(1, int(os.environ.get("LLM_LONG_PROMPT_THRESHOLD_CHARS", "8000")))
        num_ctx = (
            max(default_num_ctx, long_prompt_num_ctx)
            if len(prompt) > long_prompt_threshold
            else default_num_ctx
        )
        base_options = {
            "temperature": 0.3,
            "repeat_penalty": 1.15,
            "top_p": 0.9,
            "num_ctx": num_ctx,
        }
        retry_options = [
            base_options,
            {"temperature": 0.2, "repeat_penalty": 1.20, "top_p": 0.85, "num_predict": 512, "num_ctx": num_ctx},
            {"temperature": 0.1, "repeat_penalty": 1.25, "top_p": 0.80, "num_predict": 256, "num_ctx": num_ctx},
        ]
        last_status = 0

        attempts = max(1, max_attempts if max_attempts is not None else self.max_retries)
        for attempt in range(1, attempts + 1):
            payload = {
                "model": self.local_llm_model,
                "messages": [
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": prompt}
                ],
                "think": False, "stream": False, "format": "json",
                "options": retry_options[min(attempt - 1, len(retry_options) - 1)]
            }
            diagnostics_enabled = os.environ.get("RAKUSHU_LLM_DIAGNOSTICS", "0") == "1"
            request_tag = time.time_ns() if diagnostics_enabled else None
            request_started = time.perf_counter()
            try:
                import requests
                res = requests.post(f"{base}/api/chat", json=payload, timeout=self.timeout)
                request_elapsed_ms = (time.perf_counter() - request_started) * 1000
                last_status = res.status_code
                if diagnostics_enabled and res.status_code != 200:
                    logger.info(
                        "[LLM DIAGNOSTIC %s] model=%s attempt=%s status=%s "
                        "outcome=http_error request_ms=%.2f response_chars=%s",
                        request_tag, self.local_llm_model, attempt, res.status_code,
                        request_elapsed_ms, len(str(getattr(res, "text", "") or "")),
                    )
                if res.status_code == 200:
                    response_payload = res.json()
                    raw_content = response_payload.get("message", {}).get("content", "")
                    if diagnostics_enabled:
                        def metric_number(field: str):
                            try:
                                value = float(response_payload.get(field))
                                if value < 0:
                                    return None
                                return int(value) if value.is_integer() else value
                            except (TypeError, ValueError):
                                return None

                        def metric_ms(field: str):
                            value = metric_number(field)
                            return round(value / 1_000_000, 2) if value is not None else "n/a"

                        def metric_display(field: str):
                            value = metric_number(field)
                            return value if value is not None else "n/a"

                        def tokens_per_second(count_field: str, duration_field: str):
                            count = metric_number(count_field)
                            duration_ns = metric_number(duration_field)
                            if count is None or duration_ns is None or duration_ns <= 0:
                                return "n/a"
                            return round(count * 1_000_000_000 / duration_ns, 2)

                        logger.info(
                            "[LLM DIAGNOSTIC %s] model=%s attempt=%s status=%s "
                            "prompt_chars=%s response_chars=%s num_ctx=%s http_elapsed_ms=%.2f "
                            "ollama_total_ms=%s model_load_ms=%s prompt_eval_count=%s "
                            "prompt_eval_ms=%s prompt_eval_tokens_per_sec=%s eval_count=%s "
                            "eval_ms=%s generation_tokens_per_sec=%s",
                            request_tag, self.local_llm_model, attempt, res.status_code,
                            len(prompt), len(raw_content), num_ctx, request_elapsed_ms,
                            metric_ms("total_duration"), metric_ms("load_duration"),
                            metric_display("prompt_eval_count"), metric_ms("prompt_eval_duration"),
                            tokens_per_second("prompt_eval_count", "prompt_eval_duration"),
                            metric_display("eval_count"), metric_ms("eval_duration"),
                            tokens_per_second("eval_count", "eval_duration"),
                        )
                    try:
                        data = self._extract_json(raw_content)
                    except Exception as parse_exc:
                        if diagnostics_enabled:
                            logger.warning(
                                "[LLM DIAGNOSTIC %s] JSON parse failed: %s: %s",
                                request_tag, type(parse_exc).__name__, parse_exc,
                            )
                        raise
                    if diagnostics_enabled:
                        returned_sentences = data.get("sentences", [])
                        returned_ids = [
                            item.get("sentence_number")
                            for item in returned_sentences
                            if isinstance(item, dict)
                        ] if isinstance(returned_sentences, list) else []
                        logger.info(
                            "[LLM DIAGNOSTIC %s] parsed_type=%s keys=%s sentence_ids=%s",
                            request_tag, type(data).__name__,
                            list(data.keys()) if isinstance(data, dict) else [],
                            returned_ids,
                        )
                    if data:
                        return data
                response_chars = len(str(getattr(res, "text", "") or ""))
                logger.warning(
                    "[LLM Retry %s/%s] Status %s (response_chars=%s)",
                    attempt, attempts, res.status_code, response_chars,
                )
            except Exception as exc:
                request_elapsed_ms = (time.perf_counter() - request_started) * 1000
                if diagnostics_enabled:
                    logger.info(
                        "[LLM DIAGNOSTIC %s] model=%s attempt=%s outcome=exception "
                        "request_ms=%.2f error_type=%s",
                        request_tag, self.local_llm_model, attempt, request_elapsed_ms,
                        type(exc).__name__,
                    )
                logger.warning(f"[LLM Retry {attempt}/{attempts}] Call failed: {exc}")

            if attempt < attempts:
                time.sleep(1.0)

        logger.error(
            f"[LLM FAILED] Failed all {attempts} attempts "
            f"(last_status={last_status})."
        )
        return None

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extracts JSON object from response string."""
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(m.group(0)) if m else {}

    def build_translation_chunks(
        self,
        sentences: List[str],
        max_chars: int = 900,
        max_sentences: int = 12,
    ) -> List[Tuple[int, int]]:
        """Plans bounded chunks using both character and sentence-count limits.

        Sentence count is the primary safety bound because the chunk response contains
        structured translation/token/OOV data for every sentence.
        """
        chunks: List[Tuple[int, int]] = []
        start = 0
        chars = 0
        for idx, sentence in enumerate(sentences):
            size = len(sentence)
            count = idx - start

            if idx > start and (chars + size > max_chars or count >= max_sentences):
                chunks.append((start, idx))
                start, chars = idx, 0

            chars += size

        if start < len(sentences):
            chunks.append((start, len(sentences)))
        return chunks

    @staticmethod
    def _compact_chunk_prompt_items(
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Compacts occurrence candidates, showing each dictionary gloss only once per entry."""
        compact_items: List[Dict[str, Any]] = []
        for item in items:
            compact_candidates: Dict[str, Dict[str, Any]] = {}
            raw_candidates = item.get("token_candidates", {})
            if isinstance(raw_candidates, dict):
                for token_key, candidates in raw_candidates.items():
                    if not isinstance(candidates, list):
                        continue

                    valid_candidates = []
                    entry_refs: Dict[Tuple[str, str, str, str], str] = {}
                    gloss_emitted = set()
                    for candidate in candidates:
                        if not isinstance(candidate, dict):
                            continue
                        candidate_id = str(candidate.get("candidate_id", "")).strip()
                        meaning = str(candidate.get("meaning", "")).strip()
                        if not candidate_id or not meaning:
                            continue

                        term = str(candidate.get("term", "")).strip()
                        reading = str(candidate.get("reading", "")).strip()
                        dictionary_pos = str(candidate.get("pos", "")).strip()
                        gloss_en = str(candidate.get("gloss_en", "")).strip()[:240]
                        entry_identity = (term, reading, dictionary_pos, gloss_en)
                        entry_ref = entry_refs.get(entry_identity)
                        if entry_ref is None:
                            entry_ref = f"{token_key}-e{len(entry_refs) + 1}"
                            entry_refs[entry_identity] = entry_ref

                        include_gloss = entry_identity not in gloss_emitted
                        gloss_emitted.add(entry_identity)
                        valid_candidates.append({
                            "candidate_id": candidate_id,
                            "entry_ref": entry_ref,
                            "term": term,
                            "reading": reading,
                            "pos": dictionary_pos,
                            "meaning": meaning,
                            "gloss_en": gloss_en if include_gloss else "",
                        })

                    if valid_candidates:
                        first_candidate = next(
                            (
                                candidate for candidate in candidates
                                if isinstance(candidate, dict)
                                and str(candidate.get("candidate_id", "")).strip()
                                and str(candidate.get("meaning", "")).strip()
                            ),
                            {},
                        )
                        compact_candidates[str(token_key)] = {
                            "token": str(first_candidate.get("token", "")).strip(),
                            "lemma": str(first_candidate.get("lemma", "")).strip(),
                            "token_pos": str(first_candidate.get("token_pos", "")).strip(),
                            "candidates": valid_candidates,
                        }
            compact_items.append({
                "sentence_number": int(item["sentence_number"]),
                "text": str(item.get("text", "")),
                "token_candidates": compact_candidates,
            })
        return compact_items

    def _translate_chunk_batch(
        self,
        items: List[Dict[str, Any]],
        max_attempts: int = 2,
    ) -> Dict[int, Dict[str, Any]]:
        """Runs one bounded batch request and validates sentence IDs."""
        if not items:
            return {}

        prompt_items = self._compact_chunk_prompt_items(items)
        prompt = f"""Bạn là engine dịch Nhật-Việt cho Rakushu.
Dịch và làm giàu TẤT CẢ các câu dưới đây trong một lần gọi.
Giữ nguyên số câu và tuyệt đối không gộp, bỏ hoặc đổi số câu.
Ngữ cảnh trong cùng chunk được dùng để hiểu đại từ, chủ thể và liên kết diễn ngôn. Candidate payload được nhóm theo token_key; mỗi nhóm có token, lemma, token_pos và candidates gồm candidate_id, entry_ref, term, reading, POS, meaning_vi, gloss_en. Nhiều candidate_id có thể cùng entry_ref nhưng mang meaning_vi khác nhau. gloss_en chỉ được ghi ở candidate đầu tiên của một entry để tránh lặp payload; nó mô tả entry và có thể chứa nhiều nghĩa, không phải bản dịch 1:1 của từng meaning_vi. Dùng gloss_en của candidate đầu tiên cùng entry_ref làm bằng chứng từ vựng chính; meaning_vi là tham khảo/fallback, không bắt buộc là cách diễn đạt tiếng Việt tốt nhất trong câu. Với MỖI token_key có candidates, phải trả về đúng một token_selection; không được bỏ sót chỉ vì bản dịch câu đã hoàn tất. Chỉ chọn candidate_id có sẵn trong đúng token_key, ưu tiên entry_ref/term/reading/POS phù hợp. Đối chiếu gloss_en với ngữ cảnh để xác định sense tiếng Anh phù hợp, sau đó điền selected_sense_en bằng cụm nghĩa ngắn có căn cứ trong gloss_en và contextual_meaning_vi bằng nghĩa tiếng Việt ngắn gọn, đúng ngữ cảnh. contextual_meaning_vi được phép khác meaning_vi có sẵn nếu gloss_en và câu nguồn hỗ trợ cách hiểu đó; không dịch tất cả sense trong gloss_en cùng lúc. Không chọn nghĩa chỉ vì mặt chữ giống nhau nếu POS/reading không phù hợp. Không tự sinh candidate_id hoặc gán candidate từ token_key khác. Nếu gloss_en trống hoặc không đủ thông tin, dùng meaning_vi làm fallback và không đoán một nghĩa mới.

INPUT:
{json.dumps(prompt_items, ensure_ascii=False)}

Trả về JSON đúng cấu trúc:
{{
  "sentences": [
    {{
      "sentence_number": 1,
      "translation_vi": "...",
      "token_selections": [{{"token_key": "t1", "candidate_id": "t1-c1", "selected_sense_en": "sense grounded in gloss_en", "contextual_meaning_vi": "nghĩa tiếng Việt theo ngữ cảnh"}}],
      "oov_learning": [{{"term": "...", "pos": "NOUN", "suggested_meaning": "...", "confidence_score": 0.95}}]
    }}
  ]
}}
Chỉ trả JSON. Không thêm markdown hay giải thích.
"""
        res = self._call_llm(prompt, max_attempts=max_attempts)
        if not res or not isinstance(res.get("sentences"), list):
            return {}

        expected = {int(item["sentence_number"]) for item in items}
        parsed: Dict[int, Dict[str, Any]] = {}
        for item in res["sentences"]:
            if not isinstance(item, dict):
                continue
            try:
                number = int(item.get("sentence_number"))
            except (TypeError, ValueError):
                continue
            translation = str(item.get("translation_vi", "")).strip()
            if number in expected and number not in parsed and translation:
                normalized_selections = []
                for selection in item.get("token_selections", []) or []:
                    if isinstance(selection, dict):
                        normalized = dict(selection)
                        normalized["sentence_number"] = number
                        normalized_selections.append(normalized)
                item["token_selections"] = normalized_selections
                parsed[number] = item
        return parsed

    @staticmethod
    def _find_translation_collapse_groups(
        items: List[Dict[str, Any]],
        parsed: Dict[int, Dict[str, Any]],
    ) -> List[List[int]]:
        """Finds exact repeated translations across at least three distinct source sentences."""
        item_by_number = {
            int(item["sentence_number"]): item
            for item in items
        }
        groups: Dict[str, List[int]] = {}
        source_texts: Dict[int, str] = {}

        for number, item in item_by_number.items():
            source_texts[number] = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()
            result = parsed.get(number)
            if not result:
                continue
            translation = re.sub(
                r"\s+", " ", str(result.get("translation_vi", "")).strip()
            ).casefold()
            if translation:
                groups.setdefault(translation, []).append(number)

        collapsed: List[List[int]] = []
        for numbers in groups.values():
            distinct_sources = {source_texts[number] for number in numbers}
            if len(numbers) >= 3 and len(distinct_sources) >= 3:
                collapsed.append(sorted(numbers))
        return collapsed

    @staticmethod
    def _normalize_translation(text: str) -> str:
        """Normalizes whitespace and case for repeat-detection comparisons."""
        return re.sub(r"\s+", " ", str(text or "").strip()).casefold()

    def _translate_chunk_adaptive(
        self,
        items: List[Dict[str, Any]],
        depth: int = 0,
    ) -> Dict[int, Dict[str, Any]]:
        """Recovers failed chunk translations by retrying missing items, then splitting."""
        if not items:
            return {}

        parsed = self._translate_chunk_batch(items, max_attempts=2)

        # Semantic guard: non-empty output is not enough when unrelated source
        # sentences all receive the same translation. Retry suspect outputs one by
        # one; unresolved items remain missing so the pipeline can use its existing
        # sentence-level fallback rather than publishing a known-collapsed batch.
        collapse_groups = self._find_translation_collapse_groups(items, parsed)
        if collapse_groups:
            suspect_numbers = sorted({
                number for group in collapse_groups for number in group
            })
            logger.warning(
                f"[Chunk Quality Validation] Suspected repeated-translation collapse "
                f"for sentences={suspect_numbers}; retrying individually."
            )
            item_by_number = {
                int(item["sentence_number"]): item
                for item in items
            }
            for number in suspect_numbers:
                previous = parsed.get(number, {})
                previous_translation = self._normalize_translation(
                    previous.get("translation_vi", "")
                )
                recovered = self._translate_chunk_batch(
                    [item_by_number[number]], max_attempts=2
                ).get(number)
                recovered_translation = self._normalize_translation(
                    recovered.get("translation_vi", "") if recovered else ""
                )
                if recovered_translation and recovered_translation != previous_translation:
                    parsed[number] = recovered
                    logger.info(
                        f"[Chunk Quality Validation] Sentence #{number} recovered "
                        "with an isolated translation."
                    )
                else:
                    parsed.pop(number, None)
                    logger.warning(
                        f"[Chunk Quality Validation] Sentence #{number} still collapsed "
                        "or empty after isolated retry; leave for sentence-level fallback."
                    )
            return parsed


        expected = {int(item["sentence_number"]) for item in items}
        missing = sorted(expected - set(parsed))

        if not missing:
            return parsed

        logger.warning(
            f"[Chunk Validation] Batch returned {len(parsed)}/{len(items)}; "
            f"missing={missing}"
        )

        # Retry only the missing subset once, instead of retrying the original large batch.
        missing_items = [
            item for item in items
            if int(item["sentence_number"]) in missing
        ]
        if missing_items and len(missing_items) < len(items):
            logger.info(
                f"[Chunk Recovery] Retrying only {len(missing_items)} missing sentences "
                f"from batch size {len(items)}."
            )
            recovered = self._translate_chunk_batch(missing_items, max_attempts=2)
            parsed.update(recovered)
            missing = sorted(expected - set(parsed))
            if not missing:
                return parsed

        # If the batch remains incomplete or completely failed, split it.
        # This prevents repeated expensive retries of the same oversized prompt.
        if len(items) > 1:
            midpoint = len(items) // 2
            left_items = items[:midpoint]
            right_items = items[midpoint:]
            logger.warning(
                f"[Chunk Adaptive Split] depth={depth} splitting "
                f"{len(items)} sentences into {len(left_items)} + {len(right_items)}."
            )
            left_results = self._translate_chunk_adaptive(left_items, depth + 1)
            right_results = self._translate_chunk_adaptive(right_items, depth + 1)
            parsed.update(left_results)
            parsed.update(right_results)

        final_missing = sorted(expected - set(parsed))
        if final_missing:
            logger.warning(
                f"[Chunk Recovery] Still missing after adaptive split: {final_missing}"
            )
        return parsed

    def _split_chunk_items_by_budget(
        self,
        items: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        """Splits incoming chunks before inference by prompt payload and output size."""
        max_sentences = max(1, int(os.environ.get("LLM_CHUNK_MAX_SENTENCES", "4")))
        max_payload_chars = max(
            1, int(os.environ.get("LLM_CHUNK_PROMPT_PAYLOAD_MAX_CHARS", "7000"))
        )
        groups: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []

        for item in items:
            candidate_group = current + [item]
            compact_payload_chars = len(json.dumps(
                self._compact_chunk_prompt_items(candidate_group),
                ensure_ascii=False,
            ))
            exceeds_sentence_limit = len(candidate_group) > max_sentences
            exceeds_payload_limit = compact_payload_chars > max_payload_chars
            if current and (exceeds_sentence_limit or exceeds_payload_limit):
                groups.append(current)
                current = [item]
            else:
                current = candidate_group

        if current:
            groups.append(current)
        return groups

    def translate_chunk(
        self,
        sentences: List[Dict[str, Any]],
    ) -> Dict[int, Dict[str, Any]]:
        """Translates a bounded chunk with adaptive recovery and splitting."""
        if not sentences:
            return {}

        items = [
            {
                "sentence_number": int(item["sentence_number"]),
                "text": item["text"],
                # Preserve server-generated candidates for contextual sense selection.
                # The resolver remains the source of truth for final meanings.
                "token_candidates": item.get("token_candidates", {}),
            }
            for item in sentences
        ]
        groups = self._split_chunk_items_by_budget(items)
        if len(groups) == 1:
            return self._translate_chunk_adaptive(groups[0])

        logger.info(
            f"[Chunk Planner] Split batch of {len(items)} sentences into "
            f"{[len(group) for group in groups]} before inference "
            f"(max_sentences={os.environ.get('LLM_CHUNK_MAX_SENTENCES', '4')}, "
            f"max_payload_chars={os.environ.get('LLM_CHUNK_PROMPT_PAYLOAD_MAX_CHARS', '7000')})."
        )
        combined: Dict[int, Dict[str, Any]] = {}
        for group in groups:
            combined.update(self._translate_chunk_adaptive(group))
        return combined

    def build_sentence_context(
        self,
        sentences: List[str],
        target_index: int,
        before: int = 3,
        after: int = 2,
    ) -> str:
        """Builds bounded local context around one target sentence."""
        start = max(0, target_index - before)
        end = min(len(sentences), target_index + after + 1)
        lines = []
        for idx in range(start, end):
            label = "TARGET" if idx == target_index else "CONTEXT"
            lines.append(f"[{label} {idx + 1}] {sentences[idx]}")
        return "\n".join(lines)

    def translate_sentence_with_context(
        self,
        sentence_text: str,
        tokens: List[TokenModel] = None,
        matched_knowledge: List[DictionaryEntry] = None,
        oov_tokens: List[TokenModel] = None,
        token_candidates: Dict[str, List[Dict[str, str]]] = None,
        context_text: str = "",
        expanded_context_text: str = "",
    ) -> Tuple[str, Dict[str, str], List[OovCandidate]]:
        """Stage 2: Translates one sentence using bounded local context."""
        self.last_sentence_translation_failed = False
        tokens = tokens or []
        matched_knowledge = matched_knowledge or []
        oov_tokens = oov_tokens or []
        token_candidates = token_candidates or {}
        known_summary = [f"- {k.term} ({k.reading}): {k.meaning}" for k in matched_knowledge]
        content_tokens = [t.surface for t in tokens if t.pos not in ("PUNCTUATION", "PARTICLE", "AUX_VERB")]
        oov_terms_list = list({tok.surface: tok for tok in oov_tokens}.values())

        prompt = f"""Ngữ cảnh tham khảo (không cần dịch lại):
{context_text or "[Không có ngữ cảnh bổ sung]"}

 Câu cần dịch: "{sentence_text}"
 Từ điển: {known_summary}
 Từ chính: {content_tokens}
 Từ OOV: {[t.surface for t in oov_terms_list]}
 Dictionary token candidates: {json.dumps(token_candidates, ensure_ascii=False)}

Yêu cầu dịch CHỈ câu cần dịch sang tiếng Việt và trả về JSON:
- "translation_vi": bản dịch tiếng Việt của câu
- "token_selections": [{{"token_key": "t1", "candidate_id": "t1-c1", "selected_sense_en": "sense trong gloss_en", "contextual_meaning_vi": "nghĩa tiếng Việt theo ngữ cảnh"}}]; token_key PHẢI là key có trong Dictionary token candidates của câu hiện tại và candidate_id PHẢI thuộc đúng token_key. Dựa vào gloss_en để xác định sense, rồi tự diễn đạt contextual_meaning_vi ngắn gọn bằng tiếng Việt theo câu nguồn; meaning_vi chỉ là tham khảo/fallback. Chỉ tạo contextual_meaning_vi khi candidate có gloss_en phù hợp; nếu không, dùng meaning_vi làm fallback. KHÔNG tự điền token/surface/phrase trong selection.
- "oov_learning": [{{"term": "từ OOV", "pos": "NOUN", "suggested_meaning": "định nghĩa", "confidence_score": 0.95}}]
- "context_sufficient": true nếu ngữ cảnh hiện tại đủ để xác định các đại từ/tham chiếu; false nếu cần thêm ngữ cảnh
- "context_issue": mô tả ngắn điều còn thiếu, nếu có
"""

        logger.info(f"[Sentence Fallback] Target: {sentence_text}")
        logger.info(f"[Sentence Fallback] Initial context: 3 before + target + 2 after")
        logger.info(f"[Sentence Fallback] Initial context text:\n{context_text or '[NO CONTEXT]'}")
        res = self._call_llm(prompt)
        if res and res.get("context_sufficient") is False and expanded_context_text:
            logger.info(f"[Sentence Fallback] Context insufficient for target: {sentence_text}")
            logger.info(f"[Sentence Fallback] Expanding context: 10 before + target + 3 after")
            logger.info(f"[Sentence Fallback] Expanded context text:\n{expanded_context_text}")
            prompt = prompt.replace(context_text or "[Không có ngữ cảnh bổ sung]", expanded_context_text)
            prompt = prompt.replace("Ngữ cảnh hiện tại chưa đủ.", "Đây là lần thử mở rộng ngữ cảnh; ưu tiên ngữ cảnh gần target.")
            res = self._call_llm(prompt)
        if res and "translation_vi" in res:
            vi_trans = str(res.get("translation_vi", "")).strip()
            if vi_trans and vi_trans.lower() not in ("bản dịch tiếng việt tự nhiên", "bản dịch tiếng việt"):
                raw_selections = res.get("token_selections", [])
                t_meanings = self.resolve_token_selections(raw_selections, token_candidates, tokens)
                cands = self._parse_oov_candidates(res.get("oov_learning", []), oov_terms_list, sentence_text)
                logger.info(f"[Sentence Fallback] SUCCESS target='{sentence_text}' | translation='{vi_trans}'")
                return vi_trans, t_meanings, cands

        self.last_sentence_translation_failed = True
        logger.warning(f"[Sentence Fallback] FINAL FAILURE target='{sentence_text}' | returning empty translation")
        km = {d.term: d.meaning.split("(")[0].strip() for d in matched_knowledge}
        default_meanings = {
            t.surface: km.get(t.surface, km.get(t.lemma, ""))
            for t in tokens if t.pos not in ("PUNCTUATION", "PARTICLE", "AUX_VERB")
        }
        return "", default_meanings, []

    def resolve_token_selections(
        self,
        selections: Any,
        token_candidates: Dict[str, List[Dict[str, str]]],
        tokens: List[TokenModel],
        sentence_number: Optional[int] = None,
    ) -> Dict[str, str]:
        """Resolves LLM selections against occurrence-level, server-owned candidates."""
        if not isinstance(selections, list):
            selections = []

        token_by_key = {
            f"t{index + 1}": token
            for index, token in enumerate(tokens)
        }
        candidate_map = {}
        for token_key, candidates in token_candidates.items():
            token = token_by_key.get(token_key)
            for candidate in candidates:
                candidate_id = str(candidate.get("candidate_id", "")).strip()
                candidate_token_id = str(candidate.get("token_id", "")).strip()
                meaning = str(candidate.get("meaning", "")).strip()
                if (
                    token is not None
                    and candidate_token_id == token.token_id
                    and candidate_id
                    and meaning
                ):
                    candidate_map[(token_key, candidate_id)] = meaning

        # Return both token_id and surface aliases for existing downstream consumers.
        # token_id is authoritative, while surface keeps Bunsetsu/API compatibility.
        resolved: Dict[str, str] = {}
        for token_key, candidates in token_candidates.items():
            token = token_by_key.get(token_key)
            if token is None or len(candidates) != 1:
                continue
            candidate = candidates[0]
            candidate_token_id = str(candidate.get("token_id", "")).strip()
            meaning = str(candidate.get("meaning", "")).strip()
            if candidate_token_id != token.token_id or not meaning:
                continue
            resolved[token.token_id] = meaning
            resolved.setdefault(token.surface, meaning)

        for selection in selections:
            if not isinstance(selection, dict):
                continue

            selected_sentence = selection.get("sentence_number")
            if sentence_number is not None and selected_sentence is not None:
                try:
                    if int(selected_sentence) != int(sentence_number):
                        logger.warning(
                            f"[Token Selection Reject] reason=sentence_number_mismatch "
                            f"selected_sentence={selected_sentence!r} expected={sentence_number!r}"
                        )
                        continue
                except (TypeError, ValueError):
                    logger.warning(
                        f"[Token Selection Reject] reason=invalid_sentence_number "
                        f"selected_sentence={selected_sentence!r} expected={sentence_number!r}"
                    )
                    continue

            token_key = str(selection.get("token_key", "")).strip()
            candidate_id = str(selection.get("candidate_id", "")).strip()
            token = token_by_key.get(token_key)
            if token is None:
                logger.warning(
                    f"[Token Selection Reject] reason=unknown_token_key token_key={token_key!r} "
                    f"candidate_id={candidate_id!r}"
                )
                continue

            selected_token = str(selection.get("token", "")).strip()
            if selected_token and selected_token != token.surface:
                logger.warning(
                    f"[Token Selection Reject] reason=token_surface_mismatch token_key={token_key!r} "
                    f"selected_token={selected_token!r} expected={token.surface!r} "
                    f"candidate_id={candidate_id!r}"
                )
                continue

            meaning = candidate_map.get((token_key, candidate_id))
            if not meaning:
                candidates_for_token = token_candidates.get(token_key, [])
                known_candidate_ids = [str(c.get("candidate_id", "")) for c in candidates_for_token]
                if candidate_id not in known_candidate_ids:
                    reason = "unknown_candidate"
                else:
                    candidate_entry = next(
                        (c for c in candidates_for_token if str(c.get("candidate_id", "")) == candidate_id),
                        None,
                    )
                    reason = "candidate_token_id_mismatch" if candidate_entry and str(candidate_entry.get("token_id", "")) != token.token_id else "missing_meaning"
                logger.warning(
                    f"[Token Selection Reject] reason={reason} token_key={token_key!r} "
                    f"candidate_id={candidate_id!r} token_id={token.token_id!r}"
                )
                continue

            selected_candidate = next(
                (
                    candidate for candidate in token_candidates.get(token_key, [])
                    if str(candidate.get("candidate_id", "")).strip() == candidate_id
                ),
                None,
            )
            contextual_meaning = str(selection.get("contextual_meaning_vi", "")).strip()
            selected_sense_en = str(selection.get("selected_sense_en", "")).strip()
            candidate_gloss = str(selected_candidate.get("gloss_en", "")).strip() if selected_candidate else ""

            # Only use generated meanings after validating the exact candidate and token occurrence.
            # Keep the dictionary value as fallback when the chosen entry has no English gloss.
            if contextual_meaning and candidate_gloss:
                if len(contextual_meaning) <= 120 and "\n" not in contextual_meaning:
                    meaning = contextual_meaning
                    logger.debug(
                        "[Token Contextual Meaning] token_key=%s candidate_id=%s "
                        "selected_sense_en=%r contextual_meaning_vi=%r",
                        token_key, candidate_id, selected_sense_en, contextual_meaning,
                    )
                else:
                    logger.warning(
                        "[Token Contextual Meaning Reject] invalid contextual meaning "
                        "for token_key=%s candidate_id=%s; using dictionary fallback.",
                        token_key, candidate_id,
                    )

            resolved[token.token_id] = meaning
            resolved[token.surface] = meaning

        return resolved

    def _parse_oov_candidates(self, items: List[Dict], oov_tokens: List[TokenModel], text: str) -> List[OovCandidate]:
        """Parses learned OOV candidates from LLM output."""
        cands = []
        item_map = {i.get("term", ""): i for i in items if isinstance(i, dict)}
        for tok in oov_tokens:
            item = item_map.get(tok.surface, {})
            meaning = item.get("suggested_meaning") or f"thuật ngữ: {tok.surface}"
            pos = item.get("pos") or tok.pos or "NOUN"
            reading = item.get("reading") or tok.reading or tok.surface
            score = float(item.get("confidence_score") or item.get("score") or 0.85)
            cands.append(OovCandidate(
                token_id=tok.token_id, term=tok.surface, tentative_reading=reading,
                suggested_meaning=meaning, suggested_pos=pos, context_snippet=text,
                confidence_score=score, status="PENDING_CURATOR_REVIEW"
            ))
        return cands