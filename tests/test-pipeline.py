"""Comprehensive test suite for the Rakushu AI processing pipeline."""
import unittest
import os
import sys
import importlib

# Ensure src package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import SubtitleSegment, SentenceSubtitle, OovCandidate, DictionaryEntry, TokenModel
from src.services import NlpService, KnowledgeService, LlmEnrichmentService, BunsetsuService
runner_mod = importlib.import_module("src.pipeline.pipeline_runner")
run_pipeline = runner_mod.run_pipeline


class TestRakushuPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_text = "皆さん、こんにちは。今日のポッドキャストでは、日本の面白い若者言葉について話します。"
        cls.segment = SubtitleSegment(
            start_time=0.0,
            end_time=10.0,
            text=cls.sample_text
        )

    def test_01_ginza_sentence_splitting(self):
        """Validates GiNZA-based sentence boundary splitting."""
        nlp_svc = NlpService()
        sents = nlp_svc.split_sentences(self.sample_text)
        self.assertEqual(len(sents), 2)
        self.assertEqual(sents[0], "皆さん、こんにちは。")
        self.assertEqual(sents[1], "今日のポッドキャストでは、日本の面白い若者言葉について話します。")

    def test_02_nlp_tokenization(self):
        """Validates tokenization, POS tagging, lemmatization, and readings via GiNZA."""
        nlp_svc = NlpService()
        tokens = nlp_svc.tokenize(self.segment)

        self.assertGreater(len(tokens), 8)
        surfaces = [t.surface for t in tokens]
        self.assertIn("皆さん", surfaces)
        self.assertIn("今日", surfaces)
        self.assertIn("ポッドキャスト", surfaces)
        self.assertIn("若者言葉", surfaces)

        # Validate token attributes
        minasan = next(t for t in tokens if t.surface == "皆さん")
        self.assertEqual(minasan.pos, "NOUN")
        self.assertIn(minasan.reading.upper(), ["ミナサン", "みなさん"])

    def test_03_knowledge_and_oov_detection(self):
        """Validates matching against knowledge base and flagging OOV candidates."""
        nlp_svc = NlpService()
        tokens = nlp_svc.tokenize(self.segment)

        knowledge_svc = KnowledgeService()
        matched, oov_tokens = knowledge_svc.match_tokens(tokens)

        matched_terms = [k.term for k in matched]
        oov_surfaces = [t.surface for t in oov_tokens]

        # Verify known words matched
        self.assertIn("皆さん", matched_terms)
        self.assertIn("今日", matched_terms)
        self.assertIn("日本", matched_terms)
        self.assertIn("面白い", matched_terms)

        # Verify omitted words were flagged as OOV
        self.assertIn("ポッドキャスト", oov_surfaces)
        self.assertIn("若者言葉", oov_surfaces)

    def test_03b_candidate_builder_skips_misclassified_particles_and_prefers_lemma(self):
        """Function words mislabeled as OTHER are skipped; inflected verbs use lemma senses."""
        from unittest.mock import patch

        knowledge = KnowledgeService()
        particle = TokenModel(surface="な", lemma="な", reading="ナ", pos="OTHER")
        inflected_verb = TokenModel(surface="話し", lemma="話す", reading="ハナシ", pos="VERB")
        looked_up = []

        def fake_lookup(word, limit=5):
            looked_up.append(word)
            if word == "話す":
                return [
                    DictionaryEntry(term="話す", reading="はなす",
                                    pos="Godan verb with 'su' ending, transitive verb",
                                    meaning="nói"),
                    DictionaryEntry(term="話す", reading="はなす",
                                    pos="Godan verb with 'su' ending, transitive verb",
                                    meaning="chia sẻ"),
                ]
            if word == "話し":
                return [
                    DictionaryEntry(term="話し", reading="はなし",
                                    pos="noun (common) (futsuumeishi)",
                                    meaning="câu chuyện"),
                ]
            if word == "な":
                return [
                    DictionaryEntry(term="菜", reading="な",
                                    pos="noun (common) (futsuumeishi)",
                                    meaning="rau"),
                ]
            return []

        with patch.object(knowledge, "lookup_candidates", side_effect=fake_lookup):
            candidates = knowledge.build_token_candidates([particle, inflected_verb])

        self.assertEqual(list(candidates.keys()), ["t2"])
        self.assertEqual([c["term"] for c in candidates["t2"]], ["話す", "話す"])
        self.assertEqual([c["meaning"] for c in candidates["t2"]], ["nói", "chia sẻ"])
        self.assertEqual(candidates["t2"][0]["token_pos"], "VERB")
        self.assertEqual(looked_up[0], "話す")
        self.assertNotIn("な", looked_up)

    def test_03c_candidate_builder_uses_reading_to_filter_homographs(self):
        """Matching POS is not enough when a dictionary entry uses another reading."""
        from unittest.mock import patch

        knowledge = KnowledgeService()
        token = TokenModel(surface="自然", lemma="自然", reading="シゼン", pos="ADJECTIVE")
        entries = [
            DictionaryEntry(term="自然", reading="じねん",
                            pos="adjectival nouns or quasi-adjectives (keiyodoshi)",
                            meaning="xuất hiện tự nhiên"),
            DictionaryEntry(term="自然", reading="しぜん",
                            pos="adjectival nouns or quasi-adjectives (keiyodoshi)",
                            meaning="tự nhiên"),
        ]
        with patch.object(knowledge, "lookup_candidates", return_value=entries):
            candidates = knowledge.build_token_candidates([token])

        self.assertEqual([c["meaning"] for c in candidates["t1"]], ["tự nhiên"])
        self.assertEqual(candidates["t1"][0]["reading"], "しぜん")

    def test_03d_duplicate_surfaces_keep_occurrence_specific_resolved_meanings(self):
        """Repeated surfaces resolve by token_id so one occurrence cannot overwrite another."""
        llm_svc = LlmEnrichmentService()
        first = TokenModel(surface="思う", lemma="思う", reading="オモウ", pos="VERB")
        second = TokenModel(surface="思う", lemma="思う", reading="オモウ", pos="VERB")
        candidates = {
            "t1": [{
                "candidate_id": "t1-c1", "token_key": "t1", "token_id": first.token_id,
                "token": "思う", "lemma": "思う", "token_pos": "VERB",
                "term": "思う", "reading": "おもう", "pos": "Godan verb",
                "meaning": "nghĩ lần một",
            }],
            "t2": [{
                "candidate_id": "t2-c1", "token_key": "t2", "token_id": second.token_id,
                "token": "思う", "lemma": "思う", "token_pos": "VERB",
                "term": "思う", "reading": "おもう", "pos": "Godan verb",
                "meaning": "nghĩ lần hai",
            }],
        }
        meanings = llm_svc.resolve_token_selections([
            {"sentence_number": 4, "token_key": "t1", "candidate_id": "t1-c1"},
            {"sentence_number": 4, "token_key": "t2", "candidate_id": "t2-c1"},
        ], candidates, [first, second], sentence_number=4)

        for token in (first, second):
            if token.token_id in meanings:
                token.context_meaning = meanings[token.token_id]
            elif token.surface in meanings:
                token.context_meaning = meanings[token.surface]
        self.assertEqual(first.context_meaning, "nghĩ lần một")
        self.assertEqual(second.context_meaning, "nghĩ lần hai")

    def test_03e_candidate_builder_uses_local_context_for_kana_homographs(self):
        """Numeric つ is a counter and こと after いう is the abstract 事 sense."""
        from unittest.mock import patch

        knowledge = KnowledgeService()
        tokens = [
            TokenModel(surface="4", lemma="4", reading="ヨン", pos="OTHER"),
            TokenModel(surface="つ", lemma="つ", reading="ツ", pos="NOUN"),
            TokenModel(surface="の", lemma="の", reading="ノ", pos="PARTICLE"),
            TokenModel(surface="いう", lemma="いう", reading="イウ", pos="PARTICLE"),
            TokenModel(surface="こと", lemma="こと", reading="コト", pos="NOUN"),
        ]

        def fake_lookup(word, limit=5):
            if word == "4":
                return [DictionaryEntry(term="4", reading="ヨン", pos="NOUN", meaning="số 4")]
            if word == "つ":
                return [
                    DictionaryEntry(term="箇", reading="つ", pos="counter", meaning="đếm chung", gloss_en="general-purpose counter"),
                    DictionaryEntry(term="箇", reading="つ", pos="counter", meaning="tuổi", gloss_en="years old"),
                    DictionaryEntry(term="津", reading="つ", pos="noun (common) (futsuumeishi)", meaning="cảng"),
                ]
            if word == "こと":
                return [
                    DictionaryEntry(term="琴", reading="こと", pos="noun (common) (futsuumeishi)", meaning="đàn koto"),
                    DictionaryEntry(term="古都", reading="こと", pos="noun (common) (futsuumeishi)", meaning="thành phố cổ"),
                    DictionaryEntry(term="事", reading="こと", pos="noun (common) (futsuumeishi)", meaning="sự việc"),
                ]
            return []

        with patch.object(knowledge, "lookup_candidates", side_effect=fake_lookup):
            candidates = knowledge.build_token_candidates(tokens)

        self.assertEqual([c["term"] for c in candidates["t2"]], ["箇"])
        self.assertEqual([c["meaning"] for c in candidates["t5"]], ["sự việc"])
        self.assertNotIn("琴", [c["term"] for c in candidates["t5"]])
        self.assertNotIn("津", [c["term"] for c in candidates["t2"]])
        self.assertEqual([c["meaning"] for c in candidates["t2"]], ["đếm chung"])

    def test_04_sentence_level_translation(self):
        """Validates sentence translation with local context and token meanings."""
        llm_svc = LlmEnrichmentService()
        nlp_svc = NlpService()
        knowledge_svc = KnowledgeService()

        # Stage 2: Sentence 1 with context
        s1 = SubtitleSegment(text="皆さん、こんにちは。")
        toks1 = nlp_svc.tokenize(s1)
        mk1, oov1 = knowledge_svc.match_tokens(toks1)
        trans1, t_meanings1, cands1 = llm_svc.translate_sentence_with_context(
            sentence_text=s1.text,
            context_text=self.sample_text,
            tokens=toks1,
            matched_knowledge=mk1,
            oov_tokens=oov1,
            token_candidates=knowledge_svc.build_token_candidates(toks1),
        )
        self.assertIn("chào", trans1.lower())
        self.assertIn("皆さん", t_meanings1)

        # Stage 2: Sentence 2 with context
        s2 = SubtitleSegment(text="今日のポッドキャストでは、日本の面白い若者言葉について話します。")
        toks2 = nlp_svc.tokenize(s2)
        mk2, oov2 = knowledge_svc.match_tokens(toks2)
        trans2, t_meanings2, cands2 = llm_svc.translate_sentence_with_context(
            sentence_text=s2.text,
            context_text=self.sample_text,
            tokens=toks2,
            matched_knowledge=mk2,
            oov_tokens=oov2,
            token_candidates=knowledge_svc.build_token_candidates(toks2),
        )
        self.assertTrue(len(trans2) > 0)
        self.assertEqual(len(cands2), 2)
        self.assertNotIn("ポッドキャスト", t_meanings2)
        self.assertNotIn("若者言葉", t_meanings2)

    def test_05_local_context_builder(self):
        """Validates bounded local context and context-dependent expansion."""
        llm_svc = LlmEnrichmentService()
        sentences = ["一文目です。", "二文目です。", "三文目です。", "四文目です。", "五文目です。"]

        context = llm_svc.build_sentence_context(sentences, 3, before=3, after=2)
        self.assertIn("[CONTEXT 1] 一文目です。", context)
        self.assertIn("[CONTEXT 3] 三文目です。", context)
        self.assertIn("[TARGET 4] 四文目です。", context)
        self.assertIn("[CONTEXT 5] 五文目です。", context)


    def test_06_translation_chunk_planner_keeps_sentence_boundaries(self):
        """Plans bounded chunks without splitting individual sentences."""
        llm_svc = LlmEnrichmentService()
        sentences = ["A" * 700, "B" * 700, "C" * 700, "D" * 100]
        chunks = llm_svc.build_translation_chunks(sentences, max_chars=1400)
        self.assertEqual(chunks, [(0, 2), (2, 4)])

    def test_06b_translation_chunk_planner_limits_sentence_count(self):
        """Planner must cap sentence count even when character budget is not reached."""
        llm_svc = LlmEnrichmentService()
        sentences = [f"Câu {i}。" for i in range(25)]
        chunks = llm_svc.build_translation_chunks(sentences, max_chars=900)
        self.assertEqual(chunks, [(0, 12), (12, 24), (24, 25)])

    def test_06c_chunk_translation_receives_dictionary_token_candidates(self):
        """Chunk prompt must include server-generated candidates for contextual sense selection."""
        from unittest.mock import patch

        llm_svc = LlmEnrichmentService()
        candidate = {
            "token_id": "token-123",
            "candidate_id": "t1-c1",
            "meaning": "cảm lạnh",
        }
        prompts = []

        def fake_call_llm(prompt, system_prompt=None, max_attempts=None):
            prompts.append(prompt)
            return {
                "sentences": [
                    {
                        "sentence_number": 1,
                        "translation_vi": "Hôm qua tôi bị cảm.",
                        "token_selections": [
                            {"token_key": "t1", "candidate_id": "t1-c1"}
                        ],
                        "oov_learning": [],
                    }
                ]
            }

        with patch.object(llm_svc, "_call_llm", side_effect=fake_call_llm):
            result = llm_svc.translate_chunk([
                {
                    "sentence_number": 1,
                    "text": "昨日は風邪をひきました。",
                    "token_candidates": {"t1": [candidate]},
                }
            ])

        self.assertIn(1, result)
        self.assertEqual(result[1]["translation_vi"], "Hôm qua tôi bị cảm.")
        self.assertEqual(len(prompts), 1)
        self.assertIn('"token_candidates"', prompts[0])
        self.assertIn('"candidate_id": "t1-c1"', prompts[0])
        self.assertIn('"meaning": "cảm lạnh"', prompts[0])
        self.assertIn("phải trả về đúng một token_selection", prompts[0])
        self.assertIn("Dùng gloss_en", prompts[0])
        self.assertIn("contextual_meaning_vi", prompts[0])
        self.assertIn("meaning_vi là tham khảo/fallback", prompts[0])

    def test_06d_chunk_translation_retries_repeated_translation_collapse(self):
        """Repeated output for distinct source sentences is retried individually."""
        import json

        llm_svc = LlmEnrichmentService()
        calls = []

        def fake_call(prompt, system_prompt=None, max_attempts=None):
            payload = json.loads(
                prompt.split("INPUT:\n", 1)[1].split("\n\nTrả về JSON", 1)[0]
            )
            calls.append([item["sentence_number"] for item in payload])
            if len(payload) > 1:
                return {
                    "sentences": [
                        {
                            "sentence_number": item["sentence_number"],
                            "translation_vi": "Tôi đang học tiếng Anh.",
                        }
                        for item in payload
                    ]
                }
            item = payload[0]
            return {
                "sentences": [{
                    "sentence_number": item["sentence_number"],
                    "translation_vi": f"Bản dịch riêng của câu {item['sentence_number']}.",
                }]
            }

        llm_svc._call_llm = fake_call
        result = llm_svc.translate_chunk([
            {"sentence_number": 13, "text": "早く、自然な英語を話せるようになりたい。"},
            {"sentence_number": 14, "text": "今日は自然な日本語の4つのポイントを話します。"},
            {"sentence_number": 15, "text": "全然大丈夫だよ。"},
            {"sentence_number": 16, "text": "間違えるのがはずかしいって思ったり。"},
        ])

        self.assertEqual(sorted(result), [13, 14, 15, 16])
        translations = [result[number]["translation_vi"] for number in sorted(result)]
        self.assertEqual(len(set(translations)), 4)
        self.assertEqual(len(calls), 5)
        self.assertEqual(calls[0], [13, 14, 15, 16])
        self.assertTrue(all(len(call) == 1 for call in calls[1:]))

    def test_06e_chunk_translation_leaves_unresolved_collapse_for_fallback(self):
        """If isolated retries repeat the same collapsed output, do not publish it."""
        import json

        llm_svc = LlmEnrichmentService()
        calls = []

        def fake_call(prompt, system_prompt=None, max_attempts=None):
            payload = json.loads(
                prompt.split("INPUT:\n", 1)[1].split("\n\nTrả về JSON", 1)[0]
            )
            calls.append([item["sentence_number"] for item in payload])
            return {
                "sentences": [
                    {
                        "sentence_number": item["sentence_number"],
                        "translation_vi": "Tôi đang học tiếng Anh.",
                    }
                    for item in payload
                ]
            }

        llm_svc._call_llm = fake_call
        result = llm_svc.translate_chunk([
            {"sentence_number": 21, "text": "自然な日本語について考えます。"},
            {"sentence_number": 22, "text": "今日は四つのポイントを話します。"},
            {"sentence_number": 23, "text": "全然大丈夫ですよ。"},
            {"sentence_number": 24, "text": "間違えるのが恥ずかしいです。"},
        ])

        self.assertEqual(result, {})
        self.assertEqual(len(calls), 5)
        self.assertTrue(all(len(call) == 1 for call in calls[1:]))

    def test_06f_compact_candidates_preserves_senses_and_reduces_payload(self):
        """The chunk prompt keeps candidate IDs/meanings without repeated token metadata."""
        import json

        llm_svc = LlmEnrichmentService()
        items = [{
            "sentence_number": 1,
            "text": "昨日は風邪をひきました。",
            "token_candidates": {
                "t1": [
                    {
                        "candidate_id": "t1-c1",
                        "token_key": "t1",
                        "token_id": "token-123",
                        "token": "風邪",
                        "lemma": "風邪",
                        "term": "風邪",
                        "reading": "かぜ",
                        "pos": "NOUN",
                        "meaning": "cảm lạnh",
                        "gloss_en": "common cold; influenza; flu",
                    },
                    {
                        "candidate_id": "t1-c2",
                        "token_key": "t1",
                        "token_id": "token-123",
                        "token": "風邪",
                        "lemma": "風邪",
                        "term": "風邪",
                        "reading": "かぜ",
                        "pos": "NOUN",
                        "meaning": "lạnh, gió lạnh",
                        "gloss_en": "common cold; influenza; flu",
                    },
                ]
            },
        }]

        items[0]["token_candidates"]["t1"][0]["token_pos"] = "NOUN"
        compact = llm_svc._compact_chunk_prompt_items(items)
        self.assertEqual(compact[0]["token_candidates"]["t1"]["token"], "風邪")
        self.assertEqual(compact[0]["token_candidates"]["t1"]["token_pos"], "NOUN")
        self.assertEqual(compact[0]["token_candidates"]["t1"]["lemma"], "風邪")
        group = compact[0]["token_candidates"]["t1"]
        candidates = group["candidates"]
        self.assertEqual([c["candidate_id"] for c in candidates], ["t1-c1", "t1-c2"])
        self.assertEqual([c["meaning"] for c in candidates], ["cảm lạnh", "lạnh, gió lạnh"])
        self.assertEqual([c["entry_ref"] for c in candidates], ["t1-e1", "t1-e1"])
        self.assertTrue(all(c["term"] == "風邪" for c in candidates))
        self.assertTrue(all(c["reading"] == "かぜ" for c in candidates))
        self.assertTrue(all(c["pos"] == "NOUN" for c in candidates))
        self.assertEqual(candidates[0]["gloss_en"], "common cold; influenza; flu")
        self.assertEqual(candidates[1]["gloss_en"], "", "Shared entry gloss must not be duplicated.")
        self.assertNotIn("token_id", json.dumps(compact, ensure_ascii=False))
        self.assertLess(
            len(json.dumps(compact, ensure_ascii=False)),
            len(json.dumps(items, ensure_ascii=False)),
        )

    def test_06g_long_prompts_use_configurable_larger_context(self):
        """Long chunk prompts get the configured context while short calls keep default."""
        from unittest.mock import patch

        class FakeResponse:
            status_code = 200
            text = ""

            def json(self):
                return {"message": {"content": '{"ok": true}'}}

        captured_options = []

        def fake_post(url, json, timeout):
            captured_options.append(dict(json["options"]))
            return FakeResponse()

        service = LlmEnrichmentService()
        with patch.dict(os.environ, {
            "LLM_NUM_CTX": "4096",
            "LLM_LONG_PROMPT_NUM_CTX": "8192",
            "LLM_LONG_PROMPT_THRESHOLD_CHARS": "8000",
        }), patch("requests.post", side_effect=fake_post):
            self.assertEqual(service._call_llm("short", max_attempts=1), {"ok": True})
            self.assertEqual(service._call_llm("x" * 9001, max_attempts=1), {"ok": True})

        self.assertEqual(captured_options[0]["num_ctx"], 4096)
        self.assertEqual(captured_options[1]["num_ctx"], 8192)

    def test_06j_llm_diagnostics_report_ollama_metrics_without_response_content(self):
        """Opt-in diagnostics include request/model timing but not generated text."""
        import json
        from unittest.mock import patch

        class FakeResponse:
            status_code = 200
            text = "transport body is not needed"

            def json(self):
                return {
                    "message": {"content": json.dumps({
                        "ok": True,
                        "private_translation": "SECRET_TRANSLATION_BODY",
                    })},
                    "total_duration": 1_500_000_000,
                    "load_duration": 200_000_000,
                    "prompt_eval_count": 300,
                    "prompt_eval_duration": 500_000_000,
                    "eval_count": 100,
                    "eval_duration": 800_000_000,
                }

        service = LlmEnrichmentService()
        with (
            patch.dict(os.environ, {"RAKUSHU_LLM_DIAGNOSTICS": "1"}),
            patch("requests.post", return_value=FakeResponse()),
            self.assertLogs("RakushuPipeline", level="INFO") as captured,
        ):
            result = service._call_llm("PRIVATE_SOURCE_SENTENCE", max_attempts=1)

        self.assertTrue(result["ok"])
        logs = "\\n".join(captured.output)
        self.assertIn("http_elapsed_ms=", logs)
        self.assertIn("ollama_total_ms=1500.0", logs)
        self.assertIn("model_load_ms=200.0", logs)
        self.assertIn("prompt_eval_count=300", logs)
        self.assertIn("prompt_eval_ms=500.0", logs)
        self.assertIn("prompt_eval_tokens_per_sec=600.0", logs)
        self.assertIn("eval_count=100", logs)
        self.assertIn("eval_ms=800.0", logs)
        self.assertIn("generation_tokens_per_sec=125.0", logs)
        self.assertNotIn("SECRET_TRANSLATION_BODY", logs)
        self.assertNotIn("PRIVATE_SOURCE_SENTENCE", logs)
        self.assertNotIn("raw_response_excerpt", logs)

    def test_06k_llm_diagnostics_tolerate_missing_ollama_metrics(self):
        """Ollama-compatible responses without optional metrics remain usable."""
        from unittest.mock import patch

        class FakeResponse:
            status_code = 200
            text = ""

            def json(self):
                return {"message": {"content": '{"ok": true}'}}

        service = LlmEnrichmentService()
        with (
            patch.dict(os.environ, {"RAKUSHU_LLM_DIAGNOSTICS": "1"}),
            patch("requests.post", return_value=FakeResponse()),
            self.assertLogs("RakushuPipeline", level="INFO") as captured,
        ):
            result = service._call_llm("short", max_attempts=1)

        self.assertEqual(result, {"ok": True})
        logs = "\\n".join(captured.output)
        self.assertIn("ollama_total_ms=n/a", logs)
        self.assertIn("model_load_ms=n/a", logs)
        self.assertIn("prompt_eval_ms=n/a", logs)
        self.assertIn("eval_ms=n/a", logs)

    def test_06l_llm_diagnostics_measure_failed_attempt_without_logging_body(self):
        """Failed HTTP calls report duration/status without emitting the response body."""
        from unittest.mock import patch

        class FakeResponse:
            def __init__(self, status_code, body, text):
                self.status_code = status_code
                self._body = body
                self.text = text

            def json(self):
                return self._body

        responses = [
            FakeResponse(500, {"error": "server error"}, "SECRET_HTTP_ERROR_BODY"),
            FakeResponse(200, {"message": {"content": '{"ok": true}'}}, ""),
        ]
        service = LlmEnrichmentService()
        with (
            patch.dict(os.environ, {"RAKUSHU_LLM_DIAGNOSTICS": "1"}),
            patch("requests.post", side_effect=responses),
            patch("time.sleep"),
            self.assertLogs("RakushuPipeline", level="INFO") as captured,
        ):
            result = service._call_llm("short", max_attempts=2)

        self.assertEqual(result, {"ok": True})
        logs = "\\n".join(captured.output)
        self.assertIn("outcome=http_error", logs)
        self.assertIn("request_ms=", logs)
        self.assertNotIn("SECRET_HTTP_ERROR_BODY", logs)

    def test_06h_chunk_translation_splits_large_input_before_inference(self):
        """Large chunks are split before the LLM call to bound candidate and output size."""
        from unittest.mock import patch

        llm_svc = LlmEnrichmentService()
        calls = []

        def fake_batch(items, max_attempts=2):
            calls.append([item["sentence_number"] for item in items])
            return {
                item["sentence_number"]: {
                    "sentence_number": item["sentence_number"],
                    "translation_vi": f"Bản dịch riêng {item['sentence_number']}",
                }
                for item in items
            }

        llm_svc._translate_chunk_batch = fake_batch
        sentences = [
            {"sentence_number": index, "text": f"Câu tiếng Nhật số {index}。"}
            for index in range(1, 13)
        ]
        with patch.dict(os.environ, {
            "LLM_CHUNK_MAX_SENTENCES": "4",
            "LLM_CHUNK_PROMPT_PAYLOAD_MAX_CHARS": "7000",
        }):
            result = llm_svc.translate_chunk(sentences)

        self.assertEqual(sorted(result), list(range(1, 13)))
        self.assertEqual([len(call) for call in calls], [4, 4, 4])
        self.assertEqual(calls, [list(range(1, 5)), list(range(5, 9)), list(range(9, 13))])

    def test_06i_chunk_translation_splits_by_candidate_payload_budget(self):
        """Candidate-heavy inputs split even when they contain fewer than max sentences."""
        from unittest.mock import patch

        llm_svc = LlmEnrichmentService()
        calls = []

        def fake_batch(items, max_attempts=2):
            calls.append([item["sentence_number"] for item in items])
            return {
                item["sentence_number"]: {
                    "sentence_number": item["sentence_number"],
                    "translation_vi": f"Bản dịch riêng {item['sentence_number']}",
                }
                for item in items
            }

        llm_svc._translate_chunk_batch = fake_batch
        sentences = [
            {
                "sentence_number": 1,
                "text": "一。",
                "token_candidates": {
                    "t1": [{
                        "candidate_id": "t1-c1",
                        "token": "一",
                        "lemma": "一",
                        "term": "一",
                        "meaning": "nghĩa " + ("a" * 1800),
                    }]
                },
            },
            {
                "sentence_number": 2,
                "text": "二。",
                "token_candidates": {
                    "t1": [{
                        "candidate_id": "t1-c1",
                        "token": "二",
                        "lemma": "二",
                        "term": "二",
                        "meaning": "nghĩa " + ("b" * 1800),
                    }]
                },
            },
        ]
        with patch.dict(os.environ, {
            "LLM_CHUNK_MAX_SENTENCES": "12",
            "LLM_CHUNK_PROMPT_PAYLOAD_MAX_CHARS": "3000",
        }):
            result = llm_svc.translate_chunk(sentences)

        self.assertEqual(sorted(result), [1, 2])
        self.assertEqual(calls, [[1], [2]])

    def test_07_adaptive_retry_changes_llm_options(self):
        """A failed inference retry must not resend the exact same decoding config."""
        from unittest.mock import patch
        class FakeResponse:
            def __init__(self, status_code, body):
                self.status_code = status_code
                self._body = body
                self.text = body if isinstance(body, str) else str(body)

            def json(self):
                return self._body

        calls = []
        responses = [
            FakeResponse(500, {"error": "prediction aborted, token repeat limit reached"}),
            FakeResponse(200, {"message": {"content": '{"translation_vi":"dịch"}'}}),
        ]

        def fake_post(url, json, timeout):
            calls.append(json["options"])
            return responses.pop(0)

        llm_svc = LlmEnrichmentService()
        with patch("requests.post", side_effect=fake_post):
            result = llm_svc._call_llm("dịch câu này")

        self.assertEqual(result["translation_vi"], "dịch")
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0], calls[1])
        self.assertGreater(calls[1]["repeat_penalty"], calls[0]["repeat_penalty"])

    def test_08_chunk_translation_validates_sentence_numbers(self):
        """Chunk translation returns valid items and leaves missing sentences for fallback."""
        llm_svc = LlmEnrichmentService()
        llm_svc._call_llm = lambda prompt, system_prompt=None, max_attempts=None: {
            "sentences": [
                {"sentence_number": 1, "translation_vi": "dịch 1", "token_meanings": {}, "oov_learning": []},
                {"sentence_number": 3, "translation_vi": "dịch 3", "token_meanings": {}, "oov_learning": []},
            ]
        }
        result = llm_svc.translate_chunk([
            {"sentence_number": 1, "text": "一。"},
            {"sentence_number": 2, "text": "二。"},
            {"sentence_number": 3, "text": "三。"},
        ])
        self.assertEqual(sorted(result), [1, 3])
        self.assertEqual(result[1]["translation_vi"], "dịch 1")

    def test_09_adaptive_chunk_splits_after_partial_batch(self):
        """A persistently partial batch is split instead of retrying the same large prompt."""
        llm_svc = LlmEnrichmentService()
        calls = []

        def fake_call(prompt, system_prompt=None, max_attempts=None):
            import json
            calls.append((prompt, max_attempts))
            payload = json.loads(prompt.split("INPUT:\n", 1)[1].split("\n\nTrả về JSON", 1)[0])
            items = payload
            if len(items) > 2:
                return {
                    "sentences": [
                        {"sentence_number": items[0]["sentence_number"], "translation_vi": "dịch"}
                    ]
                }
            return {
                "sentences": [
                    {"sentence_number": item["sentence_number"], "translation_vi": "dịch"}
                    for item in items
                ]
            }

        llm_svc._call_llm = fake_call
        result = llm_svc.translate_chunk([
            {"sentence_number": 1, "text": "一。"},
            {"sentence_number": 2, "text": "二。"},
            {"sentence_number": 3, "text": "三。"},
            {"sentence_number": 4, "text": "四。"},
        ])

        self.assertEqual(sorted(result), [1, 2, 3, 4])
        self.assertTrue(any(len(prompt) > 0 and max_attempts == 2 for prompt, max_attempts in calls))
        self.assertGreater(len(calls), 2)

    def test_10_context_fallback_uses_expanded_window(self):
        """Retries with expanded context only when the first context is insufficient."""
        llm_svc = LlmEnrichmentService()
        prompts = []
        responses = iter([
            {"context_sufficient": False, "context_issue": "referent unclear"},
            {"translation_vi": "dịch", "token_meanings": {}, "oov_learning": [], "context_sufficient": True},
        ])
        def fake_call(prompt, system_prompt=None, max_attempts=None):
            prompts.append(prompt)
            return next(responses)
        llm_svc._call_llm = fake_call
        llm_svc.translate_sentence_with_context(
            sentence_text="Câu 5.",
            context_text="Câu 2.\nCâu 3.\nCâu 4.\nCâu 5.",
            expanded_context_text="Câu 1.\nCâu 2.\nCâu 3.\nCâu 4.\nCâu 5.\nCâu 6.",
        )
        self.assertEqual(len(prompts), 2)
        self.assertIn("Câu 1.", prompts[1])
        self.assertIn("Câu 6.", prompts[1])
    def test_11_bunsetsu_grouping(self):
        """Validates Bunsetsu phrase segmentation rules (Jiritsugo + Fuzokugo)."""
        nlp_svc = NlpService()
        tokens = nlp_svc.tokenize(self.segment)

        bunsetsu_svc = BunsetsuService()
        phrases = bunsetsu_svc.group_bunsetsu(self.segment, tokens)

        self.assertEqual(len(phrases), 8)
        phrase_texts = [p.text for p in phrases]
        expected_chunks = [
            "皆さん、", "こんにちは。", "今日の", "ポッドキャストでは、",
            "日本の", "面白い", "若者言葉について", "話します。"
        ]
        self.assertEqual(phrase_texts, expected_chunks)

        for idx, phrase in enumerate(phrases, 1):
            self.assertEqual(phrase.phrase_order, idx)
            self.assertTrue(len(phrase.translation) > 0)
            self.assertLess(phrase.start_time, phrase.end_time)

    def test_12_sentence_translation_uses_local_context(self):
        """Ensures Stage 2 receives nearby context instead of the transcript prefix."""
        llm_svc = LlmEnrichmentService()
        captured = {}
        llm_svc._call_llm = lambda prompt, system_prompt=None: (captured.update(prompt=prompt) or {
            "translation_vi": "dịch",
            "token_meanings": {},
            "oov_learning": [],
        })
        sentences = [f"Câu {i} về chủ đề riêng." for i in range(1, 8)]
        target = sentences[6]
        context = llm_svc.build_sentence_context(sentences, 6, before=3, after=2)
        llm_svc.translate_sentence_with_context(sentence_text=target, context_text=context)
        self.assertIn("Câu 4 về chủ đề riêng.", captured["prompt"])
        self.assertIn("Câu 7 về chủ đề riêng.", captured["prompt"])
        self.assertNotIn("Câu 1 về chủ đề riêng.", captured["prompt"])
        self.assertIn("Câu cần dịch", captured["prompt"])
    def test_13_end_to_end_pipeline(self):
        """Runs full end-to-end pipeline and checks sentence-level result objects."""
        media_file = "samples/japanese_podcast_10s.mp4"
        if not os.path.exists(media_file):
            create_media = importlib.import_module("samples.create-sample-media")
            create_media.create_sample_media(media_file)

        result = run_pipeline(media_file)

        self.assertIsNotNone(result.segment)
        self.assertEqual(len(result.sentences), 2)
        self.assertEqual(len(result.bunsetsu_phrases), 8)
        self.assertGreaterEqual(len(result.oov_candidates), 2)

        # Validate TokenModel has context_meaning populated
        tokens_with_meanings = [t for t in result.tokens if t.context_meaning]
        self.assertGreater(len(tokens_with_meanings), 3)

        # Validate SentenceSubtitle structures
        s1 = result.sentences[0]
        self.assertEqual(s1.segment.text, "皆さん、こんにちは。")
        self.assertTrue(len(s1.translation) > 0)
        self.assertEqual(len(s1.bunsetsu_phrases), 2)

        s2 = result.sentences[1]
        self.assertIn(s2.segment.text, [
            "今日のポッドキャストでは、日本の面白い若者言葉について話します。",
            "今日のポッドキャストでは、日本の面白い若もの言葉について話します。"
        ])
        self.assertTrue(len(s2.translation) > 0)
        self.assertEqual(len(s2.bunsetsu_phrases), 6)

        # Validate exportable dictionary format
        result_dict = result.to_dict()
        self.assertNotIn("full_translation", result_dict)
        self.assertIn("sentences", result_dict)
        self.assertEqual(len(result_dict["sentences"]), 2)



    def test_14_dictionary_candidate_selection_resolves_server_meaning(self):
        """LLM selects a candidate ID; backend supplies the actual dictionary meaning."""
        llm_svc = LlmEnrichmentService()
        tokens = [SubtitleSegment(text="やすい")]
        token_model = NlpService().tokenize(tokens[0])[0]
        token_model.surface = "やすい"
        token_model.lemma = "やすい"
        token_model.pos = "ADJECTIVE"
        candidates = {
            "t1": [
                {
                    "candidate_id": "t1-c1",
                    "token_key": "t1",
                    "token_id": token_model.token_id,
                    "token": "やすい",
                    "lemma": "やすい",
                    "term": "やすい",
                    "reading": "やすい",
                    "pos": "ADJECTIVE",
                    "meaning": "rẻ, giá thấp",
                },
                {
                    "candidate_id": "t1-c2",
                    "token_key": "t1",
                    "token_id": token_model.token_id,
                    "token": "やすい",
                    "lemma": "やすい",
                    "term": "やすい",
                    "reading": "やすい",
                    "pos": "ADJECTIVE",
                    "meaning": "dễ, dễ dàng",
                },
            ]
        }

        selected = llm_svc.resolve_token_selections(
            [{"token_key": "t1", "token": "やすい", "candidate_id": "t1-c1"}],
            candidates,
            [token_model],
        )
        self.assertEqual(selected[token_model.token_id], "rẻ, giá thấp")
        self.assertEqual(selected["やすい"], "rẻ, giá thấp")

        invalid = llm_svc.resolve_token_selections(
            [{"token_key": "t1", "token": "やすai", "candidate_id": "t1-c1"}],
            candidates,
            [token_model],
        )
        self.assertEqual(invalid, {})

        phrase = llm_svc.resolve_token_selections(
            [{"token": "彼は今、たしか30歳ぐらいで", "candidate_id": "t1-c1"}],
            candidates,
            [token_model],
        )
        self.assertEqual(phrase, {})

    def test_14a_gloss_based_contextual_meaning_overrides_static_vietnamese(self):
        """A validated candidate can yield a contextual Vietnamese sense based on its gloss."""
        llm_svc = LlmEnrichmentService()
        token = NlpService().tokenize(SubtitleSegment(text="思う"))[0]
        candidates = {
            "t1": [{
                "candidate_id": "t1-c1",
                "token_key": "t1",
                "token_id": token.token_id,
                "token": token.surface,
                "lemma": "思う",
                "term": "思う",
                "reading": "おもう",
                "pos": "VERB",
                "meaning": "tưởng tượng",
                "gloss_en": "to think; to consider; to believe; to imagine",
            }]
        }

        resolved = llm_svc.resolve_token_selections(
            [{
                "token_key": "t1",
                "candidate_id": "t1-c1",
                "selected_sense_en": "to think",
                "contextual_meaning_vi": "nghĩ",
            }],
            candidates,
            [token],
        )

        self.assertEqual(resolved[token.token_id], "nghĩ")
        self.assertEqual(resolved[token.surface], "nghĩ")

    def test_14aa_contextual_meaning_falls_back_without_gloss(self):
        """LLM cannot replace the dictionary meaning when there is no English gloss evidence."""
        llm_svc = LlmEnrichmentService()
        token = NlpService().tokenize(SubtitleSegment(text="思う"))[0]
        candidates = {
            "t1": [{
                "candidate_id": "t1-c1",
                "token_key": "t1",
                "token_id": token.token_id,
                "token": token.surface,
                "lemma": "思う",
                "term": "思う",
                "reading": "おもう",
                "pos": "VERB",
                "meaning": "tưởng tượng",
                "gloss_en": "",
            }]
        }

        resolved = llm_svc.resolve_token_selections(
            [{
                "token_key": "t1",
                "candidate_id": "t1-c1",
                "selected_sense_en": "to think",
                "contextual_meaning_vi": "nghĩ",
            }],
            candidates,
            [token],
        )

        self.assertEqual(resolved[token.token_id], "tưởng tượng")

    def test_14b_chunk_selection_is_scoped_to_sentence(self):
        """A selection from another sentence must be rejected even when token keys collide."""
        llm_svc = LlmEnrichmentService()
        token = NlpService().tokenize(SubtitleSegment(text="今日"))[0]
        candidates = {
            "t1": [
                {
                    "candidate_id": "t1-c1",
                    "token_key": "t1",
                    "token_id": token.token_id,
                    "token": "今日",
                    "lemma": "今日",
                    "term": "今日",
                    "reading": "きょう",
                    "pos": "NOUN",
                    "meaning": "hôm nay",
                },
                {
                    "candidate_id": "t1-c2",
                    "token_key": "t1",
                    "token_id": token.token_id,
                    "token": "今日",
                    "lemma": "今日",
                    "term": "今日",
                    "reading": "きょう",
                    "pos": "NOUN",
                    "meaning": "ngày hôm nay",
                },
            ]
        }
        result = llm_svc.resolve_token_selections(
            [{"sentence_number": 1, "token_key": "t1", "candidate_id": "t1-c1"}],
            candidates,
            [token],
            sentence_number=2,
        )
        self.assertEqual(result, {})

    def test_15_single_dictionary_candidate_does_not_require_llm_selection(self):
        """A deterministic single sense can be resolved without an LLM-generated meaning."""
        llm_svc = LlmEnrichmentService()
        token = NlpService().tokenize(SubtitleSegment(text="日本"))[0]
        candidates = {
            "t1": [{
                "candidate_id": "t1-c1",
                "token_key": "t1",
                "token_id": token.token_id,
                "token": "日本",
                "lemma": "日本",
                "term": "日本",
                "reading": "にほん",
                "pos": "NOUN",
                "meaning": "Nhật Bản",
            }]
        }
        result = llm_svc.resolve_token_selections([], candidates, [token])
        self.assertEqual(result[token.token_id], "Nhật Bản")
        self.assertEqual(result["日本"], "Nhật Bản")

    def test_17_resolver_rejection_reasons_are_explicit(self):
        """Every invalid selection shape is rejected with a specific audit reason."""
        llm_svc = LlmEnrichmentService()
        token = NlpService().tokenize(SubtitleSegment(text="今日"))[0]
        valid = {
            "t1": [{
                "candidate_id": "t1-c1",
                "token_key": "t1",
                "token_id": token.token_id,
                "token": "今日",
                "lemma": "今日",
                "term": "今日",
                "reading": "きょう",
                "pos": "NOUN",
                "meaning": "hôm nay",
            }]
        }

        cases = [
            ({"token_key": "t9", "candidate_id": "t9-c1"}, "unknown_token_key"),
            ({"token_key": "t1", "token": "明日", "candidate_id": "t1-c1"}, "token_surface_mismatch"),
            ({"token_key": "t1", "candidate_id": "t1-c9"}, "unknown_candidate"),
        ]
        # Keep two valid candidates so an invalid selection cannot be masked by
        # deterministic single-candidate auto-resolution.
        valid["t1"].append(dict(valid["t1"][0], candidate_id="t1-c2", meaning="ngày hôm nay"))

        for selection, reason in cases:
            with self.subTest(reason=reason), self.assertLogs("RakushuPipeline", level="WARNING") as logs:
                result = llm_svc.resolve_token_selections([selection], valid, [token])
            self.assertEqual(result, {})
            self.assertTrue(any(f"reason={reason}" in line for line in logs.output))

        mismatched_id = dict(valid["t1"][0])
        mismatched_id["token_id"] = "other-token-id"
        with self.assertLogs("RakushuPipeline", level="WARNING") as logs:
            result = llm_svc.resolve_token_selections(
                [{"token_key": "t1", "candidate_id": "t1-c1"}],
                {"t1": [mismatched_id]},
                [token],
            )
        self.assertEqual(result, {})
        self.assertTrue(any("reason=candidate_token_id_mismatch" in line for line in logs.output))

        missing_meaning = dict(valid["t1"][0])
        missing_meaning["meaning"] = ""
        with self.assertLogs("RakushuPipeline", level="WARNING") as logs:
            result = llm_svc.resolve_token_selections(
                [{"token_key": "t1", "candidate_id": "t1-c1"}],
                {"t1": [missing_meaning, valid["t1"][1]]},
                [token],
            )
        self.assertEqual(result, {})
        self.assertTrue(any("reason=missing_meaning" in line for line in logs.output))

        with self.assertLogs("RakushuPipeline", level="WARNING") as logs:
            result = llm_svc.resolve_token_selections(
                [{"sentence_number": "bad", "token_key": "t1", "candidate_id": "t1-c1"}],
                valid,
                [token],
                sentence_number=1,
            )
        self.assertEqual(result, {})
        self.assertTrue(any("reason=invalid_sentence_number" in line for line in logs.output))

    def test_18_hierarchical_matching_prefers_longest_span_and_priority(self):
        """Phrase/grammar/compound matching is server-owned and longest-first."""
        from types import SimpleNamespace
        from src.services.hierarchical_matcher import HierarchicalKnowledgeMatcher

        class FakeKnowledge:
            def lookup_grammar(self, term):
                if term == "ABC":
                    return SimpleNamespace(reading="grammar", meaning="grammar")
                return None

            def lookup_phrase(self, term):
                if term in {"ABC", "AB"}:
                    return SimpleNamespace(reading="phrase", meaning=f"phrase:{term}")
                return None

            def lookup_compound_word(self, term):
                if term == "BC":
                    return SimpleNamespace(reading="compound", meaning="compound:BC")
                return None

            def lookup(self, term):
                return None

            def is_rejected_oov(self, term):
                return False

        tokens = [
            SimpleNamespace(surface="A", lemma="A", reading="A", pos="NOUN"),
            SimpleNamespace(surface="B", lemma="B", reading="B", pos="NOUN"),
            SimpleNamespace(surface="C", lemma="C", reading="C", pos="NOUN"),
        ]
        matcher = HierarchicalKnowledgeMatcher(FakeKnowledge())
        matched, oov = matcher.match(tokens, "ABC")

        self.assertEqual([(u.unit_type, u.surface, u.start_token_idx, u.end_token_idx) for u in matched], [
            ("GRAMMAR", "ABC", 0, 2),
        ])
        self.assertEqual(oov, [])

    def test_19_hierarchical_matching_uses_compound_then_word_fallback(self):
        """When no grammar/phrase span exists, compound and word layers remain usable."""
        from types import SimpleNamespace
        from src.services.hierarchical_matcher import HierarchicalKnowledgeMatcher

        class FakeKnowledge:
            def lookup_grammar(self, term):
                return None

            def lookup_phrase(self, term):
                return None

            def lookup_compound_word(self, term):
                if term == "BC":
                    return SimpleNamespace(reading="BC", meaning="compound meaning")
                return None

            def lookup(self, term):
                if term == "A":
                    return DictionaryEntry(term="A", reading="A", meaning="word meaning")
                return None

            def is_rejected_oov(self, term):
                return False

        tokens = [
            SimpleNamespace(surface="A", lemma="A", reading="A", pos="NOUN"),
            SimpleNamespace(surface="B", lemma="B", reading="B", pos="NOUN"),
            SimpleNamespace(surface="C", lemma="C", reading="C", pos="NOUN"),
        ]
        matched, oov = HierarchicalKnowledgeMatcher(FakeKnowledge()).match(tokens, "ABC")
        self.assertEqual([(u.unit_type, u.surface) for u in matched], [
            ("WORD", "A"),
            ("COMPOUND_WORD", "BC"),
        ])
        self.assertEqual([(u.start_token_idx, u.end_token_idx) for u in matched], [(0, 0), (1, 2)])
        self.assertEqual(oov, [])

    def test_16_selected_meaning_propagates_to_rules_bunsetsu(self):
        """A resolved dictionary sense must survive the Bunsetsu rules fallback."""
        llm_svc = LlmEnrichmentService()
        nlp_svc = NlpService()
        tokens = nlp_svc.tokenize(SubtitleSegment(text="安いです。"))
        candidates = {
            "t1": [
                {
                    "candidate_id": "t1-c1",
                    "token_key": "t1",
                    "token_id": tokens[0].token_id,
                    "token": "安い",
                    "lemma": "安い",
                    "term": "安い",
                    "reading": "やすい",
                    "pos": "ADJECTIVE",
                    "meaning": "rẻ, giá thấp",
                },
                {
                    "candidate_id": "t1-c2",
                    "token_key": "t1",
                    "token_id": tokens[0].token_id,
                    "token": "安い",
                    "lemma": "安い",
                    "term": "安い",
                    "reading": "やすい",
                    "pos": "ADJECTIVE",
                    "meaning": "dễ, dễ dàng",
                },
            ]
        }

        selected = llm_svc.resolve_token_selections(
            [{"token_key": "t1", "token": "安い", "candidate_id": "t1-c1"}],
            candidates,
            tokens,
        )
        self.assertEqual(selected[tokens[0].token_id], "rẻ, giá thấp")
        self.assertEqual(selected["安い"], "rẻ, giá thấp")

        bunsetsu_svc = BunsetsuService()
        bunsetsu_svc._nlp = None
        segment = SubtitleSegment(text="安いです。", start_time=0.0, end_time=1.0)
        phrases = bunsetsu_svc.group_bunsetsu(segment, tokens, selected)

        self.assertTrue(any("rẻ, giá thấp" in phrase.translation for phrase in phrases))
        self.assertFalse(any("安い" == phrase.translation for phrase in phrases))

if __name__ == "__main__":
    unittest.main()
