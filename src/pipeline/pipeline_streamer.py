"""Streaming Pipeline Runner: Yields real-time SSE events as sentences are processed."""
import os
import sys
import time
import logging
import importlib
from typing import Generator

from src.utils import create_pipeline_log, close_pipeline_log

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models import SubtitleSegment, DictionaryEntry, MediaSourceType
from src.services import (
    AsrService, NlpService, KnowledgeService, LlmEnrichmentService,
    BunsetsuService, YouTubeService, OovService,
    InvalidMediaError, InvalidLanguageError, ProhibitedContentError,
)
_stream_utils = importlib.import_module(".stream_utils", package="src.utils")
format_sse, execute_with_heartbeat = _stream_utils.format_sse, _stream_utils.execute_with_heartbeat
build_sentence_payload, log_sentence_breakdown = _stream_utils.build_sentence_payload, _stream_utils.log_sentence_breakdown
push_oovs_to_backend = _stream_utils.push_oovs_to_backend
from src.services.pipeline_output_store import pipeline_output_store

logger = logging.getLogger("PipelineStreamer")


def stream_pipeline(
    media_path: str,
    video_id: str = "",
    source_filename: str = "",
) -> Generator[str, None, None]:
    """Processes media and yields SSE events progressively at each stage and sentence."""
    t_start = time.time()
    t_yt, t_asr, t_seg, t_sentences = 0.0, 0.0, 0.0, 0.0
    log_path, log_handler = create_pipeline_log("stream_pipeline")
    logger.info(f"[RUN LOG] Full pipeline log: {os.path.abspath(log_path)}")
    logger.info(f"\n{'='*75}\n[STREAM PIPELINE] Starting for media: '{media_path}'\n{'='*75}")
    yield format_sse("pipeline_started", {"media_path": media_path, "timestamp": time.time()})

    try:
        # Step 0: Source Detection & Download
        youtube_svc = YouTubeService()
        video_meta, media_file_path = {}, media_path
        is_yt = youtube_svc.is_youtube_url(media_path)
        source_type = MediaSourceType.YOUTUBE_URL if is_yt else MediaSourceType.LOCAL_UPLOAD

        # Manual uploads receive their identity at the API boundary so the same
        # video_id can be propagated through every pipeline stage and SSE event.
        if not is_yt and video_id:
            logger.info(f">>> [Step 0: Upload] Assigned video_id: '{video_id}'")
            yield format_sse("upload_ready", {
                "video_id": video_id,
                "source_type": source_type,
                "filename": source_filename,
            })

        if is_yt:
            t0_yt = time.time()
            youtube_video_id = youtube_svc.extract_video_id(media_path) or "unknown"
            logger.info(
                f">>> [Step 0: YouTube] Video ID: '{youtube_video_id}' | "
                f"Detected URL: '{media_path}'. Downloading..."
            )
            yield format_sse("progress", {"step": "youtube_download", "message": f"Downloading YouTube: {media_path}"})
            try:
                media_file_path, video_meta = youtube_svc.download_video(media_path)
                t_yt = round(time.time() - t0_yt, 2)
                youtube_video_id = video_meta.get("video_id", "")
                logger.info(
                    f"    [Step 0 Done] Video ID: '{youtube_video_id}' | "
                    f"Title: \"{video_meta.get('title', '')}\" | "
                    f"Saved: {media_file_path} | Time: {t_yt}s"
                )
                yield format_sse("youtube_ready", {"title": video_meta.get("title", ""), "video_id": youtube_video_id})
            except Exception as e:
                logger.error(f"    [Step 0 Failed] YouTube download error: {e}")
                yield format_sse("error", {"error_type": "DownloadError", "message": str(e)})
                return

        # Step 1: ASR Transcription with Heartbeat Keep-Alive
        t0_asr = time.time()
        logger.info(">>> [Step 1: ASR] Transcribing speech via Whisper ASR...")
        yield format_sse("progress", {"step": "asr", "message": "Transcribing speech via Whisper ASR..."})
        asr_svc = AsrService()
        title = video_meta.get("title", "")
        try:
            full_segment = yield from execute_with_heartbeat(
                asr_svc.transcribe, media_file_path, title=title, source_type=source_type
            )
        except (InvalidLanguageError, ProhibitedContentError, InvalidMediaError) as err:
            logger.error(f"    [Step 1 Validation Error]: {err}")
            yield format_sse("error", {"error_type": "ValidationError", "message": str(err)})
            return
        except Exception as exc:
            logger.error(f"    [Step 1 ASR Error]: {exc}")
            yield format_sse("error", {"error_type": "AsrError", "message": str(exc)})
            return

        if is_yt and video_meta:
            full_segment.video_id = video_meta.get("video_id", full_segment.video_id)
            full_segment.video_title = video_meta.get("title", "")
            full_segment.source_url = media_path
        elif video_id:
            full_segment.video_id = video_id
            full_segment.video_title = source_filename
            full_segment.source_url = media_path

        duration = round(max(full_segment.end_time - full_segment.start_time, 0.0), 2)
        t_asr = round(time.time() - t0_asr, 2)
        logger.info(f"    [Step 1 Done] Transcribed {duration}s in {t_asr}s | Text: \"{full_segment.text}\"")
        yield format_sse("asr_completed", {"text": full_segment.text, "duration": duration, "video_title": full_segment.video_title})

        if not full_segment.text.strip():
            logger.info("    [Step 1] No speech detected in audio.")
            yield format_sse("pipeline_completed", {"message": "No speech detected", "total_sentences": 0})
            return

        llm_svc = LlmEnrichmentService()

        # Step 3: Sentence Segmentation
        t0_seg = time.time()
        nlp_svc, knowledge_svc = NlpService(), KnowledgeService()
        oov_svc, bunsetsu_svc = OovService(knowledge_service=knowledge_svc), BunsetsuService()
        try:
            sentence_texts = nlp_svc.split_sentences(full_segment.text)
        except Exception as exc:
            logger.error(f"Sentence splitting error: {exc}")
            sentence_texts = [full_segment.text] if full_segment.text else []
        total_sentences = len(sentence_texts)
        t_seg = round(time.time() - t0_seg, 2)
        logger.info(f">>> [Step 3: Segmentation] Identified {total_sentences} sentences in {t_seg}s.")
        yield format_sse("sentences_identified", {"total_sentences": total_sentences})

        # Precompute dictionary candidates before chunk translation so the LLM can only select known senses.
        sentence_tokens = {}
        sentence_token_candidates = {}
        sentence_nlp_output = {}
        for sentence_number, sentence_text in enumerate(sentence_texts, start=1):
            candidate_segment = SubtitleSegment(
                text=sentence_text, sequence_number=sentence_number
            )
            nlp_output = nlp_svc.analyze_json(candidate_segment)
            candidate_tokens = nlp_svc.tokenize(candidate_segment)
            sentence_nlp_output[sentence_number] = {
                "sentence_number": sentence_number,
                "text": sentence_text,
                "nlp": nlp_output,
            }
            sentence_tokens[sentence_number] = candidate_tokens
            sentence_token_candidates[sentence_number] = knowledge_svc.build_token_candidates(
                candidate_tokens
            )

        total_chars = max(len(full_segment.text), 1)
        total_dur = full_segment.end_time - full_segment.start_time
        curr_char = 0
        total_grammar, total_phrases, total_compounds = 0, 0, 0
        all_oovs = []
        translation_failures = []
        chunk_ranges = llm_svc.build_translation_chunks(sentence_texts)
        chunk_map = {}
        for chunk_no, (chunk_start, chunk_end) in enumerate(chunk_ranges, start=1):
            for sentence_index in range(chunk_start, chunk_end):
                chunk_map[sentence_index] = chunk_no
        logger.info(f"    [Step 4 Planner] Created {len(chunk_ranges)} LLM translation chunks.")

        t0_sentences = time.time()
        chunk_results_by_sentence = {}
        chunk_batch_meta = {}
        for chunk_no, (chunk_start, chunk_end) in enumerate(chunk_ranges, start=1):
            chunk_t0 = time.time()
            chunk_input = []
            for i in range(chunk_start, chunk_end):
                chunk_input.append({
                    "sentence_number": i + 1,
                    "text": sentence_texts[i],
                    "token_candidates": sentence_token_candidates.get(i + 1, {}),
                })
            try:
                chunk_results = yield from execute_with_heartbeat(
                    llm_svc.translate_chunk, chunk_input
                )
            except Exception as exc:
                logger.warning(f"    [Chunk #{chunk_no}] Batch translation error: {exc}")
                chunk_results = {}
            expected_count = chunk_end - chunk_start
            returned_count = len(chunk_results)
            elapsed = round(time.time() - chunk_t0, 2)
            chunk_batch_meta[chunk_no] = {
                "expected": expected_count, "returned": returned_count, "elapsed": elapsed,
            }
            if chunk_results:
                chunk_results_by_sentence.update(chunk_results)
                logger.info(
                    f"    [Chunk #{chunk_no} Batch] {returned_count}/{expected_count} results in {elapsed}s."
                )
                for sentence_no in range(chunk_start + 1, chunk_end + 1):
                    item = chunk_results.get(sentence_no)
                    if item:
                        logger.info(f"       [#{sentence_no}] JP: {sentence_texts[sentence_no - 1]}")
                        logger.info(f"       [#{sentence_no}] VI: {str(item.get('translation_vi', '')).strip()}")
                missing = [
                    sentence_no for sentence_no in range(chunk_start + 1, chunk_end + 1)
                    if sentence_no not in chunk_results
                ]
                if missing:
                    logger.warning(f"       [Chunk #{chunk_no} Batch Missing] {missing}")
            else:
                logger.warning(
                    f"    [Chunk #{chunk_no} Batch] FAILED after {elapsed}s; "
                    f"sentence-level fallback will be used."
                )

        # Step 4: Stream Sentences Progressively
        chunk_sentence_outputs = {}
        for idx, sent_text in enumerate(sentence_texts, start=1):
            t0_sent = time.time()
            sent_len = len(sent_text)
            s_start = round(full_segment.start_time + (curr_char / total_chars) * total_dur, 2)
            curr_char += sent_len
            s_end = round(full_segment.start_time + (curr_char / total_chars) * total_dur, 2)

            sent_seg = SubtitleSegment(
                segment_id=f"{full_segment.segment_id}-s{idx}", video_id=full_segment.video_id,
                start_time=s_start, end_time=s_end, text=sent_text, sequence_number=idx,
            )

            try:
                chunk_no = chunk_map.get(idx - 1, 1)
                logger.info(f"    [Chunk #{chunk_no}] Processing sentence #{idx}/{total_sentences}.")
                tokens = sentence_tokens.get(idx, [])
                nlp_output_entry = sentence_nlp_output.get(idx)
                if nlp_output_entry:
                    nlp_output_entry["segment_id"] = sent_seg.segment_id
                    nlp_output_entry["start_time"] = s_start
                    nlp_output_entry["end_time"] = s_end
                for token in tokens:
                    token.segment_id = sent_seg.segment_id
                token_candidates = sentence_token_candidates.get(idx, {})
                hier_units, oov_toks = knowledge_svc.match_hierarchical(tokens, sent_text)
                matched_k = [DictionaryEntry(term=u.surface, reading=u.reading, pos=u.unit_type, meaning=u.meaning) for u in hier_units]

                chunk_item = chunk_results_by_sentence.get(idx)
                if chunk_item:
                    s_trans_vi = str(chunk_item.get("translation_vi", "")).strip()
                    raw_token_selections = chunk_item.get("token_selections", [])
                    if raw_token_selections:
                        logger.info(
                            f"    [Token Candidate Trace] Sentence #{idx} | candidates={token_candidates}"
                        )
                    token_meanings = llm_svc.resolve_token_selections(
                        raw_token_selections, token_candidates, tokens, sentence_number=idx
                    )
                    if raw_token_selections:
                        logger.info(
                            f"    [Token Meaning Trace] Sentence #{idx} | "
                            f"selected={raw_token_selections} | resolved={token_meanings}"
                        )
                    enriched_oovs = llm_svc._parse_oov_candidates(
                        chunk_item.get("oov_learning", []), oov_toks, sent_text
                    )
                else:
                    context_text = llm_svc.build_sentence_context(
                        sentence_texts, idx - 1, before=3, after=2
                    )
                    expanded_context = llm_svc.build_sentence_context(
                        sentence_texts, idx - 1, before=10, after=3
                    )
                    s_trans_vi, token_meanings, enriched_oovs = yield from execute_with_heartbeat(
                        llm_svc.translate_sentence_with_context,
                        sentence_text=sent_text, context_text=context_text,
                        expanded_context_text=expanded_context,
                        tokens=tokens, matched_knowledge=matched_k, oov_tokens=oov_toks,
                        token_candidates=token_candidates,
                    )
                    if llm_svc.last_sentence_translation_failed:
                        translation_failures.append({
                            "sentence_number": idx,
                            "chunk_number": chunk_no,
                            "text": sent_text,
                        })
                for tok in tokens:
                    # Prefer the exact occurrence before the potentially ambiguous surface alias.
                    if tok.token_id in token_meanings:
                        tok.context_meaning = token_meanings[tok.token_id]
                    elif tok.surface in token_meanings:
                        tok.context_meaning = token_meanings[tok.surface]

                logger.info(
                    f"    [Meaning Propagation Trace] Sentence #{idx} | "
                    f"resolved={token_meanings} | "
                    f"token_context={[(t.surface, t.context_meaning) for t in tokens if t.context_meaning]}"
                )
                phrases = bunsetsu_svc.group_bunsetsu(sent_seg, tokens, token_meanings)
                for u in hier_units:
                    if u.unit_type == "GRAMMAR": total_grammar += 1
                    elif u.unit_type == "PHRASE": total_phrases += 1
                    elif u.unit_type == "COMPOUND_WORD": total_compounds += 1
                for o in enriched_oovs:
                    oov_svc.persist_candidate(o)
                all_oovs.extend(enriched_oovs)

                t_sent = round(time.time() - t0_sent, 2)
                log_sentence_breakdown(logger, idx, total_sentences, s_start, s_end, t_sent, sent_text, s_trans_vi, tokens, phrases, enriched_oovs)
                payload = build_sentence_payload(sent_seg, idx, total_sentences, s_start, s_end, sent_text, s_trans_vi, phrases, tokens, hier_units, enriched_oovs)
            except Exception as sent_err:
                logger.error(f"Error enriching sentence {idx}: {sent_err}", exc_info=True)
                payload = build_sentence_payload(sent_seg, idx, total_sentences, s_start, s_end, sent_text, "", [], [], [], [])



            yield format_sse("sentence_processed", payload)

            chunk_sentence_outputs[idx] = payload.get("translation", "")
            if idx in {end for _, end in chunk_ranges}:
                chunk_no = chunk_map.get(idx - 1, 1)
                chunk_start = next(start for start, end in chunk_ranges if end == idx)
                chunk_failed = [
                    f["sentence_number"] for f in translation_failures
                    if chunk_start + 1 <= f["sentence_number"] <= idx
                ]
                batch_meta = chunk_batch_meta.get(chunk_no, {})
                status = "PARTIAL" if chunk_failed else "SUCCESS"
                logger.info(
                    f"\n{'='*70}\n"
                    f"[CHUNK #{chunk_no} COMPLETED] "
                    f"Sentences #{chunk_start + 1}-#{idx} | Status: {status} | "
                    f"Batch: {batch_meta.get('returned', 0)}/{batch_meta.get('expected', idx - chunk_start)} "
                    f"in {batch_meta.get('elapsed', 0)}s\n"
                    f"{'='*70}"
                )
                for sentence_no in range(chunk_start + 1, idx + 1):
                    logger.info(f"  [#{sentence_no}] JP: {sentence_texts[sentence_no - 1]}")
                    logger.info(f"  [#{sentence_no}] VI: {chunk_sentence_outputs.get(sentence_no, '')}")
                if chunk_failed:
                    logger.warning(f"  [Chunk #{chunk_no} Fallback Failed] {chunk_failed}")
                logger.info(f"{'='*70}")
                yield format_sse("chunk_completed", {
                    "chunk_number": chunk_no,
                    "start_sentence": chunk_start + 1,
                    "end_sentence": idx,
                    "status": status,
                    "batch_expected": batch_meta.get("expected", idx - chunk_start),
                    "batch_returned": batch_meta.get("returned", 0),
                    "batch_elapsed": batch_meta.get("elapsed", 0),
                    "fallback_failures": chunk_failed,
                    "sentences": [
                        {
                            "sentence_number": sentence_no,
                            "text": sentence_texts[sentence_no - 1],
                            "translation": chunk_sentence_outputs.get(sentence_no, ""),
                        }
                        for sentence_no in range(chunk_start + 1, idx + 1)
                    ],
                })

        # Step 5: Final Summary
        t_sentences = round(time.time() - t0_sentences, 2)
        total_time = round(time.time() - t_start, 2)
        avg_sent = round(t_sentences / max(total_sentences, 1), 2)
        if translation_failures:
            logger.warning(
                f"    [Step 4 Partial] Failed sentence translations: "
                f"{[f['sentence_number'] for f in translation_failures]}"
            )
        logger.info(f"[RUN LOG] Full pipeline log saved at: {os.path.abspath(log_path)}")
        logger.info(
            f"\n{'='*75}\n[PIPELINE COMPLETED] Summary Timing Breakdown:\n"
            f" - Step 0 (YouTube Download):  {t_yt}s\n"
            f" - Step 1 (Whisper ASR):       {t_asr}s (Audio Duration: {duration}s)\n"
            f" - Step 3 (Segmentation):      {t_seg}s ({total_sentences} sentences)\n"
            f" - Step 4 (Sentence Enrich):   {t_sentences}s (avg: {avg_sent}s/câu)\n"
            f" >>> TOTAL TIME ELAPSED:       {total_time}s\n{'='*75}\n"
        )
        if all_oovs:
            push_oovs_to_backend(all_oovs, logger)
        pipeline_status = "PARTIAL_SUCCESS" if translation_failures else "COMPLETED"
        pipeline_output_store.save(
            full_segment.video_id,
            [sentence_nlp_output[number] for number in sorted(sentence_nlp_output)],
        )
        summary_payload = {
            "video_id": full_segment.video_id, "video_title": full_segment.video_title,
            "duration": duration, "total_processing_time": total_time,
            "status": pipeline_status,
            "summary": {
                "total_sentences": total_sentences, "total_grammar_matched": total_grammar,
                "total_phrases_matched": total_phrases, "total_compound_words_matched": total_compounds,
                "total_oov_discovered": len(all_oovs),
                "translation_failures": translation_failures,
            }
        }
        yield format_sse("pipeline_completed", summary_payload)
    except Exception as fatal_exc:
        logger.error(f"Fatal streaming pipeline exception: {fatal_exc}", exc_info=True)
        yield format_sse("error", {"error_type": "PipelineFatalError", "message": str(fatal_exc)})
    finally:
        logger.info(f"[RUN LOG] Pipeline log finalized: {os.path.abspath(log_path)}")
        close_pipeline_log(log_handler)