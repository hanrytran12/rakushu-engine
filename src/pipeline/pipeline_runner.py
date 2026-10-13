import os
import sys
import time
import logging

from src.utils import create_pipeline_log, close_pipeline_log

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.models import SubtitleSegment, SentenceSubtitle, PipelineResult, DictionaryEntry, MediaSourceType
from src.services import (
    AsrService, NlpService, KnowledgeService, LlmEnrichmentService,
    BunsetsuService, YouTubeService, OovService,
    InvalidMediaError, InvalidLanguageError, ProhibitedContentError,
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RakushuPipeline")

def _print_pipeline_summary(result: PipelineResult):
    """Prints a structured summary of translation, tokens, bunsetsu, and OOVs."""
    print(f"\n{'=' * 80}\n [OUTCOME] TWO-STAGE TRANSLATION & TOKEN MEANING BREAKDOWN:\n{'=' * 80}")
    print(f'Transcript: "{result.segment.text}"')
    for idx, s in enumerate(result.sentences, start=1):
        print(f"\n[Sentence #{idx}] ({s.segment.start_time:.1f}s -> {s.segment.end_time:.1f}s)")
        print(f"  JP: {s.segment.text}\n  VI: {s.translation}")
        print("  Token-level contextual meanings:")
        for t in s.tokens:
            if t.context_meaning:
                print(f"    - {t.surface} ({t.pos}): {t.context_meaning}")
        print("  Bunsetsu interactive chunks:")
        for bp in s.bunsetsu_phrases:
            print(f"    * [{bp.text}] ({bp.start_time:.1f}s-{bp.end_time:.1f}s) -> {bp.translation}")

    print("\n" + "=" * 80)
    if result.oov_candidates:
        print(f" [DISCOVERED OOV TERMS] Phát hiện {len(result.oov_candidates)} từ mới ngoài từ điển:")
        print("=" * 80)
        seen_oov = set()
        for o in result.oov_candidates:
            if o.term not in seen_oov:
                print(f"  * {o.term} ({o.suggested_pos}): {o.suggested_meaning} [Score: {o.confidence_score*100:.0f}%]")
                seen_oov.add(o.term)
    else:
        print(" [DISCOVERED OOV TERMS] 100% từ vựng đều có sẵn trong kho tri thức.\n" + "=" * 80)

    if result.segment.source_url:
        print(f"\nSource URL: {result.segment.source_url}\nVideo Title: {result.segment.video_title}")
    if result.segment.video_path:
        print(f"Video Path: {result.segment.video_path}")

def run_pipeline(media_path: str = "samples/japanese_podcast_10s.mp4") -> PipelineResult:
    total_start_time = time.time()
    log_path, log_handler = create_pipeline_log("pipeline")
    logger.info(f"[RUN LOG] Full pipeline log: {os.path.abspath(log_path)}")
    print("\n" + "=" * 80)
    print(" [RAKUSHU AI CORE] ASR -> GINZA NLP -> SENTENCE TRANSLATION PIPELINE\n" + "=" * 80)

    # 0. Source-based handling: Detect YouTube URL vs Local Upload
    youtube_svc = YouTubeService()
    video_meta = {}
    media_file_path = media_path
    is_yt = youtube_svc.is_youtube_url(media_path)
    source_type = MediaSourceType.YOUTUBE_URL if is_yt else MediaSourceType.LOCAL_UPLOAD

    if is_yt:
        logger.info(f">>> DETECTED YOUTUBE URL: {media_path}")
        logger.info(">>> Downloading YouTube video (.mp4 format)...")
        yt_t0 = time.time()
        media_file_path, video_meta = youtube_svc.download_video(media_path)
        logger.info(f"   [YouTube Video Download Completed] In: {time.time() - yt_t0:.2f}s")
        logger.info(f"   [Saved MP4 Path]: {media_file_path}")
        logger.info(f"   [Video Title]: \"{video_meta.get('title', '')}\"")

    # 1. ASR Transcription via Whisper with native validation
    t0 = time.time()
    logger.info(">>> STEP 1: Processing video via Whisper ASR...")
    asr_svc = AsrService()
    title = video_meta.get("title", "")
    try:
        full_segment = asr_svc.transcribe(media_file_path, title=title, source_type=source_type)
    except (InvalidLanguageError, ProhibitedContentError, InvalidMediaError) as err:
        logger.error(f"\n{'='*80}\n [REJECTED] MEDIA INPUT VALIDATION FAILED:\n - Reason: {err}\n - Action: Pipeline aborted early.\n{'='*80}")
        close_pipeline_log(log_handler)
        return PipelineResult(segment=SubtitleSegment(text=f"[REJECTED] {err}"))
    if video_meta:
        full_segment.video_id = video_meta.get("video_id", full_segment.video_id)
        full_segment.video_title = video_meta.get("title", "")
        full_segment.source_url = media_path
        full_segment.video_path = media_file_path
    elif os.path.exists(media_path):
        full_segment.video_path = media_path
    logger.info(f"   [ASR Result] Duration: {full_segment.start_time:.1f}s -> {full_segment.end_time:.1f}s")
    logger.info(f"   [Transcription]: \"{full_segment.text}\"")
    logger.info(f"   [STEP 1 COMPLETED] Time: {time.time() - t0:.2f}s")

    # Initialize Services
    nlp_svc, knowledge_svc = NlpService(), KnowledgeService()
    llm_svc, bunsetsu_svc = LlmEnrichmentService(), BunsetsuService()
    oov_svc = OovService(knowledge_service=knowledge_svc)

    if not full_segment.text.strip():
        logger.warning("   [ASR Notice] No speech detected in media. Returning empty result.")
        close_pipeline_log(log_handler)
        return PipelineResult(segment=full_segment, tokens=[], matched_knowledge=[], oov_candidates=[], bunsetsu_phrases=[], sentences=[])

    # Chunk-level translation is the canonical translation source.
    # 3. Split Transcription into Sentences via GiNZA
    t2 = time.time()
    logger.info("\n>>> STEP 3: Splitting transcription into sentences via GiNZA...")
    sentence_texts = nlp_svc.split_sentences(full_segment.text)
    logger.info(f"   Identified {len(sentence_texts)} discrete sentences. Time: {time.time() - t2:.2f}s")

    # Precompute dictionary candidates before chunk translation so the LLM can only select known senses.
    sentence_tokens = {}
    sentence_token_candidates = {}
    for sentence_number, sentence_text in enumerate(sentence_texts, start=1):
        candidate_segment = SubtitleSegment(
            text=sentence_text, sequence_number=sentence_number
        )
        candidate_tokens = nlp_svc.tokenize(candidate_segment)
        sentence_tokens[sentence_number] = candidate_tokens
        sentence_token_candidates[sentence_number] = knowledge_svc.build_token_candidates(
            candidate_tokens
        )

    total_chars = max(len(full_segment.text), 1)
    total_dur = full_segment.end_time - full_segment.start_time
    curr_char = 0
    sentence_subtitles, all_tokens, all_matched, all_knowledge_units, all_bunsetsu = [], [], [], [], []
    translation_failures = []

    # 4. Stage 2: Sentence translation inside bounded chunks.
    chunk_ranges = llm_svc.build_translation_chunks(sentence_texts)
    chunk_map = {}
    for chunk_no, (chunk_start, chunk_end) in enumerate(chunk_ranges, start=1):
        for sentence_index in range(chunk_start, chunk_end):
            chunk_map[sentence_index] = chunk_no
    logger.info(f"   [Translation Planner] Created {len(chunk_ranges)} LLM translation chunks.")

    # One LLM request per chunk; sentence-level translation is fallback-only.
    chunk_results_by_sentence = {}
    chunk_batch_meta = {}
    for chunk_no, (chunk_start, chunk_end) in enumerate(chunk_ranges, start=1):
        chunk_t0 = time.time()
        chunk_input = [
            {
                "sentence_number": i + 1,
                "text": sentence_texts[i],
                "token_candidates": sentence_token_candidates.get(i + 1, {}),
            }
            for i in range(chunk_start, chunk_end)
        ]
        try:
            chunk_results = llm_svc.translate_chunk(chunk_input)
        except Exception as exc:
            logger.warning(f"   [Chunk #{chunk_no}] Batch translation error: {exc}")
            chunk_results = {}
        expected_count = chunk_end - chunk_start
        returned_count = len(chunk_results)
        chunk_batch_meta[chunk_no] = {
            "expected": expected_count,
            "returned": returned_count,
            "elapsed": round(time.time() - chunk_t0, 2),
        }
        if chunk_results:
            chunk_results_by_sentence.update(chunk_results)
            logger.info(
                f"   [Chunk #{chunk_no} Batch] {returned_count}/{expected_count} results "
                f"in {chunk_batch_meta[chunk_no]['elapsed']}s."
            )
            for sentence_no in range(chunk_start + 1, chunk_end + 1):
                item = chunk_results.get(sentence_no)
                if item:
                    logger.info(
                        f"      [#{sentence_no}] JP: {sentence_texts[sentence_no - 1]}"
                    )
                    logger.info(
                        f"      [#{sentence_no}] VI: {str(item.get('translation_vi', '')).strip()}"
                    )
            missing = [
                sentence_no for sentence_no in range(chunk_start + 1, chunk_end + 1)
                if sentence_no not in chunk_results
            ]
            if missing:
                logger.warning(f"      [Chunk #{chunk_no} Batch Missing] {missing}")
        else:
            logger.warning(
                f"   [Chunk #{chunk_no} Batch] FAILED after "
                f"{chunk_batch_meta[chunk_no]['elapsed']}s; sentence-level fallback will be used."
            )

    t3 = time.time()
    chunk_sentence_outputs = {}
    for idx, sent_text in enumerate(sentence_texts, start=1):
        s_t0 = time.time()
        sent_len = len(sent_text)
        s_start = round(full_segment.start_time + (curr_char / total_chars) * total_dur, 2)
        curr_char += sent_len
        s_end = round(full_segment.start_time + (curr_char / total_chars) * total_dur, 2)

        sent_seg = SubtitleSegment(
            segment_id=f"{full_segment.segment_id}-s{idx}",
            video_id=full_segment.video_id,
            video_title=full_segment.video_title,
            source_url=full_segment.source_url,
            video_path=full_segment.video_path,
            start_time=s_start,
            end_time=s_end, text=sent_text, sequence_number=idx,
        )
        chunk_no = chunk_map.get(idx - 1, 1)
        logger.info(
            f"\n--- Chunk #{chunk_no} | Sentence #{idx} "
            f"[{s_start:.1f}s - {s_end:.1f}s]: \"{sent_text}\" ---"
        )

        # Step A: NLP Morphological Analysis
        tokens = sentence_tokens.get(idx, [])
        for token in tokens:
            token.segment_id = sent_seg.segment_id
        token_candidates = sentence_token_candidates.get(idx, {})
        # Step B: Hierarchical Knowledge Matching (Grammar -> Phrase -> CompoundWord -> Word)
        hier_units, oov_toks = knowledge_svc.match_hierarchical(tokens, sent_text)
        matched_k = [DictionaryEntry(term=u.surface, reading=u.reading, pos=u.unit_type, meaning=u.meaning) for u in hier_units]
        all_matched.extend(matched_k)
        all_knowledge_units.extend(hier_units)

        # Step C: Consume chunk result; fall back to sentence-level context translation only if needed.
        chunk_item = chunk_results_by_sentence.get(idx)
        if chunk_item:
            s_trans_vi = str(chunk_item.get("translation_vi", "")).strip()
            raw_token_selections = chunk_item.get("token_selections", [])
            if raw_token_selections:
                logger.info(
                    f"   [Token Candidate Trace] Sentence #{idx} | candidates={token_candidates}"
                )
            token_meanings = llm_svc.resolve_token_selections(
                raw_token_selections, token_candidates, tokens, sentence_number=idx
            )
            if raw_token_selections:
                logger.info(
                    f"   [Token Meaning Trace] Sentence #{idx} | "
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
            s_trans_vi, token_meanings, enriched_oovs = llm_svc.translate_sentence_with_context(
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
        for o in enriched_oovs:
            oov_svc.persist_candidate(o)
        sent_seg.translation = s_trans_vi
        logger.info(f"   [Sentence Translation (VI)]: \"{s_trans_vi}\"")

        # Step D: Assign Contextual Meanings to Tokens
        for tok in tokens:
            # Occurrence-level token_id is authoritative; surface aliases are only
            # a compatibility fallback and can collide when a surface repeats.
            if tok.token_id in token_meanings:
                tok.context_meaning = token_meanings[tok.token_id]
            elif tok.surface in token_meanings:
                tok.context_meaning = token_meanings[tok.surface]
        all_tokens.extend(tokens)

        # Step E: Bunsetsu Grouping with contextual token meanings
        logger.info(
            f"   [Meaning Propagation Trace] Sentence #{idx} | "
            f"resolved={token_meanings} | "
            f"token_context={[(t.surface, t.context_meaning) for t in tokens if t.context_meaning]}"
        )
        phrases = bunsetsu_svc.group_bunsetsu(sent_seg, tokens, token_meanings)
        all_bunsetsu.extend(phrases)

        sentence_subtitles.append(SentenceSubtitle(
            segment=sent_seg, translation=s_trans_vi, tokens=tokens,
            matched_knowledge=matched_k, knowledge_units=hier_units,
            oov_candidates=enriched_oovs, bunsetsu_phrases=phrases,
        ))
        logger.info(f"   [Sentence #{idx} Done] Elapsed: {time.time() - s_t0:.2f}s")
        chunk_sentence_outputs[idx] = s_trans_vi
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
    if translation_failures:
        logger.warning(
            f"   [STEP 4 PARTIAL] {len(translation_failures)} sentence translations failed: "
            f"{[f['sentence_number'] for f in translation_failures]}"
        )
    logger.info(f"   [STEP 4 COMPLETED] All sentences processed in: {time.time() - t3:.2f}s")

    # 5. Assemble Consolidated Pipeline Result
    result = PipelineResult(
        segment=full_segment,
        tokens=all_tokens, matched_knowledge=all_matched, knowledge_units=all_knowledge_units,
        oov_candidates=[o for s in sentence_subtitles for o in s.oov_candidates],
        bunsetsu_phrases=all_bunsetsu, sentences=sentence_subtitles,
    )

    _print_pipeline_summary(result)

    total_elapsed = time.time() - total_start_time
    pipeline_status = "PARTIAL_SUCCESS" if translation_failures else "COMPLETED"
    logger.info(
        f"\n>>> [PIPELINE {pipeline_status}] Total processing time for video: "
        f"{total_elapsed:.2f}s"
    )
    logger.info(f"[RUN LOG] Full pipeline log saved at: {os.path.abspath(log_path)}")
    close_pipeline_log(log_handler)
    return result

if __name__ == "__main__":
    media_file = sys.argv[1] if len(sys.argv) > 1 else "samples/japanese_podcast_10s.mp4"
    run_pipeline(media_file)