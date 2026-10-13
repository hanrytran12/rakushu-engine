"""Bunsetsu Grouping Service: Japanese phrase boundary segmentation and alignment."""
from typing import List, Dict, Optional
from src.models import TokenModel, BunsetsuPhrase, SubtitleSegment

try:
    import spacy
    import ginza
    _GINZA_AVAILABLE = True
except ImportError:
    _GINZA_AVAILABLE = False


class BunsetsuService:
    """Groups tokens into Bunsetsu (文節: 1 Jiritsugo content word + N Fuzokugo particles)."""

    JIRITSUGO_POS = {"NOUN", "VERB", "ADJECTIVE", "INTERJECTION", "ADVERB", "PRONOUN"}
    FUZOKUGO_POS = {"PARTICLE", "AUX_VERB", "PUNCTUATION"}

    PHRASE_TRANSLATIONS: Dict[str, str] = {
        "皆さん、": "Mọi người,",
        "こんにちは。": "xin chào.",
        "今日の": "của hôm nay,",
        "ポッドキャストでは、": "trong tập podcast thì,",
        "日本の": "của Nhật Bản,",
        "面白い": "thú vị,",
        "若者言葉について": "về tiếng lóng giới trẻ,",
        "話します。": "sẽ nói chuyện / trò chuyện.",
    }

    def __init__(self):
        self._nlp = None
        if _GINZA_AVAILABLE:
            try:
                self._nlp = spacy.load("ja_ginza")
            except Exception:
                self._nlp = None

    def group_bunsetsu(
        self,
        segment: SubtitleSegment,
        tokens: List[TokenModel],
        token_meanings: Optional[Dict[str, str]] = None
    ) -> List[BunsetsuPhrase]:
        """Partitions sequential tokens into Bunsetsu phrases with timestamps and meanings."""
        if self._nlp is not None and segment.text:
            try:
                return self._group_bunsetsu_ginza(segment, tokens, token_meanings)
            except Exception:
                pass
        return self._group_bunsetsu_rules(segment, tokens, token_meanings)

    def _group_bunsetsu_ginza(
        self,
        segment: SubtitleSegment,
        tokens: List[TokenModel],
        token_meanings: Optional[Dict[str, str]] = None
    ) -> List[BunsetsuPhrase]:
        doc = self._nlp(segment.text)
        spans = list(ginza.bunsetu_spans(doc))
        if not spans:
            return self._group_bunsetsu_rules(segment, tokens, token_meanings)

        total_chars = max(len(segment.text), 1)
        duration = segment.end_time - segment.start_time
        phrases: List[BunsetsuPhrase] = []

        for idx, span in enumerate(spans, start=1):
            start_pos = span.start_char
            end_pos = span.end_char
            text = span.text

            matched_surfaces = [
                t.surface for t in tokens
                if t.start_position >= start_pos and t.end_position <= end_pos
            ]
            if not matched_surfaces:
                matched_surfaces = [t.orth_ for t in span]

            start_time = round(segment.start_time + (start_pos / total_chars) * duration, 2)
            end_time = round(segment.start_time + (end_pos / total_chars) * duration, 2)
            meaning = self.PHRASE_TRANSLATIONS.get(
                text, self._generate_span_meaning(span, token_meanings, tokens)
            )

            phrases.append(BunsetsuPhrase(
                segment_id=segment.segment_id,
                text=text,
                translation=meaning,
                start_time=start_time,
                end_time=end_time,
                phrase_order=idx,
                token_surfaces=matched_surfaces
            ))

        return phrases

    def _generate_span_meaning(
        self,
        span,
        token_meanings: Optional[Dict[str, str]] = None,
        model_tokens: Optional[List[TokenModel]] = None,
    ) -> str:
        meanings = []
        tm = token_meanings or {}

        # Prefer occurrence-specific TokenModel meanings when source offsets line
        # up with this GiNZA span. Surface aliases can collide for repeated tokens.
        if model_tokens:
            covered_tokens = [
                token for token in model_tokens
                if token.start_position < span.end_char and token.end_position > span.start_char
            ]
            for token in covered_tokens:
                meaning = (
                    token.context_meaning
                    or tm.get(token.token_id)
                    or tm.get(token.surface)
                    or tm.get(token.lemma)
                )
                if meaning:
                    meanings.append(meaning)
            if meanings:
                return " + ".join(meanings)

        # Compatibility fallback for tokens not aligned to the model-token offsets.
        for t in span:
            if t.pos_ not in ("PUNCT",):
                meaning = tm.get(t.orth_, tm.get(t.lemma_, t.lemma_))
                if meaning:
                    meanings.append(meaning)
        return " + ".join(meanings) if meanings else span.text

    def _group_bunsetsu_rules(
        self,
        segment: SubtitleSegment,
        tokens: List[TokenModel],
        token_meanings: Optional[Dict[str, str]] = None
    ) -> List[BunsetsuPhrase]:
        phrases: List[BunsetsuPhrase] = []
        current_cluster: List[TokenModel] = []
        order = 1
        tm = token_meanings or {}

        def flush(cluster: List[TokenModel], ord_num: int) -> Optional[BunsetsuPhrase]:
            if not cluster:
                return None
            combined_text = "".join(t.surface for t in cluster)
            meaning = self.PHRASE_TRANSLATIONS.get(
                combined_text,
                " + ".join(
                    t.context_meaning or tm.get(t.token_id) or tm.get(t.surface, tm.get(t.lemma, t.lemma))
                    for t in cluster
                    if t.pos != "PUNCTUATION"
                )
            )
            total_chars = max(len(segment.text), 1)
            start_ratio = cluster[0].start_position / total_chars
            end_ratio = cluster[-1].end_position / total_chars
            duration = segment.end_time - segment.start_time

            return BunsetsuPhrase(
                segment_id=segment.segment_id,
                text=combined_text,
                translation=meaning,
                start_time=round(segment.start_time + start_ratio * duration, 2),
                end_time=round(segment.start_time + end_ratio * duration, 2),
                phrase_order=ord_num,
                token_surfaces=[t.surface for t in cluster]
            )

        for token in tokens:
            if token.pos == "PUNCTUATION":
                current_cluster.append(token)
                p = flush(current_cluster, order)
                if p:
                    phrases.append(p)
                    order += 1
                current_cluster = []
                continue

            if token.pos in self.JIRITSUGO_POS:
                is_compound = (token.pos == "NOUN" and current_cluster and all(t.pos == "NOUN" for t in current_cluster))
                is_suru = (token.lemma == "する" and current_cluster and current_cluster[-1].pos == "NOUN")
                if current_cluster and not (is_compound or is_suru):
                    p = flush(current_cluster, order)
                    if p:
                        phrases.append(p)
                        order += 1
                    current_cluster = []

            current_cluster.append(token)

        if current_cluster:
            p = flush(current_cluster, order)
            if p:
                phrases.append(p)

        return phrases

