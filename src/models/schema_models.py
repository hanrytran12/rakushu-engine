"""Data models aligned with the Rakushu Capstone Logical ERD."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any
import uuid


@dataclass
class SubtitleSegment:
    segment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    video_id: str = "video-sample-01"
    start_time: float = 0.0
    end_time: float = 10.0
    text: str = ""
    translation: str = ""
    sequence_number: int = 1
    video_title: str = ""
    source_url: str = ""
    video_path: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class TokenModel:
    token_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    segment_id: str = ""
    surface: str = ""
    pos: str = ""
    lemma: str = ""
    reading: str = ""
    romaji: str = ""
    start_position: int = 0
    end_position: int = 0
    context_meaning: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class DefinitionTag:
    name: str = ""
    category: str = ""
    description_en: str = ""
    description_vi: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class InflectionRule:
    rule_code: str = ""
    name_vi: str = ""
    description_vi: str = ""
    examples: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class DictionaryEntry:
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    term: str = ""
    reading: str = ""
    pos: str = ""
    definition_tags: str = ""
    rules: str = ""
    score: int = 1
    meaning: str = ""
    sequence: int = 0
    term_tags: str = ""
    gloss_en: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class OovCandidate:
    oov_candidate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    token_id: str = ""
    term: str = ""
    tentative_reading: str = ""
    suggested_meaning: str = ""
    suggested_pos: str = ""
    context_snippet: str = ""
    confidence_score: float = 0.0
    status: str = "PENDING_CURATOR_REVIEW"
    detected_at: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class BunsetsuPhrase:
    phrase_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    segment_id: str = ""
    text: str = ""
    translation: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    phrase_order: int = 0
    token_surfaces: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class SentenceSubtitle:
    """Represents a standalone subtitle sentence with translation and bunsetsu chunks."""
    sentence_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    segment: SubtitleSegment = field(default_factory=SubtitleSegment)
    translation: str = ""
    tokens: List[TokenModel] = field(default_factory=list)
    matched_knowledge: List[DictionaryEntry] = field(default_factory=list)
    knowledge_units: List[Any] = field(default_factory=list)
    oov_candidates: List[OovCandidate] = field(default_factory=list)
    bunsetsu_phrases: List[BunsetsuPhrase] = field(default_factory=list)

    def to_dict(self):
        return {
            "sentence_id": self.sentence_id,
            "segment": self.segment.to_dict(),
            "translation": self.translation,
            "tokens": [t.to_dict() for t in self.tokens],
            "matched_knowledge": [k.to_dict() for k in self.matched_knowledge],
            "knowledge_units": [u.to_dict() if hasattr(u, "to_dict") else u for u in self.knowledge_units],
            "oov_candidates": [o.to_dict() for o in self.oov_candidates],
            "bunsetsu_phrases": [p.to_dict() for p in self.bunsetsu_phrases],
        }


@dataclass
class PipelineResult:
    segment: SubtitleSegment

    tokens: List[TokenModel] = field(default_factory=list)
    matched_knowledge: List[DictionaryEntry] = field(default_factory=list)
    knowledge_units: List[Any] = field(default_factory=list)
    oov_candidates: List[OovCandidate] = field(default_factory=list)
    bunsetsu_phrases: List[BunsetsuPhrase] = field(default_factory=list)
    sentences: List[SentenceSubtitle] = field(default_factory=list)

    def to_dict(self):
        return {
            "segment": self.segment.to_dict(),

            "tokens": [t.to_dict() for t in self.tokens],
            "matched_knowledge": [k.to_dict() for k in self.matched_knowledge],
            "knowledge_units": [u.to_dict() if hasattr(u, "to_dict") else u for u in self.knowledge_units],
            "oov_candidates": [o.to_dict() for o in self.oov_candidates],
            "bunsetsu_phrases": [p.to_dict() for p in self.bunsetsu_phrases],
            "sentences": [s.to_dict() for s in self.sentences],
        }


class MediaSourceType:
    YOUTUBE_URL = "YOUTUBE_URL"
    LOCAL_UPLOAD = "LOCAL_UPLOAD"


class VideoType:
    PODCAST_TALKSHOW = "PODCAST_TALKSHOW"
    NEWS_FORMAL = "NEWS_FORMAL"
    ANIME_DRAMA = "ANIME_DRAMA"
    CASUAL_VLOG = "CASUAL_VLOG"
    MUSIC_SONG = "MUSIC_SONG"
    GENERAL_CONVERSATION = "GENERAL_CONVERSATION"


@dataclass
class MediaInspectionResult:
    is_valid: bool = True
    source_type: str = MediaSourceType.LOCAL_UPLOAD
    detected_language: str = "ja"
    language_probability: float = 1.0
    video_type: str = VideoType.GENERAL_CONVERSATION
    whisper_config: dict = field(default_factory=dict)
    rejection_reason: str = ""
    error_code: str = ""
    sample_slices_count: int = 0

    def to_dict(self):
        return asdict(self)

