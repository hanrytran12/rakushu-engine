"""Unit test suite for KnowledgeService SQLite dictionary database and OOV detection."""
import unittest
import os
import sys
import tempfile
import sqlite3

# Adjust search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services import KnowledgeService
from src.models import TokenModel, DictionaryEntry


class TestKnowledgeService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "dictionary.db"))

    def test_01_real_database_lookup(self):
        """Tests that real words are successfully retrieved from SQLite with definition tags and rules."""
        if not os.path.exists(self.real_db_path):
            self.skipTest("data/dictionary.db not found, skipping real db lookup")

        svc = KnowledgeService(db_path=self.real_db_path, exclude_terms=set())
        
        # Test basic vocabulary
        entry = svc.lookup("今日")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.term, "今日")
        self.assertEqual(entry.reading, "きょう")
        self.assertIn("hôm nay", entry.meaning.lower())

        # Test verb lookup with definition_tags and rules
        entry_verb = svc.lookup("食べる")
        self.assertIsNotNone(entry_verb)
        self.assertEqual(entry_verb.pos, "VERB")
        self.assertIn("v1", entry_verb.definition_tags)
        self.assertEqual(entry_verb.rules, "v1")
        self.assertGreater(entry_verb.sequence, 0)

        # Test get_tag_info
        tag_info = svc.get_tag_info("v1")
        self.assertIsNotNone(tag_info)
        self.assertEqual(tag_info.name, "v1")
        self.assertIn("Ichidan", tag_info.description_en)
        self.assertIn("Động từ nhóm 2", tag_info.description_vi)

        # Test get_rule_info
        rule_info = svc.get_rule_info("v1")
        self.assertIsNotNone(rule_info)
        self.assertEqual(rule_info.rule_code, "v1")
        self.assertIn("Ichidan", rule_info.name_vi)
        self.assertIn("食べる", rule_info.examples)

    def test_02_match_tokens_and_oov_detection(self):
        """Tests categorizing tokens into known vocabulary vs OOV candidates."""
        svc = KnowledgeService(exclude_terms={"テスト未知語"})
        tokens = [
            TokenModel(surface="日本", lemma="日本", pos="NOUN"),
            TokenModel(surface="テスト未知語", lemma="テスト未知語", pos="NOUN"),
            TokenModel(surface="です", lemma="です", pos="AUX_VERB"),
            TokenModel(surface="。", lemma="。", pos="PUNCTUATION"),
        ]

        matched, oov_tokens = svc.match_tokens(tokens)
        matched_terms = [m.term for m in matched]
        oov_surfaces = [t.surface for t in oov_tokens]

        self.assertIn("日本", matched_terms)
        self.assertIn("テスト未知語", oov_surfaces)
        # Functional grammar tokens should be ignored from both
        self.assertNotIn("です", matched_terms)
        self.assertNotIn("です", oov_surfaces)

    def test_03_register_enriched_oov_persistence(self):
        """Tests that registering an OOV dynamically persists it and enables future lookup."""
        # Use isolated temp database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp_db = f.name

        try:
            conn = sqlite3.connect(tmp_db)
            conn.execute("""
                CREATE TABLE dictionary (
                    term TEXT NOT NULL, reading TEXT NOT NULL DEFAULT '', pos TEXT NOT NULL DEFAULT 'NOUN',
                    definition_tags TEXT DEFAULT '', rules TEXT DEFAULT '', score INTEGER DEFAULT 1,
                    meaning TEXT NOT NULL, sequence INTEGER DEFAULT 0, term_tags TEXT DEFAULT '',
                    PRIMARY KEY (term, reading)
                );
            """)
            conn.commit()
            conn.close()

            svc = KnowledgeService(db_path=tmp_db, exclude_terms={"パリピ"})
            self.assertIsNone(svc.lookup("パリピ"))

            # Register OOV
            new_entry = DictionaryEntry(
                term="パリピ", reading="ぱりぴ", pos="NOUN",
                meaning="dân quẩy, người thích tiệc tùng (party people)", score=1
            )
            svc.register_enriched_oov(new_entry)

            # Check lookup now works
            queried = svc.lookup("パリピ")
            self.assertIsNotNone(queried)
            self.assertEqual(queried.term, "パリピ")
            self.assertEqual(queried.reading, "ぱりぴ")
            self.assertIn("tiệc tùng", queried.meaning)

            # Verify persisted in SQLite
            conn = sqlite3.connect(tmp_db)
            cur = conn.cursor()
            cur.execute("SELECT term, reading, meaning FROM dictionary WHERE term = 'パリピ'")
            row = cur.fetchone()
            conn.close()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], "パリピ")
        finally:
            if os.path.exists(tmp_db):
                os.remove(tmp_db)

    def test_04_fallback_when_no_database(self):
        """Tests that KnowledgeService falls back to built-in vocabulary when DB file is missing."""
        svc = KnowledgeService(db_path="non_existent_file.db", exclude_terms=set())
        entry = svc.lookup("今日")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.reading, "きょう")
        self.assertIn("hôm nay", entry.meaning)



    def test_05_lookup_candidates_and_token_candidate_payload(self):
        """Returns multiple dictionary senses and stable request-local candidate IDs."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp_db = f.name
        try:
            conn = sqlite3.connect(tmp_db)
            conn.execute("""
                CREATE TABLE dictionary (
                    term TEXT, reading TEXT, pos TEXT, definition_tags TEXT,
                    rules TEXT, score INTEGER, meaning TEXT, sequence INTEGER, term_tags TEXT
                );
            """)
            conn.executemany(
                "INSERT INTO dictionary "
                "(term, reading, pos, score, meaning, sequence) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    ("やすい", "やすい", "ADJECTIVE", 10, "rẻ, giá thấp", 1),
                    ("やすい", "やすい", "ADJECTIVE", 9, "dễ, dễ dàng", 2),
                    ("やすい", "やすい", "ADJECTIVE", 9, "dễ thực hiện", 3),
                ],
            )
            conn.commit()
            conn.close()

            svc = KnowledgeService(db_path=tmp_db, exclude_terms=set())
            candidates = svc.lookup_candidates("やすい", limit=2)
            self.assertEqual(len(candidates), 2)
            self.assertEqual(candidates[0].meaning, "rẻ, giá thấp")
            self.assertEqual(candidates[1].meaning, "dễ, dễ dàng")

            token = TokenModel(surface="やすい", lemma="やすい", pos="ADJECTIVE")
            payload = svc.build_token_candidates([token], limit=2)
            self.assertEqual(list(payload.keys()), ["t1"])
            self.assertEqual(
                [c["candidate_id"] for c in payload["t1"]],
                ["t1-c1", "t1-c2"],
            )
            self.assertEqual(
                [c["meaning"] for c in payload["t1"]],
                ["rẻ, giá thấp", "dễ, dễ dàng"],
            )
            self.assertEqual(payload["t1"][0]["token"], "やすい")
            self.assertEqual(payload["t1"][0]["token_id"], token.token_id)

        finally:
            if os.path.exists(tmp_db):
                os.remove(tmp_db)


    def test_06_multi_gloss_expands_to_individual_senses_and_duplicate_tokens_align(self):
        """Each gloss is a separate candidate and repeated surfaces keep separate token groups."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp_db = f.name
        try:
            conn = sqlite3.connect(tmp_db)
            conn.execute("""
                CREATE TABLE dictionary (
                    term TEXT, reading TEXT, pos TEXT, definition_tags TEXT,
                    rules TEXT, score INTEGER, meaning TEXT, sequence INTEGER, term_tags TEXT
                );
            """)
            conn.execute(
                "INSERT INTO dictionary "
                "(term, reading, pos, score, meaning, sequence) VALUES (?, ?, ?, ?, ?, ?)",
                ("勉強", "べんきょう", "NOUN", 10,
                 "1. học tập; 2. nỗ lực học tập; 3. kinh nghiệm; 4. kiến thức", 1),
            )
            conn.commit()
            conn.close()

            svc = KnowledgeService(db_path=tmp_db, exclude_terms=set())
            entries = svc.lookup_candidates("勉強", limit=5)
            self.assertEqual(
                [entry.meaning for entry in entries],
                ["học tập", "nỗ lực học tập", "kinh nghiệm", "kiến thức"],
            )

            token1 = TokenModel(surface="勉強", lemma="勉強", pos="NOUN")
            token2 = TokenModel(surface="勉強", lemma="勉強", pos="NOUN")
            payload = svc.build_token_candidates([token1, token2], limit=2)

            self.assertEqual(list(payload.keys()), ["t1", "t2"])
            self.assertNotEqual(payload["t1"][0]["token_id"], payload["t2"][0]["token_id"])
            self.assertEqual(payload["t1"][0]["candidate_id"], "t1-c1")
            self.assertEqual(payload["t2"][0]["candidate_id"], "t2-c1")
            self.assertTrue(all(";" not in c["meaning"] for c in payload["t1"]))
        finally:
            if os.path.exists(tmp_db):
                os.remove(tmp_db)


    def test_07_candidate_builder_never_uses_english_gloss_as_meaning(self):
        """A blank Vietnamese field stays unavailable instead of becoming an English candidate."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp_db = f.name
        try:
            conn = sqlite3.connect(tmp_db)
            conn.execute(
                "CREATE TABLE staging_entries "
                "(term TEXT, reading TEXT, pos TEXT, meaning_vi TEXT, gloss_en TEXT)"
            )
            conn.execute(
                "INSERT INTO staging_entries VALUES (?, ?, ?, ?, ?)",
                ("テスト語", "てすとご", "noun", "", "an English-only test gloss"),
            )
            conn.commit()
            conn.close()

            service = KnowledgeService(db_path=tmp_db, exclude_terms=set())
            self.assertEqual(service.lookup_candidates("テスト語"), [])
            self.assertIsNone(service.lookup("テスト語"))
        finally:
            if os.path.exists(tmp_db):
                os.remove(tmp_db)

    def test_08_exact_term_candidates_take_priority_over_reading_homographs(self):
        """An exact lexical term excludes unrelated entries that merely share its reading."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp_db = f.name
        try:
            conn = sqlite3.connect(tmp_db)
            conn.execute(
                "CREATE TABLE staging_entries "
                "(term TEXT, reading TEXT, pos TEXT, meaning_vi TEXT, gloss_en TEXT)"
            )
            conn.executemany(
                "INSERT INTO staging_entries VALUES (?, ?, ?, ?, ?)",
                [
                    ("いる", "いる", "Ichidan verb", "ở, tồn tại", "to exist"),
                    ("射る", "いる", "Ichidan verb", "bắn bằng cung", "to shoot"),
                    ("炒る", "いる", "Godan verb", "rang, sao", "to roast"),
                    ("入る", "いる", "Godan verb", "đi vào", "to enter"),
                ],
            )
            conn.commit()
            conn.close()

            service = KnowledgeService(db_path=tmp_db, exclude_terms=set())
            candidates = service.lookup_candidates("いる", limit=5)
            self.assertEqual([entry.term for entry in candidates], ["いる"])
            self.assertEqual(candidates[0].meaning, "ở, tồn tại")
            lookup = service.lookup("いる")
            self.assertIsNotNone(lookup)
            self.assertEqual(lookup.term, "いる")
        finally:
            if os.path.exists(tmp_db):
                os.remove(tmp_db)

    def test_09_auxiliary_iru_after_te_does_not_receive_lexical_candidates(self):
        """The auxiliary iru in a te-iru construction is handled grammatically, not lexically."""
        from unittest.mock import patch

        service = KnowledgeService(exclude_terms=set())
        tokens = [
            TokenModel(surface="食べ", lemma="食べる", reading="タベ", pos="VERB"),
            TokenModel(surface="て", lemma="て", reading="テ", pos="PARTICLE"),
            TokenModel(surface="い", lemma="いる", reading="イル", pos="VERB"),
            TokenModel(surface="ます", lemma="ます", reading="マス", pos="AUX_VERB"),
        ]
        with patch.object(
            service,
            "lookup_candidates",
            return_value=[DictionaryEntry(
                term="いる", reading="いる", pos="Ichidan verb", meaning="ở"
            )],
        ):
            candidates = service.build_token_candidates(tokens)
        self.assertNotIn("t3", candidates)


    def test_10_lookup_skips_empty_vi_rows_before_valid_exact_term_row(self):
        """A blank first row does not hide a Vietnamese entry for the same exact term."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp_db = f.name
        try:
            conn = sqlite3.connect(tmp_db)
            conn.execute(
                "CREATE TABLE staging_entries "
                "(term TEXT, reading TEXT, pos TEXT, meaning_vi TEXT, gloss_en TEXT)"
            )
            conn.executemany(
                "INSERT INTO staging_entries VALUES (?, ?, ?, ?, ?)",
                [
                    ("後続語", "こうぞくご", "noun", "", "English-only first row"),
                    ("後続語", "こうぞくご", "noun", "nghĩa tiếng Việt", "English metadata"),
                    ("同音語", "こうぞくご", "noun", "nghĩa homograph", "homograph gloss"),
                ],
            )
            conn.commit()
            conn.close()

            service = KnowledgeService(db_path=tmp_db, exclude_terms=set())
            entry = service.lookup("後続語")
            candidates = service.lookup_candidates("後続語")
            self.assertIsNotNone(entry)
            self.assertEqual(entry.meaning, "nghĩa tiếng Việt")
            self.assertEqual([candidate.meaning for candidate in candidates], ["nghĩa tiếng Việt"])
            self.assertEqual([candidate.term for candidate in candidates], ["後続語"])
        finally:
            if os.path.exists(tmp_db):
                os.remove(tmp_db)

if __name__ == "__main__":
    unittest.main()
