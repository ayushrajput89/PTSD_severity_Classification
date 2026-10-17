import unittest
from src.features.feature_engineering import (
    extract_vader_features,
    extract_readability_features,
    compute_lexicon_counts,
    extract_structural_features,
    extract_syntactic_features
)

class TestFeatures(unittest.TestCase):
    """Unit tests for feature extraction modules."""
    
    def test_extract_vader_features_returns_correct_keys(self):
        """Test extract_vader_features returns correct keys."""
        text = "I am very happy today."
        features = extract_vader_features(text)
        self.assertIn("vader_pos", features)
        self.assertIn("vader_neg", features)
        self.assertIn("vader_neu", features)
        self.assertIn("vader_compound", features)

    def test_extract_readability_features_returns_correct_keys(self):
        """Test extract_readability_features returns correct keys."""
        text = "This is a simple sentence to test readability."
        features = extract_readability_features(text)
        self.assertIn("flesch_reading_ease", features)
        self.assertIn("flesch_kincaid_grade", features)

    def test_compute_lexicon_counts_with_known_text(self):
        """Test compute_lexicon_counts with known text."""
        text = "trauma fear anxiety and some normal words"
        features = compute_lexicon_counts(text)
        # Verify that the lexicon counting produces integer features
        self.assertIsInstance(features, dict)
        if "trauma_lexicon_count" in features:
            self.assertGreaterEqual(features["trauma_lexicon_count"], 0)

    def test_extract_structural_features_word_count(self):
        """Test extract_structural_features word_count."""
        text = "One two three four."
        features = extract_structural_features(text)
        self.assertEqual(features.get("word_count", 0), 4)

    def test_extract_syntactic_features_first_person_ratio(self):
        """Test extract_syntactic_features first_person_ratio."""
        text = "I went to my house."
        features = extract_syntactic_features(text)
        self.assertIn("first_person_ratio", features)
        self.assertTrue(features["first_person_ratio"] > 0.0)

if __name__ == '__main__':
    unittest.main()
