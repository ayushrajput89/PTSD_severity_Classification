import unittest
from src.features.preprocessor import clean_text

class TestPreprocessor(unittest.TestCase):
    """Unit tests for text preprocessing functions."""
    
    def test_clean_text_removes_urls(self):
        """Test clean_text removes URLs."""
        text = "Check this link http://example.com"
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "check this link")

    def test_clean_text_removes_html_tags(self):
        """Test clean_text removes HTML tags."""
        text = "<p>Hello <b>World</b></p>"
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "hello world")

    def test_clean_text_handles_empty_string(self):
        """Test clean_text handles empty string."""
        text = ""
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "")

    def test_clean_text_lowercases(self):
        """Test clean_text lowercases."""
        text = "HeLLo WoRLd"
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "hello world")

    def test_clean_text_removes_special_characters_keeps_apostrophes(self):
        """Test clean_text removes special characters but keeps apostrophes."""
        text = "Don't panic! It's going to be okay @user #hope"
        cleaned = clean_text(text)
        self.assertIn("don't", cleaned)
        self.assertIn("it's", cleaned)
        self.assertNotIn("@", cleaned)
        self.assertNotIn("#", cleaned)

    def test_clean_text_collapses_spaces(self):
        """Test clean_text collapses spaces."""
        text = "This    has   too many    spaces."
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "this has too many spaces")

if __name__ == '__main__':
    unittest.main()
