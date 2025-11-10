"""
Test LinkedIn fetcher with new AI companies.

Tests that the fetcher correctly extracts company names from LinkedIn titles
for all the newly added AI companies.
"""

import pytest
from src.fetch.linkedin_fetcher import LinkedinFetcher
from unittest.mock import patch


class TestNewCompanyExtraction:
    """Test company extraction for newly added AI companies."""

    @patch('src.fetch.linkedin_fetcher.get_queries_config')
    def test_extract_major_ai_labs(self, mock_config):
        """Test extraction of major AI lab companies."""
        mock_config.return_value = {'linkedin': {'rate_limit_delay': 1}}
        fetcher = LinkedinFetcher()

        test_cases = [
            ("Research Scientist at OpenAI", "OpenAI"),
            ("ML Engineer at Anthropic", "Anthropic"),
            ("Senior Researcher at Google DeepMind", "Google DeepMind"),
            ("AI Researcher at xAI", "xAI"),
            ("Research Scientist at Meta AI", "Meta AI"),
            ("Engineer at Mistral AI", "Mistral AI"),
            ("DeepSeek - Research Scientist", "DeepSeek"),
            ("Qwen Team at Alibaba", "Qwen"),
        ]

        for title, expected_company in test_cases:
            company = fetcher._extract_company(title)
            assert company == expected_company, f"Failed for '{title}': expected '{expected_company}', got '{company}'"

    @patch('src.fetch.linkedin_fetcher.get_queries_config')
    def test_extract_research_institutions(self, mock_config):
        """Test extraction of research institution companies."""
        mock_config.return_value = {'linkedin': {'rate_limit_delay': 1}}
        fetcher = LinkedinFetcher()

        test_cases = [
            ("Principal Researcher at Microsoft Research", "Microsoft Research"),
            ("Research Scientist at NVIDIA Research", "NVIDIA"),
            ("AI Researcher at IBM Research", "IBM Research"),
            ("Research Scientist at AI2", "AI2"),
            ("ML Engineer at Hugging Face", "Hugging Face"),
            ("Research Scientist at Cohere", "Cohere"),
            ("Engineer at Minimax", "Minimax"),
            ("Researcher at Kimi K2 (Moonshot AI)", "Kimi K2"),
        ]

        for title, expected_company in test_cases:
            company = fetcher._extract_company(title)
            assert company == expected_company, f"Failed for '{title}': expected '{expected_company}', got '{company}'"

    @patch('src.fetch.linkedin_fetcher.get_queries_config')
    def test_extract_emerging_companies(self, mock_config):
        """Test extraction of emerging AI companies."""
        mock_config.return_value = {'linkedin': {'rate_limit_delay': 1}}
        fetcher = LinkedinFetcher()

        test_cases = [
            ("Research Scientist at Harmonic", "Harmonic"),
            ("ML Engineer at Axiom", "Axiom"),
            ("AI Researcher at Deep Cogito", "Deep Cogito"),
            ("Engineer at Z.AI", "Z.AI"),
        ]

        for title, expected_company in test_cases:
            company = fetcher._extract_company(title)
            assert company == expected_company, f"Failed for '{title}': expected '{expected_company}', got '{company}'"

    @patch('src.fetch.linkedin_fetcher.get_queries_config')
    def test_extract_tech_giant_ai_divisions(self, mock_config):
        """Test extraction of tech giant AI divisions."""
        mock_config.return_value = {'linkedin': {'rate_limit_delay': 1}}
        fetcher = LinkedinFetcher()

        test_cases = [
            ("ML Researcher at Apple", "Apple"),
            ("Research Scientist at Amazon Science", "Amazon"),
            ("Engineer at Google Brain", "Google Brain"),
            ("AI Researcher at Baidu Research", "Baidu"),
            ("ML Engineer at Tencent AI Lab", "Tencent AI"),
            ("Researcher at ByteDance AI Lab", "ByteDance AI"),
        ]

        for title, expected_company in test_cases:
            company = fetcher._extract_company(title)
            assert company == expected_company, f"Failed for '{title}': expected '{expected_company}', got '{company}'"

    @patch('src.fetch.linkedin_fetcher.get_queries_config')
    def test_various_title_formats(self, mock_config):
        """Test extraction with various title formats."""
        mock_config.return_value = {'linkedin': {'rate_limit_delay': 1}}
        fetcher = LinkedinFetcher()

        test_cases = [
            # Various formats
            ("OpenAI | Research Scientist", "OpenAI"),
            ("Research Scientist - OpenAI", "OpenAI"),
            ("@OpenAI Research Scientist", "OpenAI"),
            ("Working on LLMs at DeepSeek", "DeepSeek"),
            ("Former researcher at Google DeepMind, now at Anthropic", "Anthropic"),
            ("Passionate about AI, currently at xAI", "xAI"),
            ("Building the future at Meta AI", "Meta AI"),

            # Complex titles
            ("Senior Staff Research Scientist, Large Language Models at OpenAI", "OpenAI"),
            ("Principal Engineer, Foundation Models at Anthropic", "Anthropic"),
            ("Research Director, AI Safety at Google DeepMind", "Google DeepMind"),

            # Edge cases
            ("Just a title", None),
            ("", None),
            ("Consultant", None),
        ]

        for title, expected_company in test_cases:
            company = fetcher._extract_company(title)
            assert company == expected_company, f"Failed for '{title}': expected '{expected_company}', got '{company}'"

    @patch('src.fetch.linkedin_fetcher.get_queries_config')
    def test_verified_researcher_detection(self, mock_config):
        """Test verified researcher detection for new companies."""
        mock_config.return_value = {'linkedin': {'rate_limit_delay': 1}}
        fetcher = LinkedinFetcher()

        # Should be verified (research title OR top company)
        verified_cases = [
            ("Research Scientist at OpenAI", "OpenAI"),
            ("ML Engineer at DeepSeek", "DeepSeek"),
            ("Engineer at xAI", "xAI"),
            ("Researcher at Qwen", "Qwen"),
            ("AI Researcher at Mistral AI", "Mistral AI"),
            ("PhD Student working on LLMs", None),
            ("Doctoral Researcher", None),
            ("Staff Engineer at NVIDIA", "NVIDIA"),
            ("Product Manager at Meta AI", "Meta AI"),  # Top company, not research title
        ]

        for title, company in verified_cases:
            is_verified = fetcher._is_verified_researcher(title, company)
            assert is_verified, f"Should be verified: '{title}' at '{company}'"

        # Should NOT be verified
        non_verified_cases = [
            ("Software Engineer at Random Corp", "Random Corp"),
            ("Data Analyst at StartupXYZ", "StartupXYZ"),
            ("Consultant", None),
            ("Sales Manager at TechCorp", "TechCorp"),
        ]

        for title, company in non_verified_cases:
            is_verified = fetcher._is_verified_researcher(title, company)
            assert not is_verified, f"Should NOT be verified: '{title}' at '{company}'"

    @patch('src.fetch.linkedin_fetcher.get_queries_config')
    def test_company_name_variations(self, mock_config):
        """Test handling of company name variations."""
        mock_config.return_value = {'linkedin': {'rate_limit_delay': 1}}
        fetcher = LinkedinFetcher()

        # Test variations that should map to the same company
        variations = [
            ("Research Scientist at Google DeepMind", "Google DeepMind"),
            ("Researcher at DeepMind", "Google DeepMind"),
            ("ML Engineer at Meta AI", "Meta AI"),
            ("Engineer at Meta", "Meta AI"),
            ("Researcher at AI2", "AI2"),
            ("Allen Institute for AI", "AI2"),
            ("Engineer at Hugging Face", "Hugging Face"),
            ("ML Engineer at Huggingface", "Hugging Face"),
            ("Research at xAI", "xAI"),
            ("Research at X.AI", "xAI"),
            ("Kimi K2 Research", "Kimi K2"),
            ("Moonshot AI Engineer", "Kimi K2"),
        ]

        for title, expected_company in variations:
            company = fetcher._extract_company(title)
            assert company == expected_company, f"Failed for '{title}': expected '{expected_company}', got '{company}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])