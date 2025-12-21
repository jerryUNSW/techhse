"""
Tests for PhraseDP mechanism.
"""

import pytest
from unittest.mock import Mock, patch
from sentence_transformers import SentenceTransformer

from dpprivqa.mechanisms.phrasedp import phrasedp_sanitize, PhraseDP


def test_phrasedp_sanitize_basic(mock_local_client):
    """Test basic PhraseDP sanitization."""
    text = "What is the treatment for diabetes?"
    
    # Mock the candidate generation and selection
    with patch('dpprivqa.mechanisms.phrasedp.generate_sentence_replacements_with_nebius_diverse') as mock_gen, \
         patch('dpprivqa.mechanisms.phrasedp.differentially_private_replacement') as mock_select:
        
        mock_gen.return_value = [
            "What treatment exists for diabetes?",
            "How is diabetes treated?",
            "What are diabetes treatment options?"
        ]
        mock_select.return_value = "What treatment exists for diabetes?"
        
        result = phrasedp_sanitize(
            text=text,
            epsilon=2.0,
            nebius_client=mock_local_client,
            nebius_model_name="test-model"
        )
        
        assert result != text  # Should be perturbed
        assert isinstance(result, str)
        assert len(result) > 0


def test_phrasedp_sanitize_empty_text(mock_local_client):
    """Test PhraseDP with empty text."""
    result = phrasedp_sanitize(
        text="",
        epsilon=2.0,
        nebius_client=mock_local_client,
        nebius_model_name="test-model"
    )
    assert result == ""


def test_phrasedp_sanitize_error_handling(mock_local_client):
    """Test PhraseDP error handling."""
    with pytest.raises(ValueError):
        phrasedp_sanitize(
            text="test",
            epsilon=-1.0,  # Invalid epsilon
            nebius_client=mock_local_client,
            nebius_model_name="test-model"
        )


def test_phrasedp_class(mock_local_client):
    """Test PhraseDP class."""
    phrasedp = PhraseDP(
        nebius_client=mock_local_client,
        nebius_model_name="test-model"
    )
    
    assert phrasedp.get_name() == "phrasedp"
    assert phrasedp.medical_mode == False


def test_phrasedp_medical_mode(mock_local_client):
    """Test PhraseDP with medical mode enabled."""
    phrasedp = PhraseDP(
        nebius_client=mock_local_client,
        nebius_model_name="test-model",
        medical_mode=True
    )
    
    assert phrasedp.medical_mode == True


