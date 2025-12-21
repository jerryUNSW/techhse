"""
Tests for QA scenarios.
"""

import pytest
from dpprivqa.qa.scenarios import run_local_only, run_local_with_cot, run_remote_only
from dpprivqa.qa.prompts import check_mcq_correctness


def test_local_only_scenario(mock_local_client, sample_question):
    """Test local-only scenario."""
    result = run_local_only(
        mock_local_client,
        "test-model",
        sample_question["question"],
        sample_question["options"]
    )
    
    assert 'answer' in result
    assert 'question' in result
    assert 'options' in result
    assert 'processing_time' in result
    assert result['scenario'] == 'local'


def test_local_with_cot_scenario(mock_local_client, mock_remote_client, sample_question):
    """Test local + CoT scenario."""
    result = run_local_with_cot(
        mock_local_client,
        mock_remote_client,
        "test-local-model",
        "test-remote-model",
        sample_question["question"],
        sample_question["options"]
    )
    
    assert 'answer' in result
    assert 'cot_text' in result
    assert 'processing_time' in result
    assert result['scenario'] == 'local_cot'


def test_remote_only_scenario(mock_remote_client, sample_question):
    """Test remote-only scenario."""
    result = run_remote_only(
        mock_remote_client,
        "test-model",
        sample_question["question"],
        sample_question["options"]
    )
    
    assert 'answer' in result
    assert 'question' in result
    assert 'processing_time' in result
    assert result['scenario'] == 'remote'


