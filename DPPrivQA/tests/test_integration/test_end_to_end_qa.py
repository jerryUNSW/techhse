"""
End-to-end QA pipeline tests.
"""

import pytest
from dpprivqa.qa.scenarios import run_local_only, run_local_with_cot, run_remote_only
from dpprivqa.qa.dpprivqa import run_dpprivqa
from dpprivqa.qa.prompts import check_mcq_correctness


def test_end_to_end_qa_pipeline(mock_local_client, mock_remote_client, sample_question):
    """Test complete QA pipeline from question to answer."""
    # Run all scenarios
    local_result = run_local_only(
        mock_local_client, "test-model",
        sample_question["question"], sample_question["options"]
    )
    
    cot_result = run_local_with_cot(
        mock_local_client, mock_remote_client,
        "test-local", "test-remote",
        sample_question["question"], sample_question["options"]
    )
    
    remote_result = run_remote_only(
        mock_remote_client, "test-model",
        sample_question["question"], sample_question["options"]
    )
    
    # All should produce valid results
    for result in [local_result, cot_result, remote_result]:
        assert 'answer' in result
        assert 'question' in result
        assert 'options' in result


@pytest.mark.skip(reason="Requires actual API keys and models")
def test_dpprivqa_pipeline_integration(mock_local_client, mock_remote_client, sample_question):
    """Test DPPrivQA pipeline integration (requires actual models)."""
    # This test would require actual API keys and models
    # Marked as skip for now
    pass


