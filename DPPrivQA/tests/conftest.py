"""
pytest fixtures and shared test utilities.
"""

import pytest
import sqlite3
import tempfile
import os
from unittest.mock import Mock, MagicMock
from openai import OpenAI

from dpprivqa.database.writer import ExperimentDBWriter


@pytest.fixture
def mock_local_client():
    """Mock local LLM client."""
    client = Mock(spec=OpenAI)
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="A"))]
    client.chat.completions.create.return_value = mock_response
    return client


@pytest.fixture
def mock_remote_client():
    """Mock remote LLM client."""
    client = Mock(spec=OpenAI)
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Let me think... The answer is A."))]
    client.chat.completions.create.return_value = mock_response
    return client


@pytest.fixture
def sample_question():
    """Sample question for testing."""
    return {
        "question": "What is the treatment for hypertension?",
        "options": {"A": "Aspirin", "B": "Lisinopril", "C": "Ibuprofen", "D": "Paracetamol"},
        "answer_idx": "B"
    }


@pytest.fixture
def temp_db():
    """Temporary database for testing."""
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    yield db_path
    os.close(db_fd)
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def db_writer(temp_db):
    """Database writer with temporary database."""
    return ExperimentDBWriter(temp_db)


