#!/usr/bin/env python3
"""
Comprehensive tests for the PhraseDP Agent.

This test suite validates:
- Dataset analysis capabilities
- Question type detection
- Context summarization
- PII detection and replacement
- Text perturbation
- Metadata generation
- End-to-end processing
"""

import json
import pytest
from phrasedp_agent import PhraseDPAgent, DatasetInfo, QuestionType, PerturbationResult


class TestPhraseDPAgent:
    """Test suite for PhraseDP Agent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = PhraseDPAgent()
    
    def test_dataset_analysis_medical(self):
        """Test medical dataset analysis."""
        question = "What is the treatment for hypertension?"
        context = "Hypertension is a common cardiovascular condition..."
        
        result = self.agent.analyze_dataset(question, context)
        
        assert isinstance(result, DatasetInfo)
        assert result.dataset_type == "medical"
        assert result.domain in ["clinical", "medical", "healthcare"]
        assert result.privacy_level in ["high", "medium", "low"]
        assert len(result.key_terminology) > 0
    
    def test_dataset_analysis_general(self):
        """Test general dataset analysis."""
        question = "What is the capital of France?"
        context = "France is a country in Europe..."
        
        result = self.agent.analyze_dataset(question, context)
        
        assert isinstance(result, DatasetInfo)
        assert result.dataset_type in ["general", "geography", "academic"]
        assert result.domain in ["general", "geography", "academic"]
    
    def test_question_type_detection_multiple_choice(self):
        """Test multiple choice question detection."""
        question = "What is the treatment for hypertension?"
        options = ["A) Medication", "B) Surgery", "C) Diet", "D) Exercise"]
        
        result = self.agent.detect_question_type(question, options)
        
        assert isinstance(result, QuestionType)
        assert result.question_type == "multiple_choice"
        assert result.has_options is True
        assert result.options_count == 4
    
    def test_question_type_detection_open_ended(self):
        """Test open-ended question detection."""
        question = "Explain the process of photosynthesis."
        
        result = self.agent.detect_question_type(question)
        
        assert isinstance(result, QuestionType)
        assert result.question_type == "open_ended"
        assert result.has_options is False
        assert result.options_count == 0
    
    def test_context_summarization(self):
        """Test context summarization."""
        context = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially criticized by some of France's leading artists and intellectuals for its design, but it has become a global cultural icon of France and one of the most recognizable structures in the world."
        
        dataset_info = DatasetInfo(
            dataset_type="general",
            domain="geography",
            complexity="moderate",
            privacy_level="low",
            key_terminology=["Eiffel Tower", "Paris", "France", "Gustave Eiffel"]
        )
        
        result = self.agent.summarize_context(context, dataset_info, max_length=50)
        
        assert isinstance(result, str)
        assert len(result.split()) <= 50
        assert "Eiffel Tower" in result or "Paris" in result or "France" in result
    
    def test_pii_detection_and_replacement(self):
        """Test PII detection and replacement."""
        text = "Contact Dr. John Smith at john.smith@hospital.com or call (555) 123-4567. Visit us at 123 Main Street, New York, NY 10001."
        
        result = self.agent.detect_and_replace_pii(text)
        
        assert isinstance(result, str)
        assert "[email address]" in result
        assert "[phone number]" in result
        assert "[person name]" in result
        assert "[street address]" in result
        assert "john.smith@hospital.com" not in result
        assert "(555) 123-4567" not in result
    
    def test_candidate_generation(self):
        """Test candidate generation."""
        text = "What is the treatment for hypertension?"
        dataset_info = DatasetInfo(
            dataset_type="medical",
            domain="clinical",
            complexity="moderate",
            privacy_level="high",
            key_terminology=["hypertension", "treatment", "medication"]
        )
        
        result = self.agent.generate_candidates(text, dataset_info, epsilon=1.0)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(candidate, str) for candidate in result)
    
    def test_text_perturbation(self):
        """Test text perturbation."""
        text = "What is the treatment for hypertension?"
        candidates = ["What is the therapy for high blood pressure?", "What is the medication for hypertension?", "What is the cure for elevated blood pressure?"]
        dataset_info = DatasetInfo(
            dataset_type="medical",
            domain="clinical",
            complexity="moderate",
            privacy_level="high",
            key_terminology=["hypertension", "treatment", "medication"]
        )
        
        result = self.agent.perturb_text(text, candidates, dataset_info, epsilon=1.0)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert result != text  # Should be different from original
    
    def test_batch_perturb_options(self):
        """Test batch perturbation of options."""
        options = ["A) Medication", "B) Surgery", "C) Diet", "D) Exercise"]
        candidates = ["therapy", "treatment", "cure", "remedy"]
        dataset_info = DatasetInfo(
            dataset_type="medical",
            domain="clinical",
            complexity="moderate",
            privacy_level="high",
            key_terminology=["medication", "surgery", "diet", "exercise"]
        )
        
        result = self.agent._batch_perturb_options(options, candidates, dataset_info)
        
        assert isinstance(result, str)
        assert "Option A:" in result
        assert "Option B:" in result
        assert "Option C:" in result
        assert "Option D:" in result
    
    def test_metadata_generation(self):
        """Test metadata generation."""
        dataset_info = DatasetInfo(
            dataset_type="medical",
            domain="clinical",
            complexity="moderate",
            privacy_level="high",
            key_terminology=["hypertension", "treatment", "medication"]
        )
        question_type = QuestionType(
            question_type="multiple_choice",
            has_options=True,
            answer_format="single_letter",
            options_count=4
        )
        
        result = self.agent._generate_metadata(dataset_info, question_type, epsilon=1.0)
        
        assert isinstance(result, dict)
        assert "dataset_type" in result
        assert "domain" in result
        assert "question_type" in result
        assert "privacy_level" in result
        assert "epsilon" in result
        assert "perturbation_strategy" in result
        assert result["dataset_type"] == "medical"
        assert result["question_type"] == "multiple_choice"
        assert result["epsilon"] == 1.0
    
    def test_end_to_end_medical_multiple_choice(self):
        """Test end-to-end processing for medical multiple choice."""
        question = "What is the first-line treatment for hypertension?"
        options = ["A) ACE inhibitors", "B) Beta-blockers", "C) Diuretics", "D) Calcium channel blockers"]
        
        result = self.agent.process(question, options=options, epsilon=1.0)
        
        assert isinstance(result, PerturbationResult)
        assert isinstance(result.perturbed_question, str)
        assert isinstance(result.perturbed_options, str)
        assert isinstance(result.metadata, dict)
        assert result.perturbed_context is None  # No context provided
        assert result.metadata["question_type"] == "multiple_choice"
        assert result.metadata["has_options"] is True
    
    def test_end_to_end_with_context(self):
        """Test end-to-end processing with context."""
        question = "Who designed the Eiffel Tower?"
        context = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower."
        
        result = self.agent.process(question, context=context, epsilon=2.0)
        
        assert isinstance(result, PerturbationResult)
        assert isinstance(result.perturbed_question, str)
        assert isinstance(result.perturbed_context, str)
        assert isinstance(result.metadata, dict)
        assert result.perturbed_options is None  # No options provided
        assert result.metadata["has_context"] is True
        assert result.metadata["epsilon"] == 2.0
    
    def test_end_to_end_with_pii(self):
        """Test end-to-end processing with PII."""
        question = "Contact Dr. Smith at john.smith@hospital.com or call (555) 123-4567 for appointments."
        
        result = self.agent.process(question, epsilon=1.5)
        
        assert isinstance(result, PerturbationResult)
        assert isinstance(result.perturbed_question, str)
        assert isinstance(result.metadata, dict)
        # Check that PII is replaced
        assert "[email address]" in result.perturbed_question or "[phone number]" in result.perturbed_question
        assert result.metadata["epsilon"] == 1.5
    
    def test_candidate_count_calculation(self):
        """Test candidate count calculation based on epsilon."""
        assert self.agent._calculate_candidate_count(0.5) == 10
        assert self.agent._calculate_candidate_count(1.0) == 10
        assert self.agent._calculate_candidate_count(1.5) == 20
        assert self.agent._calculate_candidate_count(2.0) == 20
        assert self.agent._calculate_candidate_count(2.5) == 30
        assert self.agent._calculate_candidate_count(3.0) == 30
        assert self.agent._calculate_candidate_count(4.0) == 40


def run_manual_tests():
    """Run manual tests that require actual model calls."""
    print("🧪 Running Manual Tests for PhraseDP Agent")
    print("=" * 60)
    
    agent = PhraseDPAgent()
    
    # Test scenarios
    test_cases = [
        {
            "name": "Medical Multiple Choice",
            "question": "What is the first-line treatment for hypertension?",
            "options": ["A) ACE inhibitors", "B) Beta-blockers", "C) Diuretics", "D) Calcium channel blockers"],
            "context": None,
            "epsilon": 1.0
        },
        {
            "name": "General Question with Context",
            "question": "Who designed the Eiffel Tower?",
            "options": None,
            "context": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
            "epsilon": 2.0
        },
        {
            "name": "Text with PII",
            "question": "Contact Dr. Smith at john.smith@hospital.com or call (555) 123-4567 for appointments.",
            "options": None,
            "context": None,
            "epsilon": 1.5
        },
        {
            "name": "Legal Question",
            "question": "What constitutes a breach of contract?",
            "options": None,
            "context": "A breach of contract occurs when one party fails to fulfill their obligations under the terms of the agreement.",
            "epsilon": 1.0
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print("-" * 40)
        
        try:
            result = agent.process(
                question=test_case["question"],
                context=test_case["context"],
                options=test_case["options"],
                epsilon=test_case["epsilon"]
            )
            
            print(f"✅ Success!")
            print(f"Original Question: {test_case['question']}")
            print(f"Perturbed Question: {result.perturbed_question}")
            
            if result.perturbed_context:
                print(f"Perturbed Context: {result.perturbed_context}")
            
            if result.perturbed_options:
                print(f"Perturbed Options: {result.perturbed_options}")
            
            print(f"Metadata: {json.dumps(result.metadata, indent=2)}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n🎉 Manual tests completed!")


if __name__ == "__main__":
    # Run manual tests (requires actual model calls)
    run_manual_tests()
    
    # Uncomment to run unit tests (requires pytest)
    # pytest.main([__file__, "-v"])







