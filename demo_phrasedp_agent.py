#!/usr/bin/env python3
"""
Demo script for PhraseDP Agent.

This script demonstrates the capabilities of the PhraseDP Agent with various
real-world scenarios including medical questions, general knowledge, and PII handling.
"""

import json
from phrasedp_agent import PhraseDPAgent


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")


def print_result(result, original_question, original_context=None, original_options=None):
    """Print formatted results."""
    print(f"\n📝 Original Question: {original_question}")
    if original_context:
        print(f"📄 Original Context: {original_context}")
    if original_options:
        print(f"📋 Original Options: {original_options}")
    
    print(f"\n🔄 Perturbed Question: {result.perturbed_question}")
    if result.perturbed_context:
        print(f"🔄 Perturbed Context: {result.perturbed_context}")
    if result.perturbed_options:
        print(f"🔄 Perturbed Options: {result.perturbed_options}")
    
    print(f"\n📊 Metadata:")
    print(json.dumps(result.metadata, indent=2))


def demo_medical_scenarios():
    """Demonstrate medical question scenarios."""
    print_section("Medical Question Scenarios")
    
    agent = PhraseDPAgent()
    
    # Scenario 1: Medical multiple choice
    print("\n🏥 Scenario 1: Medical Multiple Choice")
    result1 = agent.process(
        question="What is the first-line treatment for hypertension?",
        options=["A) ACE inhibitors", "B) Beta-blockers", "C) Diuretics", "D) Calcium channel blockers"],
        epsilon=1.0
    )
    print_result(
        result1,
        "What is the first-line treatment for hypertension?",
        original_options=["A) ACE inhibitors", "B) Beta-blockers", "C) Diuretics", "D) Calcium channel blockers"]
    )
    
    # Scenario 2: Medical question with context
    print("\n🏥 Scenario 2: Medical Question with Context")
    medical_context = "A 65-year-old male presents with chest pain, shortness of breath, and elevated blood pressure. Physical examination reveals signs of heart failure. Laboratory tests show elevated cardiac enzymes."
    result2 = agent.process(
        question="What is the most likely diagnosis?",
        context=medical_context,
        epsilon=2.0
    )
    print_result(
        result2,
        "What is the most likely diagnosis?",
        original_context=medical_context
    )


def demo_general_knowledge():
    """Demonstrate general knowledge scenarios."""
    print_section("General Knowledge Scenarios")
    
    agent = PhraseDPAgent()
    
    # Scenario 1: Geography with context
    print("\n🌍 Scenario 1: Geography with Context")
    geography_context = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair."
    result1 = agent.process(
        question="Who designed the Eiffel Tower?",
        context=geography_context,
        epsilon=1.5
    )
    print_result(
        result1,
        "Who designed the Eiffel Tower?",
        original_context=geography_context
    )
    
    # Scenario 2: History question
    print("\n📚 Scenario 2: History Question")
    result2 = agent.process(
        question="When did World War II end?",
        epsilon=1.0
    )
    print_result(result2, "When did World War II end?")


def demo_pii_handling():
    """Demonstrate PII handling scenarios."""
    print_section("PII Handling Scenarios")
    
    agent = PhraseDPAgent()
    
    # Scenario 1: Contact information
    print("\n🔒 Scenario 1: Contact Information")
    result1 = agent.process(
        question="Contact Dr. Smith at john.smith@hospital.com or call (555) 123-4567 for appointments.",
        epsilon=1.0
    )
    print_result(result1, "Contact Dr. Smith at john.smith@hospital.com or call (555) 123-4567 for appointments.")
    
    # Scenario 2: Personal information
    print("\n🔒 Scenario 2: Personal Information")
    result2 = agent.process(
        question="My name is John Doe, I live at 123 Main Street, New York, NY 10001, and my SSN is 123-45-6789.",
        epsilon=2.0
    )
    print_result(result2, "My name is John Doe, I live at 123 Main Street, New York, NY 10001, and my SSN is 123-45-6789.")


def demo_legal_scenarios():
    """Demonstrate legal question scenarios."""
    print_section("Legal Question Scenarios")
    
    agent = PhraseDPAgent()
    
    # Scenario 1: Legal question with context
    print("\n⚖️ Scenario 1: Legal Question with Context")
    legal_context = "A breach of contract occurs when one party fails to fulfill their obligations under the terms of the agreement. This can include failure to deliver goods, provide services, or meet deadlines as specified in the contract."
    result1 = agent.process(
        question="What constitutes a breach of contract?",
        context=legal_context,
        epsilon=1.0
    )
    print_result(
        result1,
        "What constitutes a breach of contract?",
        original_context=legal_context
    )


def demo_epsilon_comparison():
    """Demonstrate how epsilon affects perturbation."""
    print_section("Epsilon Comparison (Privacy vs Utility)")
    
    agent = PhraseDPAgent()
    question = "What is the treatment for diabetes?"
    
    # Test different epsilon values
    epsilon_values = [0.5, 1.0, 2.0, 3.0]
    
    for epsilon in epsilon_values:
        print(f"\n🎯 Epsilon = {epsilon}")
        result = agent.process(question, epsilon=epsilon)
        print(f"Original: {question}")
        print(f"Perturbed: {result.perturbed_question}")
        print(f"Candidate Pool Size: {result.metadata['candidate_pool_size']}")


def main():
    """Run all demos."""
    print("🚀 PhraseDP Agent Demo")
    print("Intelligent Privacy-Preserving Text Sanitization")
    
    try:
        # Run all demo scenarios
        demo_medical_scenarios()
        demo_general_knowledge()
        demo_pii_handling()
        demo_legal_scenarios()
        demo_epsilon_comparison()
        
        print_section("Demo Complete")
        print("✅ All demos completed successfully!")
        print("\nThe PhraseDP Agent demonstrates:")
        print("• Intelligent dataset analysis")
        print("• Question type detection")
        print("• Context summarization")
        print("• PII detection and replacement")
        print("• Adaptive text perturbation")
        print("• Rich metadata generation")
        print("• Privacy-utility trade-off control")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("Make sure the local model is available and configured correctly.")


if __name__ == "__main__":
    main()







