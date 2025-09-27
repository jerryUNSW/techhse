#!/usr/bin/env python3
"""
Test PhraseDP LangChain Agent on Medical Questions
==================================================

This script tests the enhanced PhraseDP agent with candidate distribution analysis
on medical questions to demonstrate reasoning steps and decision-making process.
"""

import json
import time
from phrasedp_langchain_agent import PhraseDPLangChainAgent

def test_medical_questions():
    """Test the agent on various medical questions with detailed reasoning output."""
    
    print("🏥 Testing PhraseDP LangChain Agent on Medical Questions")
    print("=" * 70)
    
    # Initialize the agent
    print("🤖 Initializing PhraseDP LangChain Agent...")
    agent = PhraseDPLangChainAgent()
    print("✅ Agent initialized successfully!\n")
    
    # Test cases with different complexity levels
    test_cases = [
        {
            "name": "Simple Medical Question",
            "question": "What is the first-line treatment for hypertension?",
            "options": ["A) ACE inhibitors", "B) Beta-blockers", "C) Diuretics", "D) Calcium channel blockers"],
            "context": None,
            "epsilon": 1.0,
            "expected_domain": "medical",
            "expected_complexity": "moderate"
        },
        {
            "name": "Complex Medical Scenario",
            "question": "A 65-year-old male with diabetes presents with chest pain. What is the most appropriate initial diagnostic test?",
            "options": ["A) ECG", "B) Chest X-ray", "C) Troponin levels", "D) Echocardiogram"],
            "context": "Patient has a history of type 2 diabetes mellitus and hypertension. Pain started 2 hours ago and is described as pressure-like.",
            "epsilon": 1.5,
            "expected_domain": "medical",
            "expected_complexity": "complex"
        },
        {
            "name": "Medical Terminology Heavy",
            "question": "Which medication is contraindicated in patients with pheochromocytoma?",
            "options": ["A) Metoprolol", "B) Labetalol", "C) Propranolol", "D) Atenolol"],
            "context": "Pheochromocytoma is a rare tumor of the adrenal medulla that secretes catecholamines.",
            "epsilon": 2.0,
            "expected_domain": "medical",
            "expected_complexity": "complex"
        },
        {
            "name": "Medical Question with PII",
            "question": "Dr. Smith at john.smith@hospital.com recommends (555) 123-4567 for follow-up. What is the standard protocol?",
            "options": ["A) Call within 24 hours", "B) Schedule follow-up in 1 week", "C) No follow-up needed", "D) Emergency consultation"],
            "context": None,
            "epsilon": 1.0,
            "expected_domain": "medical",
            "expected_complexity": "simple"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"📋 TEST {i}: {test_case['name']}")
        print(f"{'='*70}")
        
        print(f"🔍 Question: {test_case['question']}")
        if test_case['context']:
            print(f"📝 Context: {test_case['context']}")
        print(f"📊 Options: {test_case['options']}")
        print(f"🔒 Epsilon: {test_case['epsilon']}")
        print(f"🎯 Expected Domain: {test_case['expected_domain']}")
        print(f"📈 Expected Complexity: {test_case['expected_complexity']}")
        
        print(f"\n🤖 Starting Agent Processing...")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # Process with the agent
            result = agent.process(
                question=test_case['question'],
                context=test_case['context'],
                options=test_case['options'],
                epsilon=test_case['epsilon']
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"\n✅ Processing Complete!")
            print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
            print(f"🔄 Attempts: {result.attempts}")
            print(f"📊 Quality Score: {result.quality_score:.3f}")
            print(f"🎯 Success: {result.success}")
            
            # Display results
            print(f"\n📤 RESULTS:")
            print(f"Original Question: {test_case['question']}")
            print(f"Perturbed Question: {result.perturbed_question}")
            
            if result.perturbed_context:
                print(f"Perturbed Context: {result.perturbed_context}")
            
            if result.perturbed_options:
                print(f"Perturbed Options: {result.perturbed_options}")
            
            # Display detailed quality metrics
            if result.metadata.get('detailed_quality_metrics'):
                print(f"\n📊 DETAILED QUALITY METRICS:")
                quality_metrics = result.metadata['detailed_quality_metrics']
                
                if 'perturbation_quality' in quality_metrics:
                    print(f"  🔄 Perturbation Quality:")
                    for metric, score in quality_metrics['perturbation_quality'].items():
                        print(f"    - {metric.replace('_', ' ').title()}: {score:.3f}")
                
                if 'candidate_distribution' in quality_metrics:
                    print(f"  📈 Candidate Distribution:")
                    for metric, score in quality_metrics['candidate_distribution'].items():
                        print(f"    - {metric.replace('_', ' ').title()}: {score:.3f}")
                
                if 'improvement_suggestions' in quality_metrics:
                    print(f"  💡 Improvement Suggestions:")
                    for suggestion in quality_metrics['improvement_suggestions']:
                        print(f"    - {suggestion}")
            
            # Store result for summary
            results.append({
                'test_name': test_case['name'],
                'success': result.success,
                'quality_score': result.quality_score,
                'attempts': result.attempts,
                'processing_time': processing_time,
                'has_context': test_case['context'] is not None,
                'has_pii': '@' in test_case['question'] or '(' in test_case['question'],
                'epsilon': test_case['epsilon']
            })
            
        except Exception as e:
            print(f"❌ Error processing test case: {e}")
            results.append({
                'test_name': test_case['name'],
                'success': False,
                'error': str(e)
            })
        
        print(f"\n{'='*70}")
    
    # Summary report
    print(f"\n📊 SUMMARY REPORT")
    print(f"{'='*70}")
    
    successful_tests = [r for r in results if r.get('success', False)]
    failed_tests = [r for r in results if not r.get('success', False)]
    
    print(f"✅ Successful Tests: {len(successful_tests)}/{len(results)}")
    print(f"❌ Failed Tests: {len(failed_tests)}/{len(results)}")
    
    if successful_tests:
        avg_quality = sum(r['quality_score'] for r in successful_tests) / len(successful_tests)
        avg_attempts = sum(r['attempts'] for r in successful_tests) / len(successful_tests)
        avg_time = sum(r['processing_time'] for r in successful_tests) / len(successful_tests)
        
        print(f"\n📈 Performance Metrics:")
        print(f"  - Average Quality Score: {avg_quality:.3f}")
        print(f"  - Average Attempts: {avg_attempts:.1f}")
        print(f"  - Average Processing Time: {avg_time:.2f}s")
        
        print(f"\n📋 Test-by-Test Results:")
        for result in results:
            status = "✅" if result.get('success', False) else "❌"
            quality = f"{result.get('quality_score', 0):.3f}" if result.get('quality_score') else "N/A"
            attempts = result.get('attempts', 0)
            time_taken = f"{result.get('processing_time', 0):.2f}s" if result.get('processing_time') else "N/A"
            
            print(f"  {status} {result['test_name']}: Quality={quality}, Attempts={attempts}, Time={time_taken}")
    
    # Analysis insights
    print(f"\n🔍 ANALYSIS INSIGHTS:")
    
    # Context impact
    with_context = [r for r in successful_tests if r.get('has_context', False)]
    without_context = [r for r in successful_tests if not r.get('has_context', False)]
    
    if with_context and without_context:
        avg_quality_with = sum(r['quality_score'] for r in with_context) / len(with_context)
        avg_quality_without = sum(r['quality_score'] for r in without_context) / len(without_context)
        print(f"  - Context Impact: With context={avg_quality_with:.3f}, Without context={avg_quality_without:.3f}")
    
    # Epsilon impact
    epsilon_1 = [r for r in successful_tests if r.get('epsilon', 0) == 1.0]
    epsilon_1_5 = [r for r in successful_tests if r.get('epsilon', 0) == 1.5]
    epsilon_2 = [r for r in successful_tests if r.get('epsilon', 0) == 2.0]
    
    if epsilon_1:
        avg_quality_1 = sum(r['quality_score'] for r in epsilon_1) / len(epsilon_1)
        print(f"  - Epsilon 1.0: Average quality={avg_quality_1:.3f}")
    
    if epsilon_1_5:
        avg_quality_1_5 = sum(r['quality_score'] for r in epsilon_1_5) / len(epsilon_1_5)
        print(f"  - Epsilon 1.5: Average quality={avg_quality_1_5:.3f}")
    
    if epsilon_2:
        avg_quality_2 = sum(r['quality_score'] for r in epsilon_2) / len(epsilon_2)
        print(f"  - Epsilon 2.0: Average quality={avg_quality_2:.3f}")
    
    # PII handling
    pii_tests = [r for r in successful_tests if r.get('has_pii', False)]
    if pii_tests:
        print(f"  - PII Handling: {len(pii_tests)} tests with PII processed successfully")
    
    print(f"\n🎉 Testing Complete!")
    
    return results

def test_agent_reasoning_workflow():
    """Test the agent's reasoning workflow step by step."""
    
    print("\n🧠 TESTING AGENT REASONING WORKFLOW")
    print("=" * 70)
    
    agent = PhraseDPLangChainAgent()
    
    # Simple test case to observe reasoning
    test_question = "What is the treatment for diabetes?"
    test_options = ["A) Insulin", "B) Metformin", "C) Diet", "D) Exercise"]
    
    print(f"🔍 Test Question: {test_question}")
    print(f"📊 Options: {test_options}")
    print(f"🔒 Epsilon: 1.0")
    
    print(f"\n🤖 Agent Reasoning Process:")
    print("-" * 50)
    
    # This will show the agent's step-by-step reasoning
    result = agent.process(
        question=test_question,
        options=test_options,
        epsilon=1.0
    )
    
    print(f"\n📊 Final Result Analysis:")
    print(f"  - Quality Score: {result.quality_score:.3f}")
    print(f"  - Attempts: {result.attempts}")
    print(f"  - Success: {result.success}")
    
    if result.metadata.get('detailed_quality_metrics'):
        print(f"\n🔍 Detailed Reasoning Analysis:")
        quality_metrics = result.metadata['detailed_quality_metrics']
        
        if 'perturbation_quality' in quality_metrics:
            print(f"  Perturbation Quality Breakdown:")
            for metric, score in quality_metrics['perturbation_quality'].items():
                print(f"    - {metric}: {score:.3f}")
        
        if 'candidate_distribution' in quality_metrics:
            print(f"  Candidate Distribution Analysis:")
            for metric, score in quality_metrics['candidate_distribution'].items():
                print(f"    - {metric}: {score:.3f}")

if __name__ == "__main__":
    # Run comprehensive medical question tests
    results = test_medical_questions()
    
    # Run detailed reasoning workflow test
    test_agent_reasoning_workflow()
    
    print(f"\n🏁 All tests completed!")
    print(f"📊 Total tests run: {len(results)}")
    print(f"✅ Successful: {len([r for r in results if r.get('success', False)])}")
    print(f"❌ Failed: {len([r for r in results if not r.get('success', False)])}")


