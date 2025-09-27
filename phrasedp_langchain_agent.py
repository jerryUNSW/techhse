#!/usr/bin/env python3
"""
PhraseDP Agent with LangChain - Intelligent Privacy-Preserving Text Sanitization

A proper agent implementation using LangChain with:
- Tool calling capabilities
- Reasoning and decision making
- Quality assessment and retry logic
- Adaptive behavior based on results
"""

import os
import json
import re
import yaml
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool, StructuredTool
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler
from pydantic import BaseModel, Field

import utils

# Load environment variables
load_dotenv()

@dataclass
class PerturbationResult:
    """Result of the perturbation process."""
    perturbed_question: str
    perturbed_context: Optional[str]
    perturbed_options: Optional[str]
    metadata: Dict[str, Any]
    quality_score: float
    attempts: int
    success: bool

class PhraseDPLangChainAgent:
    """
    Intelligent PhraseDP agent using LangChain with proper reasoning capabilities.
    """
    
    def __init__(self, local_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        """
        Initialize the PhraseDP agent with LangChain.
        
        Args:
            local_model: Name of the local model to use via Nebius
        """
        self.local_model = local_model
        self.nebius_client = utils.get_nebius_client()
        
        # Load configuration
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        
        # Initialize LLM - Use GPT-4o for agent reasoning (not Llama 8B)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Use GPT-4o for agent reasoning
            temperature=0.1,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Initialize memory for conversation context
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Define tools
        self.tools = self._create_tools()
        
        # Create agent
        self.agent = self._create_agent()
        
        # Create agent executor with retry capabilities
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=5,  # Allow up to 5 attempts
            early_stopping_method="force",
            handle_parsing_errors=True
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent."""

        return [
            Tool(
                name="analyze_dataset",
                description="Analyze the dataset type, domain, complexity, and privacy requirements. Input: question text. Output: JSON with dataset analysis.",
                func=self._analyze_dataset_tool
            ),
            Tool(
                name="detect_question_type",
                description="Detect the question type and format. Input: question text. Output: JSON with question type analysis.",
                func=self._detect_question_type_tool
            ),
            Tool(
                name="generate_candidates",
                description="Generate semantic candidates for text perturbation. Input: text to perturb. Output: list of candidate replacements.",
                func=self._generate_candidates_tool
            ),
            Tool(
                name="perturb_text",
                description="Apply privacy-preserving perturbation to text. Input: text, candidates, dataset_info, epsilon. Output: perturbed text.",
                func=self._perturb_text_tool
            ),
            Tool(
                name="assess_quality",
                description="Assess the quality of perturbation and decide if retry is needed. Input: original_text, perturbed_text, dataset_info. Output: JSON with quality score and retry decision.",
                func=self._assess_quality_tool
            ),
            Tool(
                name="detect_pii",
                description="Detect and replace PII in text. Input: text. Output: text with PII replaced by placeholders.",
                func=self._detect_pii_tool
            ),
            Tool(
                name="batch_perturb_options",
                description="Perturb multiple choice options in batch. Input: options_list, candidates, dataset_info. Output: perturbed options as semicolon-separated string.",
                func=self._batch_perturb_options_tool
            ),
            Tool(
                name="analyze_candidate_distribution",
                description="Analyze the distribution and quality of generated candidates. Input: JSON string with 'candidates' and 'original_text' fields. Example: '{\"candidates\": [\"c1\", \"c2\"], \"original_text\": \"text\"}'. Output: JSON with distribution analysis.",
                func=self._analyze_candidate_distribution_tool
            ),
            Tool(
                name="generate_remote_cot",
                description="Generate Chain-of-Thought from remote model using perturbed question and options. Input: JSON string with 'perturbed_question' and 'perturbed_options' fields. Example: '{\"perturbed_question\": \"text\", \"perturbed_options\": \"text\"}'. Output: CoT text.",
                func=self._generate_remote_cot_tool
            ),
            Tool(
                name="answer_with_local_cot",
                description="Use local model with original question/options + private CoT to answer. Input: JSON string with 'original_question', 'original_options', 'cot_text', 'correct_answer' fields. Example: '{\"original_question\": \"text\", \"original_options\": [\"A\", \"B\"], \"cot_text\": \"text\", \"correct_answer\": \"A\"}'. Output: final answer with correctness check.",
                func=self._answer_with_local_cot_tool
            ),
            Tool(
                name="probabilistic_sampling",
                description="Apply exponential mechanism for probabilistic sampling from candidates. Input: JSON string with 'original_text', 'candidates', 'epsilon' fields. Example: '{\"original_text\": \"text\", \"candidates\": [\"c1\", \"c2\"], \"epsilon\": 1.0}'. Output: selected candidate.",
                func=self._probabilistic_sampling_tool
            ),
            Tool(
                name="generate_local_cot",
                description="Generate Chain-of-Thought from local model using original question/options. Input: JSON string with 'original_question' and 'original_options' fields. Example: '{\"original_question\": \"text\", \"original_options\": [\"A\", \"B\"]}'. Output: CoT text.",
                func=self._generate_local_cot_tool
            )
        ]
    
    def _create_agent(self):
        """Create the LangChain agent."""
        system_prompt = """You are PhraseDP, an intelligent privacy-preserving text sanitization agent.

Your capabilities:
1. Analyze dataset types and domains (medical, legal, general, academic, etc.)
2. Detect question formats (multiple choice, fill-in-blank, open-ended, true/false)
3. Generate sentence-level paraphrases for PhraseDP perturbation
4. Analyze candidate distribution and quality across similarity bands
5. Apply PhraseDP perturbation by selecting appropriate candidates
6. Detect and handle PII (emails, phones, addresses, names)
7. Assess both perturbation quality AND candidate distribution
8. Decide on retry strategies based on comprehensive analysis
9. Adapt your approach based on results

      CRITICAL WORKFLOW (PhraseDP Approach - 3.1.2.new):
      YOU MUST USE THE TOOLS TO COMPLETE EACH STEP WITH PROPER JSON INPUT:
      1. Use analyze_dataset tool with JSON: {{"question": "text", "context": "text", "options": ["A", "B", "C", "D"]}}
      2. For QUESTION perturbation:
         a. Use generate_candidates tool with JSON: {{"text": "question_text", "epsilon": 1.0}}
         b. Use analyze_candidate_distribution tool with JSON: {{"candidates": ["c1", "c2", ...], "original_text": "text"}}
         c. If any band is missing candidates, REGENERATE using generate_candidates tool
         d. Repeat until you have 10 candidates satisfying band distribution
         e. Use probabilistic_sampling tool with JSON: {{"original_text": "text", "candidates": ["c1", "c2", ...], "epsilon": 1.0}}
      3. For COMBINED OPTIONS perturbation:
         a. Use generate_candidates tool with JSON: {{"text": "combined_options_text", "epsilon": 1.0}}
         b. Use analyze_candidate_distribution tool with JSON: {{"candidates": ["c1", "c2", ...], "original_text": "text"}}
         c. If any band is missing candidates, REGENERATE using generate_candidates tool
         d. Repeat until you have 10 candidates satisfying band distribution
         e. Use probabilistic_sampling tool with JSON: {{"original_text": "text", "candidates": ["c1", "c2", ...], "epsilon": 1.0}}
      4. Use generate_remote_cot tool with JSON: {{"perturbed_question": "text", "perturbed_options": "text"}}
      5. Use generate_local_cot tool with JSON: {{"original_question": "text", "original_options": ["A", "B", "C", "D"]}}
      6. Use answer_with_local_cot tool with JSON: {{"original_question": "text", "original_options": ["A", "B", "C", "D"], "cot_text": "text", "correct_answer": "A"}}
      7. The local model answers based on its own ability using the private CoT (it does NOT know the correct answer)
      8. The tool checks if the local model answered correctly by comparing with the actual answer
      9. Return final answer with privacy-preserving CoT and correctness result

      PHRASEDP SPECIFICS (3.1.2.new):
      - PhraseDP is a DIFFERENTIAL PRIVACY mechanism that uses EXPONENTIAL MECHANISM for candidate selection
      - The exponential mechanism selects candidates probabilistically based on similarity scores
      - Selection probability ∝ exp(-epsilon * distance) where distance = 1 - similarity
      - Higher epsilon = more privacy = higher probability for dissimilar candidates
      - Similarities are COMPUTED BY TOOLS using SBERT embeddings, NOT estimated by the agent
      - Generate candidates using LOCAL model (not remote model)
      - Generate exactly 10 candidates across 5 similarity bands (2 per band)
      - Band 1 (0.0-0.2): Very different expression, same meaning
      - Band 2 (0.2-0.4): Different wording, preserved meaning  
      - Band 3 (0.4-0.6): Moderate changes, core meaning intact
      - Band 4 (0.6-0.8): Minor changes, very similar
      - Band 5 (0.8-1.0): Minimal changes, nearly identical
      - CRITICAL: Do NOT stop until you have 10 candidates satisfying band distribution
      - If any band is missing candidates, REGENERATE using generate_candidates tool
      - Use analyze_candidate_distribution tool to verify band coverage after each generation
      - Perturb QUESTION using PhraseDP (sentence-level paraphrase with local model)
      - Perturb OPTIONS using batch PhraseDP (combine all options, perturb as one text with local model)
      - Generate CoT from REMOTE model using perturbed question (NO options sent)
      - Use LOCAL model with original question/options + private CoT
      - The local model answers based on its own reasoning ability (it does NOT know the correct answer)
      - Use actual PhraseDP implementation with SBERT embeddings and exponential mechanism
      - Preserve medical terminology and question structure
      - Maintain semantic meaning while providing differential privacy protection

QUALITY THRESHOLDS:
- Perturbation quality: >= 0.7
- Candidate distribution: >= 0.7
- Overall quality: >= 0.7

      Always reason step-by-step and provide clear explanations for your decisions.
      
      IMPORTANT: You MUST complete the full 3.1.2.new pipeline by using ALL the required tools:
      - perturb_text tool for question perturbation
      - batch_perturb_options tool for options perturbation  
      - generate_remote_cot tool for CoT generation
      - answer_with_local_cot tool for final answer
      
      Do not stop after perturbation - you must get the final answer from the local model using the private CoT."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        return create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
    
    def _call_local_model(self, prompt: str, max_tokens: int = 512) -> str:
        """Call the local model with the given prompt."""
        try:
            response = self.nebius_client.chat.completions.create(
                model=self.local_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling local model: {e}")
            return ""
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from the local model."""
        try:
            # Try to parse direct JSON first
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(response)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response: {response}")
            return {}

    def _parse_candidate_list(self, response: str) -> List[str]:
        """Parse candidate list from local model response, handling various formats."""
        try:
            # Look for JSON array in markdown code blocks
            json_match = re.search(r'```[^\n]*\n(\[.*?\])\n```', response, re.DOTALL)
            if json_match:
                candidates_json = json_match.group(1)
                candidates = json.loads(candidates_json)
                if isinstance(candidates, list) and len(candidates) == 10:
                    return candidates

            # Look for JSON array with objects containing "text" field
            obj_match = re.search(r'```[^\n]*\n(\[\s*\{.*?\}\s*\])\n```', response, re.DOTALL)
            if obj_match:
                candidates_json = obj_match.group(1)
                candidates_obj = json.loads(candidates_json)
                if isinstance(candidates_obj, list):
                    candidates = [item.get('text', str(item)) for item in candidates_obj]
                    if len(candidates) == 10:
                        return candidates

            # Look for plain JSON array
            array_match = re.search(r'\[\s*"[^"]+"(?:\s*,\s*"[^"]+")*\s*\]', response)
            if array_match:
                candidates = json.loads(array_match.group())
                if isinstance(candidates, list) and len(candidates) == 10:
                    return candidates

            # Fallback: try to extract quoted strings
            quotes = re.findall(r'"([^"]+)"', response)
            if len(quotes) >= 10:
                return quotes[:10]

            print(f"Could not extract 10 candidates from response: {response[:200]}...")
            return []

        except Exception as e:
            print(f"Error parsing candidate list: {e}")
            return []
    
    # Tool implementations
    def _analyze_dataset_tool(self, question: str) -> str:
        """Tool to analyze dataset type and domain."""
        context = ""
        
        prompt = f"""Analyze this text and determine:
1. Dataset type (medical, legal, general, academic, technical, etc.)
2. Domain-specific terminology present
3. Complexity level (simple, moderate, complex)
4. Privacy requirements (low, medium, high)
5. Key terminology that should be preserved

Question: {question}
Context: {context if context else "None"}

Respond in JSON format:
{{
    "dataset_type": "medical",
    "domain": "clinical_vignettes", 
    "complexity": "moderate",
    "privacy_level": "high",
    "key_terminology": ["hypertension", "medication", "treatment"]
}}"""
        
        response = self._call_local_model(prompt)
        return response
    
    def _detect_question_type_tool(self, question: str) -> str:
        """Tool to detect question type and format."""
        options = []
        
        prompt = f"""Analyze this question and determine:
1. Question type (multiple_choice, fill_in_blank, open_ended, true_false)
2. Whether options are provided
3. Expected answer format
4. Number of options (if applicable)

Question: {question}
Options: {options if options else "None"}

Respond in JSON format:
{{
    "question_type": "multiple_choice",
    "has_options": true,
    "answer_format": "single_letter",
    "options_count": 4
}}"""
        
        response = self._call_local_model(prompt)
        return response
    
    def _generate_candidates_tool(self, text: str) -> str:
        """Tool to generate balanced candidates across similarity bands in one API call."""
        dataset_info = {}
        epsilon = 1.0
        
        candidate_count = self._calculate_candidate_count(epsilon)
        
        prompt = f"""Generate exactly 10 sentence-level paraphrases distributed across 5 similarity bands (2 candidates per band) for PhraseDP.

Original text: {text}
Domain: {dataset_info.get('domain', 'general')}
Privacy level: {dataset_info.get('privacy_level', 'medium')}
Key terminology to preserve: {dataset_info.get('key_terminology', [])}
Epsilon: {epsilon}

CRITICAL: Generate exactly 2 candidates per similarity band:
- Band 1 (0.0-0.2): Very different expression, same meaning (2 candidates)
- Band 2 (0.2-0.4): Different wording, preserved meaning (2 candidates)  
- Band 3 (0.4-0.6): Moderate changes, core meaning intact (2 candidates)
- Band 4 (0.6-0.8): Minor changes, very similar (2 candidates)
- Band 5 (0.8-1.0): Minimal changes, nearly identical (2 candidates)

Requirements:
- Generate exactly 10 candidates total
- Exactly 2 candidates per similarity band
- Each candidate is a complete sentence-level paraphrase
- Preserve key terminology: {dataset_info.get('key_terminology', [])}
- Maintain semantic meaning while varying expression
- Ensure diversity within each band

Respond with candidates as a JSON array:
["candidate1", "candidate2", "candidate3", "candidate4", "candidate5", "candidate6", "candidate7", "candidate8", "candidate9", "candidate10"]"""
        
        response = self._call_local_model(prompt, max_tokens=800)
        
        # Parse and display the candidates clearly
        try:
            candidates = self._parse_candidate_list(response)
            if candidates and len(candidates) == 10:
                return f"""Generated {len(candidates)} candidates for PhraseDP perturbation:

Original text: {text}
Epsilon: {epsilon}
Number of candidates: {len(candidates)}

Candidates:
{chr(10).join([f"{i+1}. {candidate}" for i, candidate in enumerate(candidates)])}

Use analyze_candidate_distribution tool to check band coverage."""
            else:
                return f"""Error: Expected 10 candidates, got {len(candidates) if candidates else 0}

Raw response: {response}"""
        except Exception as e:
            return f"""Error parsing candidates: {str(e)}

Raw response: {response}"""
    
    def _perturb_text_tool(self, input_str: str) -> str:
        """Tool to apply PhraseDP perturbation using actual implementation with embeddings."""
        try:
            data = json.loads(input_str)
            text = data.get("text", "")
            candidates = data.get("candidates", [])
            dataset_info = data.get("dataset_info", {})
            epsilon = data.get("epsilon", 1.0)
        except:
            return "Error: Invalid input format"
        
        if not candidates:
            return text
        
        try:
            # Use actual PhraseDP implementation with embeddings
            from sentence_transformers import SentenceTransformer
            from dp_sanitizer import get_embedding, differentially_private_replacement
            
            # Load SBERT model for similarity computation
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Precompute embeddings for all candidates
            candidate_embeddings = {sent: get_embedding(sbert_model, sent).cpu().numpy() for sent in candidates}
            
            # Apply exponential mechanism using actual PhraseDP implementation
            selected_candidate = differentially_private_replacement(
                target_phrase=text,
                epsilon=epsilon,
                candidate_phrases=candidates,
                candidate_embeddings=candidate_embeddings,
                sbert_model=sbert_model
            )
            
            # Calculate actual similarities for reporting
            from sklearn.metrics.pairwise import cosine_similarity
            target_embedding = get_embedding(sbert_model, text).cpu().numpy()
            if target_embedding.ndim == 1:
                target_embedding = target_embedding.reshape(1, -1)
            
            candidate_embeddings_matrix = np.vstack([candidate_embeddings[phrase] for phrase in candidates])
            similarities = cosine_similarity(target_embedding, candidate_embeddings_matrix)[0]
            
            return f"""Applied PhraseDP exponential mechanism with epsilon={epsilon} using actual embeddings:

Original text: {text}
Number of candidates: {len(candidates)}
Actual similarity scores: {[f'{s:.4f}' for s in similarities]}
Selected candidate: {selected_candidate}
Selected similarity: {similarities[candidates.index(selected_candidate)]:.4f}

Perturbed text: {selected_candidate}"""
            
        except Exception as e:
            return f"Error applying PhraseDP with embeddings: {str(e)}"
    
    def _generate_remote_cot_tool(self, input_str: str) -> str:
        """Tool to generate Chain-of-Thought from remote model using perturbed question and perturbed options (3.1.2.new)."""
        try:
            data = json.loads(input_str)
            perturbed_question = data.get("perturbed_question", "")
            perturbed_options = data.get("perturbed_options", "")
        except:
            return "Error: Invalid input format"
        
        try:
            # Use the exact 3.1.2.new implementation from test-medqa-usmle-4-options.py
            import openai
            from dotenv import load_dotenv
            load_dotenv()
            
            # Load config to get the correct remote model
            import yaml
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            # Format question with perturbed options exactly like in 3.1.2.new
            formatted_question = f"{perturbed_question}"
            if perturbed_options:
                formatted_question += f"\n\nOptions:\n{perturbed_options}"
            formatted_question += "\n\nAnswer:"
            prompt = f"{formatted_question}\n\nPlease provide a step-by-step chain of thought to solve this medical question:"
            
            # Use the same remote model as in 3.1.2.new
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=config['remote_models']['cot_model'],
                messages=[
                    {"role": "system", "content": "You are a medical expert. Provide a clear, step-by-step chain of thought to solve the given medical multiple choice question. Focus on medical reasoning and knowledge."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.0
            )
            
            cot_text = response.choices[0].message.content
            return f"""Generated Chain-of-Thought from Remote Model (3.1.2.new):

Perturbed Question: {perturbed_question}
Remote Model: {config['remote_models']['cot_model']}

Chain-of-Thought (Remote, Fully Private):
{cot_text}"""
            
        except Exception as e:
            return f"Error generating CoT from remote model: {str(e)}"
    
    def _answer_with_local_cot_tool(self, input_str: str) -> str:
        """Tool to use local model with original question/options + private CoT to answer (3.1.2.new)."""
        try:
            data = json.loads(input_str)
            original_question = data.get("original_question", "")
            original_options = data.get("original_options", [])
            cot_text = data.get("cot_text", "")
            correct_answer = data.get("correct_answer", "")
        except:
            return "Error: Invalid input format"
        
        try:
            # Use the exact 3.1.2.new implementation from test-medqa-usmle-4-options.py
            import utils
            import yaml
            
            # Load config
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            # Get Nebius client
            nebius_client = utils.get_nebius_client()
            
            # Format question with options exactly like in 3.1.2.new
            formatted_question = f"{original_question}"
            if original_options:
                formatted_question += "\n\nOptions:\n"
                for key, value in original_options.items():
                    formatted_question += f"{key}) {value}\n"
            formatted_question += "\n\nAnswer:"
            
            # Create full prompt exactly like in 3.1.2.new
            # IMPORTANT: Do NOT include the correct answer in the prompt - the model should not know it
            full_prompt = f"""{formatted_question}

Chain of Thought:
{cot_text}

Based on the chain of thought above, what is the correct answer? Provide only the letter (A, B, C, or D):"""
            
            # Use the same local model setup as in 3.1.2.new
            def _find_working_nebius_model(client):
                """Find working Nebius model."""
                candidates = [config.get('local_model', 'microsoft/phi-4')]
                if 'microsoft/phi-4' not in candidates:
                    candidates.append('microsoft/phi-4')
                
                for model in candidates:
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": "ping"}],
                            max_tokens=1,
                            temperature=0.0,
                        )
                        return model
                    except Exception:
                        continue
                return candidates[0]
            
            # Get working model and make the call
            local_model = _find_working_nebius_model(nebius_client)
            response = nebius_client.chat.completions.create(
                model=local_model,
                messages=[
                    {"role": "system", "content": "You are a medical expert. Use the provided chain of thought to answer the multiple choice question. Provide only the letter (A, B, C, or D) of the correct option."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=256,
                temperature=0.0
            )
            
            local_answer = response.choices[0].message.content.strip()
            
            # Extract letter from answer and check correctness
            def extract_letter_from_answer(answer):
                """Extract the letter (A, B, C, D) from the model's answer."""
                answer = answer.strip().upper()
                
                # Look for single letters
                for letter in ['A', 'B', 'C', 'D']:
                    if answer == letter or answer.startswith(letter) or f" {letter}" in answer:
                        return letter
                
                # Look for patterns like "Option A", "Choice A", etc.
                patterns = ['OPTION', 'CHOICE', 'ANSWER']
                for pattern in patterns:
                    for letter in ['A', 'B', 'C', 'D']:
                        if f"{pattern} {letter}" in answer:
                            return letter
                
                return answer[:1] if answer else "Error"
            
            predicted_letter = extract_letter_from_answer(local_answer)
            is_correct = predicted_letter.upper() == correct_answer.upper() if correct_answer else "Unknown"
            
            return f"""Local Model Answer with Private CoT (3.1.2.new):

Original Question: {original_question}
Original Options: {original_options}
Private CoT: {cot_text[:200]}...
Local Model: {local_model}

Local Answer (Fully Private CoT-Aided): {local_answer}
Extracted Letter: {predicted_letter}
Correct Answer: {correct_answer}
Result: {'✅ CORRECT' if is_correct == True else '❌ INCORRECT' if is_correct == False else '❓ UNKNOWN'}

🎯 FINAL GOAL: {'ACHIEVED' if is_correct == True else 'NOT ACHIEVED' if is_correct == False else 'UNKNOWN'}"""
            
        except Exception as e:
            return f"Error in local model with CoT inference: {str(e)}"
    
    def _generate_local_cot_tool(self, input_str: str) -> str:
        """Tool to generate Chain-of-Thought from local model using original question/options."""
        try:
            data = json.loads(input_str)
            original_question = data.get("original_question", "")
            original_options = data.get("original_options", [])
        except:
            return "Error: Invalid input format"
        
        try:
            # Use the exact implementation from test-medqa-usmle-4-options.py
            import utils
            import yaml
            
            # Load config
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            # Get Nebius client
            nebius_client = utils.get_nebius_client()
            
            # Format question with options exactly like in the original implementation
            formatted_question = f"{original_question}"
            if original_options:
                formatted_question += "\n\nOptions:\n"
                for key, value in original_options.items():
                    formatted_question += f"{key}) {value}\n"
            formatted_question += "\n\nAnswer:"
            
            prompt = f"{formatted_question}\n\nPlease provide a step-by-step chain of thought to solve this medical question:"
            
            # Use the same local model setup as in the original implementation
            def _find_working_nebius_model(client):
                """Find working Nebius model."""
                candidates = [config.get('local_model', 'microsoft/phi-4')]
                if 'microsoft/phi-4' not in candidates:
                    candidates.append('microsoft/phi-4')
                
                for model in candidates:
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": "ping"}],
                            max_tokens=1,
                            temperature=0.0,
                        )
                        return model
                    except Exception:
                        continue
                return candidates[0]
            
            # Get working model and make the call
            local_model = _find_working_nebius_model(nebius_client)
            response = nebius_client.chat.completions.create(
                model=local_model,
                messages=[
                    {"role": "system", "content": "You are a medical expert. Provide a clear, step-by-step chain of thought to solve the given medical multiple choice question. Focus on medical reasoning and knowledge."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.0
            )
            
            cot_text = response.choices[0].message.content.strip()
            
            return f"""Generated Chain-of-Thought from Local Model:

Original Question: {original_question}
Original Options: {original_options}
Local Model: {local_model}

Chain-of-Thought (Local, Non-Private):
{cot_text}"""
            
        except Exception as e:
            return f"Error generating CoT from local model: {str(e)}"
    
    def _probabilistic_sampling_tool(self, input_str: str) -> str:
        """Tool to apply PhraseDP exponential mechanism for probabilistic sampling from candidates."""
        try:
            data = json.loads(input_str)
            original_text = data.get("original_text", "")
            candidates = data.get("candidates", [])
            epsilon = data.get("epsilon", 1.0)
        except:
            return "Error: Invalid input format"
        
        if not candidates:
            return "Error: No candidates provided"
        
        try:
            # Use the exact PhraseDP implementation from the codebase
            from sentence_transformers import SentenceTransformer
            from dp_sanitizer import get_embedding, differentially_private_replacement
            
            # Load SBERT model (same as in PhraseDP implementation)
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Precompute embeddings for all candidates (same as PhraseDP)
            candidate_embeddings = {sent: get_embedding(sbert_model, sent).cpu().numpy() for sent in candidates}
            
            # Apply PhraseDP exponential mechanism (exact same as in PhraseDP implementation)
            selected_candidate = differentially_private_replacement(
                target_phrase=original_text,
                epsilon=epsilon,
                candidate_phrases=candidates,
                candidate_embeddings=candidate_embeddings,
                sbert_model=sbert_model
            )
            
            # Calculate actual similarities for reporting (same as PhraseDP)
            from sklearn.metrics.pairwise import cosine_similarity
            target_embedding = get_embedding(sbert_model, original_text).cpu().numpy()
            if target_embedding.ndim == 1:
                target_embedding = target_embedding.reshape(1, -1)
            
            candidate_embeddings_matrix = np.vstack([candidate_embeddings[phrase] for phrase in candidates])
            similarities = cosine_similarity(target_embedding, candidate_embeddings_matrix)[0]
            
            selected_similarity = similarities[candidates.index(selected_candidate)]
            
            return f"""Applied PhraseDP exponential mechanism:

Original text: {original_text}
Number of candidates: {len(candidates)}
Epsilon: {epsilon}
Actual similarity scores: {[f'{s:.4f}' for s in similarities]}
Selected candidate: {selected_candidate}
Selected similarity: {selected_similarity:.4f}

This uses the exact PhraseDP implementation:
- SBERT embeddings for similarity computation
- Exponential mechanism: exp(-epsilon * distance)
- Probabilistic sampling based on similarity scores

Perturbed text: {selected_candidate}"""
            
        except Exception as e:
            return f"Error applying PhraseDP exponential mechanism: {str(e)}"
    
    def _assess_quality_tool(self, input_str: str) -> str:
        """Tool to assess perturbation quality and candidate distribution."""
        try:
            data = json.loads(input_str)
            original_text = data.get("original_text", "")
            perturbed_text = data.get("perturbed_text", "")
            candidates = data.get("candidates", [])
            dataset_info = data.get("dataset_info", {})
            epsilon = data.get("epsilon", 1.0)
        except:
            return "Error: Invalid input format"
        
        prompt = f"""Comprehensively assess the quality of this perturbation for {dataset_info.get('dataset_type', 'general')} text.

Original text: {original_text}
Perturbed text: {perturbed_text}
Candidates used: {candidates}
Domain: {dataset_info.get('domain', 'general')}
Key terminology: {dataset_info.get('key_terminology', [])}
Epsilon: {epsilon}

Evaluate PERTURBATION QUALITY:
1. Semantic similarity (0.0-1.0) - How well does perturbed text maintain original meaning?
2. Terminology preservation (0.0-1.0) - Are key domain terms preserved appropriately?
3. Grammar and coherence (0.0-1.0) - Is the text grammatically correct and coherent?
4. Privacy level achieved (0.0-1.0) - How well does it protect privacy?

Evaluate CANDIDATE DISTRIBUTION (Similarity Band Uniformity):
5. Band coverage (0.0-1.0) - How many similarity bands (0.0-0.1, 0.1-0.2, etc.) are covered?
6. Band uniformity (0.0-1.0) - Are candidates evenly distributed across similarity bands?
7. Low similarity coverage (0.0-1.0) - Are there enough candidates in 0.0-0.3 bands for privacy?
8. High similarity coverage (0.0-1.0) - Are there enough candidates in 0.7-1.0 bands for utility?

Overall Assessment:
9. Overall quality score (0.0-1.0) - Weighted average of all factors
10. Should retry? (true/false) - Based on quality thresholds
11. Retry strategy - What should be improved in next attempt

Respond in JSON format:
{{
    "perturbation_quality": {{
        "semantic_similarity": 0.85,
        "terminology_preservation": 0.90,
        "grammar_coherence": 0.80,
        "privacy_level": 0.75
    }},
    "candidate_distribution": {{
        "band_coverage": 0.80,
        "band_uniformity": 0.75,
        "low_similarity_coverage": 0.85,
        "high_similarity_coverage": 0.70
    }},
    "overall_quality": 0.82,
    "should_retry": false,
    "retry_strategy": "Quality is acceptable, no retry needed",
    "improvement_suggestions": [
        "Add more candidates in low similarity bands (0.0-0.3) for better privacy",
        "Add more candidates in high similarity bands (0.7-1.0) for better utility",
        "Improve uniformity across all similarity bands"
    ],
    "quality_breakdown": {{
        "perturbation_score": 0.82,
        "candidate_score": 0.78,
        "weighted_final": 0.80
    }}
}}"""
        
        response = self._call_local_model(prompt)
        return response
    
    def _detect_pii_tool(self, input_str: str) -> str:
        """Tool to detect and replace PII."""
        prompt = f"""Detect PII in this text and replace with appropriate placeholders:

Text: {input_str}

Replace:
- Email addresses → [email address]
- Phone numbers → [phone number]
- Street addresses → [street address]
- Person names → [person name]
- Social security numbers → [SSN]
- Credit card numbers → [credit card number]

Return the modified text with PII replaced by placeholders.
Preserve the original text structure and flow."""
        
        response = self._call_local_model(prompt, max_tokens=300)
        return response
    
    def _batch_perturb_options_tool(self, input_str: str) -> str:
        """Tool to perturb multiple choice options using PhraseDP batch processing (3.1.2.new)."""
        try:
            data = json.loads(input_str)
            options = data.get("options", [])
            dataset_info = data.get("dataset_info", {})
            epsilon = data.get("epsilon", 1.0)
        except:
            return "Error: Invalid input format"
        
        if not options:
            return "Error: Missing options"
        
        try:
            # Use the exact 3.1.2.new implementation from test-medqa-usmle-4-options.py
            import utils
            from sanitization_methods import phrasedp_sanitize_text
            
            # Get Nebius client for DP perturbation
            nebius_client = utils.get_nebius_client()
            nebius_model_name = "microsoft/phi-4"  # Default model
            
            print(f"Batch Perturbation Starting:")
            print(f"Options Count: {len(options)}")
            print(f"Epsilon: {epsilon}")
            print(f"Original Options:")
            for i, option in enumerate(options):
                print(f"  {chr(65+i)}) {option}")

            # Concatenate options with semicolon separator (exact same as 3.1.2.new)
            combined_text = "; ".join(options)

            print(f"Combined Text (semicolon-separated):\n{combined_text}")

            # Apply PhraseDP to the combined text (exact same as 3.1.2.new)
            print(f"Applying PhraseDP to combined text...")
            perturbed_combined = phrasedp_sanitize_text(
                combined_text,
                epsilon=epsilon,
                nebius_client=nebius_client,
                nebius_model_name=nebius_model_name,
            )

            print(f"Perturbed Combined Text:\n{perturbed_combined}")
            print(f"Batch perturbation complete - returning single perturbed text for CoT")

            # Return the single perturbed text - no need to parse back to individual options
            return f"""Applied PhraseDP batch perturbation (3.1.2.new):

Original options: {options}
Combined text: {combined_text}
Perturbed combined text: {perturbed_combined}
Epsilon: {epsilon}

This follows the exact 3.1.2.new pipeline:
- Options are combined with semicolon separator
- PhraseDP generates candidates internally using local model
- Exponential mechanism selects candidate based on similarity scores
- Returns single perturbed text for CoT generation

Note: The PhraseDP implementation internally generates candidates and uses exponential mechanism for selection.
The perturbed text above is the result of the probabilistic selection process."""
            
        except Exception as e:
            return f"Error applying PhraseDP batch perturbation: {str(e)}"
    
    def _analyze_candidate_distribution_tool(self, input_str: str) -> str:
        """Tool to analyze candidate distribution across similarity bands using actual embeddings."""
        try:
            data = json.loads(input_str)
            candidates = data.get("candidates", [])
            original_text = data.get("original_text", "")
            dataset_info = data.get("dataset_info", {})
            epsilon = data.get("epsilon", 1.0)
        except:
            return "Error: Invalid input format"
        
        if not candidates:
            return "No candidates to analyze"
        
        try:
            # Use actual embeddings to compute similarities
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Load SBERT model for similarity computation
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Compute actual similarities
            target_embedding = sbert_model.encode(original_text, convert_to_tensor=True).cpu().numpy()
            if target_embedding.ndim == 1:
                target_embedding = target_embedding.reshape(1, -1)
            
            candidate_embeddings = sbert_model.encode(candidates, convert_to_tensor=True).cpu().numpy()
            similarities = cosine_similarity(target_embedding, candidate_embeddings)[0]
            
            # Analyze distribution across 5 bands
            band_counts = {
                "band_1_0.0_0.2": 0,
                "band_2_0.2_0.4": 0,
                "band_3_0.4_0.6": 0,
                "band_4_0.6_0.8": 0,
                "band_5_0.8_1.0": 0
            }
            
            for sim in similarities:
                if 0.0 <= sim < 0.2:
                    band_counts["band_1_0.0_0.2"] += 1
                elif 0.2 <= sim < 0.4:
                    band_counts["band_2_0.2_0.4"] += 1
                elif 0.4 <= sim < 0.6:
                    band_counts["band_3_0.4_0.6"] += 1
                elif 0.6 <= sim < 0.8:
                    band_counts["band_4_0.6_0.8"] += 1
                elif 0.8 <= sim <= 1.0:
                    band_counts["band_5_0.8_1.0"] += 1
            
            # Calculate metrics
            covered_bands = sum(1 for count in band_counts.values() if count > 0)
            total_bands = 5
            band_coverage = covered_bands / total_bands
            
            # Check uniformity (should be 2 per band)
            expected_per_band = 2
            uniformity_score = 1.0 - (sum(abs(count - expected_per_band) for count in band_counts.values()) / (total_bands * expected_per_band))
            
            # Check for gaps
            gaps = [band for band, count in band_counts.items() if count == 0]
            should_regenerate = len(gaps) > 0 or uniformity_score < 0.8
            
            # Calculate overall score
            overall_score = (band_coverage + uniformity_score) / 2
            
            result = {
                "similarity_band_analysis": band_counts,
                "actual_similarities": [f"{s:.4f}" for s in similarities],
                "distribution_metrics": {
                    "band_coverage": band_coverage,
                    "uniformity": uniformity_score,
                    "low_similarity_coverage": 1.0 if band_counts["band_1_0.0_0.2"] >= 2 else 0.5,
                    "high_similarity_coverage": 1.0 if band_counts["band_5_0.8_1.0"] >= 2 else 0.5,
                    "medium_similarity_coverage": 1.0 if all(band_counts[f"band_{i}_0.{2*i-2}_0.{2*i}"] >= 2 for i in [2,3,4]) else 0.5,
                    "band_density": uniformity_score
                },
                "band_analysis": {
                    "covered_bands": covered_bands,
                    "total_bands": total_bands,
                    "gaps": gaps,
                    "most_populated_band": max(band_counts.items(), key=lambda x: x[1])[0],
                    "least_populated_band": min(band_counts.items(), key=lambda x: x[1])[0]
                },
                "overall_distribution_score": overall_score,
                "should_regenerate": should_regenerate,
                "improvement_recommendations": [
                    f"Add candidates to fill gaps: {gaps}" if gaps else "Perfect 5-band distribution achieved"
                ]
            }
            
            return f"""Candidate Distribution Analysis (using actual embeddings):

Original text: {original_text}
Number of candidates: {len(candidates)}
Actual similarities: {[f'{s:.4f}' for s in similarities]}

Band Distribution:
{json.dumps(result, indent=2)}"""
            
        except Exception as e:
            return f"Error analyzing candidate distribution with embeddings: {str(e)}"
    
    def _calculate_candidate_count(self, epsilon: float) -> int:
        """Calculate number of candidates based on epsilon value."""
        # Always use 10 candidates with 5 bands (2 per band)
        return 10
    
    def process(self, question: str, context: Optional[str] = None, options: Optional[List[str]] = None, epsilon: float = 1.0, correct_answer: Optional[str] = None) -> PerturbationResult:
        """
        Main processing function with intelligent reasoning and retry logic.
        
        Args:
            question: The question text to process
            context: Optional context text
            options: Optional list of answer options
            epsilon: Privacy parameter (higher = more privacy)
            correct_answer: The correct answer (A, B, C, or D) for evaluation
            
        Returns:
            PerturbationResult with all processed components and metadata
        """
        print(f"🤖 Starting PhraseDP Agent with LangChain reasoning...")
        
        # Create input for the agent
        agent_input = f"""
Process this medical question using the 3.1.2.new PhraseDP pipeline:

Question: {question}
Context: {context if context else "None"}
Options: {options if options else "None"}
Epsilon: {epsilon}
Correct Answer: {correct_answer if correct_answer else "Unknown"}

CRITICAL: You MUST complete the full 3.1.2.new pipeline:
1. Analyze dataset and question type
2. Perturb question (generate candidates, check distribution, regenerate if needed, apply exponential mechanism)
3. Perturb combined options (generate candidates, check distribution, regenerate if needed, apply exponential mechanism)
4. Generate CoT from REMOTE model using perturbed question and perturbed options
5. Generate CoT from LOCAL model using original question/options
6. Use LOCAL model with original question/options + private CoT to get final answer
7. Check if the answer is correct by comparing with the actual answer
8. Return final result with all components

Show me the perturbed question, perturbed options, both CoTs, and the final answer with correctness check.

IMPORTANT: At the end, provide a structured summary in this exact format:

**FINAL RESULTS:**
- Perturbed Question: [the actual perturbed question text]
- Perturbed Options: [the actual perturbed options text]
- Remote CoT: [the CoT from GPT-5]
- Local CoT: [the CoT from Llama 8B]
- Final Answer: [the local model's answer]
- Correctness: [CORRECT/INCORRECT/UNKNOWN]
"""
        
        try:
            # Execute the agent with reasoning
            result = self.agent_executor.invoke({"input": agent_input})
            
            # Parse the result
            output = result.get("output", "")
            
            # Extract information from the agent's output
            # This would need to be parsed based on the agent's response format
            perturbed_question = self._extract_perturbed_question(output)
            perturbed_context = self._extract_perturbed_context(output) if context else None
            perturbed_options = self._extract_perturbed_options(output) if options else None
            quality_score = self._extract_quality_score(output)
            attempts = self._extract_attempts(output)
            quality_metrics = self._extract_quality_metrics(output)
            
            metadata = {
                "epsilon": epsilon,
                "has_context": context is not None,
                "has_options": options is not None,
                "quality_score": quality_score,
                "attempts": attempts,
                "agent_framework": "langchain",
                "reasoning_enabled": True,
                "candidate_distribution_analysis": True,
                "detailed_quality_metrics": quality_metrics
            }
            
            return PerturbationResult(
                perturbed_question=perturbed_question,
                perturbed_context=perturbed_context,
                perturbed_options=perturbed_options,
                metadata=metadata,
                quality_score=quality_score,
                attempts=attempts,
                success=quality_score >= 0.7
            )
            
        except Exception as e:
            print(f"❌ Error in agent execution: {e}")
            return PerturbationResult(
                perturbed_question=question,  # Fallback to original
                perturbed_context=context,
                perturbed_options="; ".join(options) if options else None,
                metadata={"error": str(e), "agent_framework": "langchain"},
                quality_score=0.0,
                attempts=1,
                success=False
            )
    
    def _extract_perturbed_question(self, output: str) -> str:
        """Extract perturbed question from agent output."""
        # Look for the structured final results section
        if "**FINAL RESULTS:**" in output:
            lines = output.split('\n')
            for line in lines:
                if line.strip().startswith("- Perturbed Question:"):
                    return line.replace("- Perturbed Question:", "").strip()

        # Look for perturbed text in tool outputs
        perturbed_match = re.search(r'Perturbed text: (.+?)(?:\n|$)', output)
        if perturbed_match:
            return perturbed_match.group(1).strip()

        # Look for "Applied PhraseDP" results
        applied_match = re.search(r'Applied PhraseDP.*?\nSelected candidate: (.+?)(?:\n|$)', output, re.DOTALL)
        if applied_match:
            return applied_match.group(1).strip()

        return "Question extraction failed"
    
    def _extract_perturbed_context(self, output: str) -> str:
        """Extract perturbed context from agent output."""
        return output  # Placeholder
    
    def _extract_perturbed_options(self, output: str) -> str:
        """Extract perturbed options from agent output."""
        # Look for the structured final results section
        if "**FINAL RESULTS:**" in output:
            lines = output.split('\n')
            for line in lines:
                if line.strip().startswith("- Perturbed Options:"):
                    return line.replace("- Perturbed Options:", "").strip()

        # Look for batch perturbation results
        batch_match = re.search(r'Perturbed combined text: (.+?)(?:\n|$)', output)
        if batch_match:
            return batch_match.group(1).strip()

        # Look for "Applied PhraseDP batch perturbation" results
        applied_match = re.search(r'Applied PhraseDP batch perturbation.*?Perturbed combined text: (.+?)(?:\n|$)', output, re.DOTALL)
        if applied_match:
            return applied_match.group(1).strip()

        return "Options extraction failed"
    
    def _extract_quality_score(self, output: str) -> float:
        """Extract quality score from agent output."""
        # Look for overall quality score in the output
        quality_match = re.search(r'"overall_quality":\s*([0-9.]+)', output)
        if quality_match:
            return float(quality_match.group(1))
        
        # Look for weighted final score
        weighted_match = re.search(r'"weighted_final":\s*([0-9.]+)', output)
        if weighted_match:
            return float(weighted_match.group(1))
        
        # Look for distribution score
        dist_match = re.search(r'"overall_distribution_score":\s*([0-9.]+)', output)
        if dist_match:
            return float(dist_match.group(1))
        
        return 0.5  # Default
    
    def _extract_attempts(self, output: str) -> int:
        """Extract number of attempts from agent output."""
        # Count retry attempts in the output
        retry_count = output.count("retry") + output.count("attempt")
        return max(1, retry_count)
    
    def _extract_quality_metrics(self, output: str) -> Dict[str, Any]:
        """Extract detailed quality metrics from agent output."""
        metrics = {}
        
        # Extract perturbation quality metrics
        perturbation_quality = {}
        for metric in ["semantic_similarity", "terminology_preservation", "grammar_coherence", "privacy_level"]:
            match = re.search(f'"{metric}":\s*([0-9.]+)', output)
            if match:
                perturbation_quality[metric] = float(match.group(1))
        
        if perturbation_quality:
            metrics["perturbation_quality"] = perturbation_quality
        
        # Extract candidate distribution metrics
        distribution_metrics = {}
        for metric in ["count_adequacy", "semantic_diversity", "domain_relevance", "coverage_completeness", "privacy_potential", "terminology_alignment"]:
            match = re.search(f'"{metric}":\s*([0-9.]+)', output)
            if match:
                distribution_metrics[metric] = float(match.group(1))
        
        if distribution_metrics:
            metrics["candidate_distribution"] = distribution_metrics
        
        # Extract improvement suggestions
        suggestions_match = re.search(r'"improvement_recommendations":\s*\[(.*?)\]', output, re.DOTALL)
        if suggestions_match:
            suggestions_text = suggestions_match.group(1)
            suggestions = [s.strip().strip('"') for s in suggestions_text.split(',')]
            metrics["improvement_suggestions"] = suggestions
        
        return metrics


def test_langchain_agent():
    """Test the LangChain-based PhraseDP agent."""
    print("🧪 Testing PhraseDP LangChain Agent")
    print("=" * 50)
    
    agent = PhraseDPLangChainAgent()
    
    # Test medical question
    result = agent.process(
        question="What is the first-line treatment for hypertension?",
        options=["A) ACE inhibitors", "B) Beta-blockers", "C) Diuretics", "D) Calcium channel blockers"],
        epsilon=1.0
    )
    
    print(f"\n✅ Result:")
    print(f"Perturbed Question: {result.perturbed_question}")
    print(f"Quality Score: {result.quality_score}")
    print(f"Attempts: {result.attempts}")
    print(f"Success: {result.success}")
    print(f"Metadata: {json.dumps(result.metadata, indent=2)}")


def test_medqa_question():
    """Test the agent on a real MedQA question from the dataset."""
    print("🏥 Testing PhraseDP Agent on Real MedQA Question")
    print("=" * 60)
    
    # Load MedQA dataset
    try:
        from datasets import load_dataset
        print("📊 Loading MedQA dataset...")
        dataset = load_dataset('GBaker/MedQA-USMLE-4-options', split='test')
        print(f"✅ Dataset loaded: {len(dataset)} questions available")
        
        # Select a specific question (let's use question index 50 as in your previous tests)
        question_index = 50
        item = dataset[question_index]
        
        print(f"\n📋 Selected Question (Index {question_index}):")
        print(f"Question: {item['question']}")
        print(f"Options:")
        options_dict = item['options']
        for key, value in options_dict.items():
            print(f"  {key}) {value}")
        print(f"Correct Answer: {item['answer_idx']}")
        
        # Print the real options clearly at the beginning
        print(f"\n🔍 REAL OPTIONS (for reference):")
        print("=" * 50)
        for key, value in options_dict.items():
            print(f"  {key}) {value}")
        print("=" * 50)
        
        # Initialize agent
        print(f"\n🤖 Initializing PhraseDP LangChain Agent...")
        agent = PhraseDPLangChainAgent()
        print("✅ Agent initialized successfully!")
        
        # Process the question
        print(f"\n🔄 Processing with Agent...")
        print("-" * 50)
        
        # Convert options dict to list
        options_list = [options_dict['A'], options_dict['B'], options_dict['C'], options_dict['D']]
        
        result = agent.process(
            question=item['question'],
            options=options_list,
            epsilon=1.0,
            correct_answer=item['answer_idx']
        )
        
        # Display results
        print(f"\n📊 RESULTS:")
        print(f"=" * 50)
        print(f"Original Question: {item['question']}")
        print(f"Perturbed Question: {result.perturbed_question}")
        print(f"Perturbed Options: {result.perturbed_options}")
        print(f"Quality Score: {result.quality_score:.3f}")
        print(f"Attempts: {result.attempts}")
        print(f"Success: {result.success}")
        
        # Display additional information from metadata
        if result.metadata:
            if 'remote_cot' in result.metadata:
                print(f"\n🤖 REMOTE CoT (GPT-5):")
                print(f"{result.metadata['remote_cot'][:200]}...")
            
            if 'local_cot' in result.metadata:
                print(f"\n🏠 LOCAL CoT (Llama 8B):")
                print(f"{result.metadata['local_cot'][:200]}...")
            
            if 'final_answer' in result.metadata:
                print(f"\n🎯 FINAL ANSWER:")
                print(f"Answer: {result.metadata['final_answer']}")
            
            if 'correctness' in result.metadata:
                print(f"\n✅ CORRECTNESS CHECK:")
                print(f"Result: {result.metadata['correctness']}")
        
        # Display detailed quality metrics
        if result.metadata.get('detailed_quality_metrics'):
            print(f"\n🔍 DETAILED QUALITY ANALYSIS:")
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
        
        print(f"\n📋 Full Metadata:")
        print(json.dumps(result.metadata, indent=2))
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_multiple_medqa_questions():
    """Test the agent on multiple MedQA questions to see reasoning patterns."""
    print("🏥 Testing PhraseDP Agent on Multiple MedQA Questions")
    print("=" * 70)
    
    try:
        from datasets import load_dataset
        dataset = load_dataset('GBaker/MedQA-USMLE-4-options', split='test')
        
        # Test multiple questions
        test_indices = [50, 51, 52, 53, 54]  # Same as your previous test
        results = []
        
        agent = PhraseDPLangChainAgent()
        
        for i, idx in enumerate(test_indices, 1):
            print(f"\n{'='*70}")
            print(f"📋 TEST {i}: Question Index {idx}")
            print(f"{'='*70}")
            
            item = dataset[idx]
            print(f"Question: {item['question']}")
            print(f"Correct Answer: {item['answer_idx']}")
            
            # Convert options dict to list
            options_dict = item['options']
            options_list = [options_dict['A'], options_dict['B'], options_dict['C'], options_dict['D']]
            
            # Process with agent
            result = agent.process(
                question=item['question'],
                options=options_list,
                epsilon=1.0
            )
            
            print(f"\n📊 Results:")
            print(f"Quality Score: {result.quality_score:.3f}")
            print(f"Attempts: {result.attempts}")
            print(f"Success: {result.success}")
            print(f"Perturbed: {result.perturbed_question[:100]}...")
            
            results.append({
                'index': idx,
                'success': result.success,
                'quality_score': result.quality_score,
                'attempts': result.attempts
            })
        
        # Summary
        print(f"\n📊 SUMMARY:")
        print(f"=" * 50)
        successful = [r for r in results if r['success']]
        print(f"✅ Successful: {len(successful)}/{len(results)}")
        
        if successful:
            avg_quality = sum(r['quality_score'] for r in successful) / len(successful)
            avg_attempts = sum(r['attempts'] for r in successful) / len(successful)
            print(f"📈 Average Quality Score: {avg_quality:.3f}")
            print(f"🔄 Average Attempts: {avg_attempts:.1f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "medqa":
        # Test on real MedQA question
        test_medqa_question()
    elif len(sys.argv) > 1 and sys.argv[1] == "multiple":
        # Test on multiple MedQA questions
        test_multiple_medqa_questions()
    else:
        # Default test
        test_langchain_agent()
