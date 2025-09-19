"""
Diagnostic Orchestrator

Implementation of the diagnostic orchestrator pattern from the Microsoft AI research paper,
featuring multi-agent coordination for clinical diagnosis with role-specialized reasoning agents.

This module provides:
- Multi-agent orchestration framework with 5 specialized medical roles
- Chain of debate coordination between agents
- Comprehensive execution tracing and decision logging
- Cost-aware diagnostic planning
"""

import os
import json
import re
import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, validator
import openai
from openai import AsyncOpenAI
# from dotenv import load_dotenv

# Load environment variables
# load_dotenv()

# Import cost estimation capabilities
from cost_estimator import cost_estimator, TestCost

# Enhanced Pydantic models for structured agent outputs with validation
from typing import Literal

class BayesianUpdate(BaseModel):
    """Structured Bayesian reasoning for probability updates"""
    prior_probability: float = Field(..., ge=0.0, le=1.0, description="Prior probability before evidence")
    posterior_probability: float = Field(..., ge=0.0, le=1.0, description="Updated probability after evidence")
    likelihood_ratio: float = Field(..., gt=0.0, description="Likelihood ratio of evidence")
    evidence_weight: float = Field(..., ge=0.0, le=1.0, description="Strength/reliability of evidence")
    reasoning: str = Field(..., min_length=20, description="Detailed Bayesian reasoning explanation")

class EvidenceItem(BaseModel):
    """Individual piece of evidence with impact assessment"""
    description: str = Field(..., min_length=5, description="Evidence description")
    impact_type: Literal["supporting", "contradictory", "neutral"] = Field(..., description="Evidence impact type")
    strength: float = Field(..., ge=0.0, le=1.0, description="Evidence strength (0=weak, 1=strong)")
    reliability: float = Field(..., ge=0.0, le=1.0, description="Evidence reliability (0=unreliable, 1=definitive)")
    source: str = Field(..., min_length=3, description="Evidence source (clinical, lab, imaging, etc.)")

class HypothesisItem(BaseModel):
    """Individual hypothesis with validated probability and enhanced Bayesian tracking"""
    condition: str = Field(..., min_length=1, description="Medical condition name")
    probability: float = Field(..., ge=0.0, le=1.0, description="Current probability between 0 and 1")
    reasoning: str = Field(..., min_length=10, description="Detailed clinical reasoning")
    supporting_evidence: List[EvidenceItem] = Field(default_factory=list, description="Supporting evidence with impact")
    contradictory_evidence: List[EvidenceItem] = Field(default_factory=list, description="Contradictory evidence with impact")
    bayesian_updates: List[BayesianUpdate] = Field(default_factory=list, description="History of Bayesian updates")
    confidence_factors: Dict[str, float] = Field(default_factory=dict, description="Factors affecting confidence")

class ConfidenceAssessment(BaseModel):
    """Numerical confidence assessment with detailed factors"""
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall diagnostic confidence")
    leading_hypothesis_strength: float = Field(..., ge=0.0, le=1.0, description="Strength of leading hypothesis")
    evidence_completeness: float = Field(..., ge=0.0, le=1.0, description="Completeness of available evidence")
    diagnostic_clarity: float = Field(..., ge=0.0, le=1.0, description="Clarity of diagnostic picture")
    uncertainty_factors: List[str] = Field(default_factory=list, description="Factors contributing to uncertainty")
    confidence_level: Literal["low", "medium", "high"] = Field(..., description="Categorical confidence level")

class HypothesisUpdate(BaseModel):
    """Enhanced structured output from Dr. Hypothesis with advanced Bayesian reasoning"""
    hypotheses: List[HypothesisItem] = Field(..., max_items=5, description="Top hypotheses ranked by probability")
    bayesian_updates: str = Field(..., min_length=10, description="Explanation of probability updates")
    confidence_assessment: ConfidenceAssessment = Field(..., description="Detailed confidence assessment")
    differential_reasoning: str = Field(..., min_length=20, description="Reasoning for differential diagnosis ranking")
    key_discriminating_features: List[str] = Field(default_factory=list, max_items=5, description="Features that distinguish between hypotheses")
    
    @validator('hypotheses')
    def validate_probabilities_sum(cls, v):
        """Ensure probabilities are reasonable (don't need to sum to 1 for differential diagnosis)"""
        if len(v) == 0:
            raise ValueError("At least one hypothesis must be provided")
        return v
    
    @validator('confidence_assessment')
    def validate_confidence_consistency(cls, v, values):
        """Ensure confidence levels are consistent with hypothesis probabilities"""
        if 'hypotheses' in values and values['hypotheses']:
            max_prob = max(h.probability for h in values['hypotheses'])
            if max_prob > 0.8 and v.confidence_level == "low":
                raise ValueError("High probability hypotheses inconsistent with low confidence")
            if max_prob < 0.3 and v.confidence_level == "high":
                raise ValueError("Low probability hypotheses inconsistent with high confidence")
        return v

class DiscriminativeValue(BaseModel):
    """Quantitative assessment of test's discriminative power"""
    hypothesis_separation_score: float = Field(..., ge=0.0, le=1.0, description="How well test separates competing hypotheses")
    sensitivity_score: float = Field(..., ge=0.0, le=1.0, description="Test sensitivity for target conditions")
    specificity_score: float = Field(..., ge=0.0, le=1.0, description="Test specificity for target conditions")
    diagnostic_yield_score: float = Field(..., ge=0.0, le=1.0, description="Overall diagnostic yield assessment")
    
class StepwiseReasoning(BaseModel):
    """Step-wise diagnostic approach reasoning"""
    diagnostic_tier: Literal["first_line", "second_line", "third_line", "specialized"] = Field(..., description="Diagnostic tier in stepwise approach")
    prerequisite_tests: List[str] = Field(default_factory=list, description="Tests that should be done first")
    accessibility: Literal["immediate", "same_day", "within_week", "referral_needed"] = Field(..., description="Test accessibility")
    cost_tier: Literal["low", "moderate", "high", "very_high"] = Field(..., description="Cost categorization")

class RedundancyAssessment(BaseModel):
    """Assessment of test redundancy with existing tests"""
    redundancy_score: float = Field(..., ge=0.0, le=1.0, description="Overlap with already performed tests (0=no overlap, 1=complete overlap)")
    overlapping_tests: List[str] = Field(default_factory=list, description="Tests that provide similar information")
    unique_information: str = Field(..., min_length=10, description="Unique diagnostic information this test provides")
    incremental_value: float = Field(..., ge=0.0, le=1.0, description="Additional diagnostic value beyond existing tests")

class TestRecommendationItem(BaseModel):
    """Enhanced individual test recommendation with discriminative scoring"""
    test_name: str = Field(..., min_length=2, description="Specific diagnostic test name")
    rationale: str = Field(..., min_length=10, description="Clinical rationale for test")
    priority: int = Field(..., ge=1, le=3, description="Priority level (1=highest, 3=lowest)")
    discriminative_value: DiscriminativeValue = Field(..., description="Quantitative discriminative power assessment")
    stepwise_reasoning: StepwiseReasoning = Field(..., description="Step-wise diagnostic approach rationale")
    redundancy_assessment: RedundancyAssessment = Field(..., description="Assessment of test redundancy")
    estimated_cost: Optional[float] = Field(None, ge=0, description="Estimated cost in USD")
    cost_effectiveness_ratio: Optional[float] = Field(None, ge=0, description="Diagnostic yield per dollar")
    
    @validator('cost_effectiveness_ratio')
    def calculate_cost_effectiveness(cls, v, values):
        """Calculate cost-effectiveness if not provided"""
        if v is None and values.get('estimated_cost') and values.get('discriminative_value'):
            cost = values['estimated_cost']
            yield_score = values['discriminative_value'].diagnostic_yield_score
            if cost > 0:
                return yield_score / cost * 1000  # Scale for readability
        return v

class TestRecommendations(BaseModel):
    """Enhanced structured output from Dr. Test-Chooser"""
    recommended_tests: List[TestRecommendationItem] = Field(..., max_items=3, description="Up to 3 recommended tests")
    reasoning: str = Field(..., min_length=10, description="Overall test selection strategy")
    stepwise_strategy: str = Field(..., min_length=20, description="Step-wise diagnostic approach explanation")
    redundancy_analysis: str = Field(..., min_length=15, description="Analysis of test redundancy and complementarity")
    cost_optimization_notes: str = Field(..., min_length=10, description="Cost-effectiveness considerations")

class ChallengeItem(BaseModel):
    """Individual challenge to current thinking"""
    target_hypothesis: str = Field(..., min_length=1, description="Hypothesis being challenged")
    challenge_type: Literal["anchoring_bias", "contradictory_evidence", "alternative_explanation", "cognitive_bias"] = Field(..., description="Type of challenge")
    reasoning: str = Field(..., min_length=10, description="Detailed challenge reasoning")
    alternative_hypothesis: Optional[str] = Field(None, description="Proposed alternative if applicable")

class ChallengeResponse(BaseModel):
    """Structured output from Dr. Challenger"""
    challenges: List[ChallengeItem] = Field(default_factory=list, max_items=5, description="List of challenges raised")
    falsifying_tests: List[str] = Field(default_factory=list, max_items=3, description="Tests that could disprove leading diagnosis")
    overlooked_possibilities: List[str] = Field(default_factory=list, max_items=3, description="Potentially missed diagnoses")
    cognitive_bias_warnings: str = Field(..., min_length=5, description="Warnings about reasoning errors")

class CostAnalysisItem(BaseModel):
    """Individual cost analysis for a test"""
    test_name: str = Field(..., min_length=1, description="Test being analyzed")
    approval_status: Literal["approved", "conditional", "rejected"] = Field(..., description="Stewardship decision")
    reasoning: str = Field(..., min_length=10, description="Cost-benefit analysis reasoning")
    cheaper_alternative: Optional[str] = Field(None, description="Suggested cheaper alternative")
    cost_category: Literal["low", "moderate", "high", "very_high"] = Field(..., description="Cost categorization")

class StewardshipReview(BaseModel):
    """Structured output from Dr. Stewardship"""
    cost_analysis: List[CostAnalysisItem] = Field(default_factory=list, description="Analysis of proposed tests")
    budget_recommendation: Literal["continue", "proceed_with_caution", "stop_and_reassess"] = Field(..., description="Budget guidance")
    stewardship_notes: str = Field(..., min_length=10, description="Overall cost-consciousness guidance")

class QualityGap(BaseModel):
    """Individual quality gap identified"""
    gap_type: Literal["missing_information", "logical_inconsistency", "incomplete_workup", "safety_concern"] = Field(..., description="Type of gap")
    description: str = Field(..., min_length=5, description="Description of the gap")
    recommendation: str = Field(..., min_length=5, description="Suggested action to address gap")

class ChecklistAssessment(BaseModel):
    """Structured output from Dr. Checklist"""
    quality_score: int = Field(..., ge=1, le=10, description="Overall quality score (1-10)")
    identified_gaps: List[QualityGap] = Field(default_factory=list, max_items=5, description="Quality gaps identified")
    completeness_assessment: str = Field(..., min_length=10, description="Assessment of diagnostic completeness")
    safety_concerns: List[str] = Field(default_factory=list, max_items=3, description="Patient safety concerns")

class ConsensusDecision(BaseModel):
    """Structured consensus decision from panel"""
    consensus_action: Literal["ask_questions", "order_tests", "make_diagnosis"] = Field(..., description="Decided action")
    action_content: Dict[str, Any] = Field(..., description="Content of the action (questions, tests, or diagnosis)")
    reasoning: str = Field(..., min_length=10, description="Consensus reasoning")
    panel_synthesis: str = Field(..., min_length=10, description="Synthesis of all panel input")
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Consensus confidence")

# Trace and execution models
class ActionType(str, Enum):
    """Types of actions the diagnostic panel can take after deliberation"""
    ASK_QUESTIONS = "ask_questions"
    ORDER_TESTS = "order_tests" 
    MAKE_DIAGNOSIS = "make_diagnosis"

@dataclass
class DiagnosticHypothesis:
    """Represents a diagnostic hypothesis with probability validation"""
    condition: str
    probability: float
    reasoning: str
    supporting_evidence: List[str] = field(default_factory=list)
    contradictory_evidence: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate probability is in valid range"""
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"Probability must be between 0.0 and 1.0, got {self.probability}")
        if not self.condition.strip():
            raise ValueError("Condition name cannot be empty")
        if not self.reasoning.strip():
            raise ValueError("Reasoning cannot be empty")

@dataclass
class TestRecommendation:
    """Represents a recommended diagnostic test with validation"""
    test_name: str
    rationale: str
    estimated_cost: Optional[float] = None
    priority: int = 1  # 1=highest, 3=lowest
    discriminative_value: str = ""
    
    def __post_init__(self):
        """Validate test recommendation fields"""
        if not self.test_name.strip():
            raise ValueError("Test name cannot be empty")
        if not self.rationale.strip():
            raise ValueError("Rationale cannot be empty")
        if not 1 <= self.priority <= 3:
            raise ValueError(f"Priority must be between 1 and 3, got {self.priority}")
        if self.estimated_cost is not None and self.estimated_cost < 0:
            raise ValueError(f"Cost cannot be negative, got {self.estimated_cost}")
    
@dataclass
class AgentMessage:
    """Represents a message from one of the specialized agents"""
    agent_role: str
    timestamp: datetime
    message_type: str  # "hypothesis", "test_recommendation", "challenge", "stewardship_review"
    content: str
    structured_data: Optional[Dict[str, Any]] = None

@dataclass
class ExecutionTrace:
    """Comprehensive trace of the diagnostic orchestration execution"""
    case_id: str
    session_id: str
    timestamp: datetime
    round_number: int
    action_type: ActionType
    actor: str
    content: str
    structured_data: Optional[Dict[str, Any]] = None
    cost_impact: Optional[float] = None

@dataclass
class DiagnosticAction:
    """Represents an action taken during diagnostic reasoning"""
    action_type: str  # "ask_questions", "order_tests", "make_diagnosis"
    content: Union[str, List[str]]
    reasoning: str
    round_number: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        """String representation for comparison in stagnation detection"""
        content_str = str(self.content) if isinstance(self.content, str) else "|".join(self.content)
        return f"{self.action_type}:{content_str}"

@dataclass 
class CaseState:
    """Enhanced state management for diagnostic process with evidence tracking and stagnation detection"""
    initial_case_info: str
    evidence_log: List[str] = field(default_factory=list)
    differential_diagnosis: Dict[str, float] = field(default_factory=dict)
    tests_performed: List[str] = field(default_factory=list)
    questions_asked: List[str] = field(default_factory=list)
    cumulative_cost: float = 0.0
    current_round: int = 0
    action_history: List[DiagnosticAction] = field(default_factory=list)
    
    def add_evidence(self, evidence: str) -> None:
        """Add new evidence to the case log"""
        if evidence not in self.evidence_log:
            self.evidence_log.append(evidence)
    
    def update_differential(self, diagnosis_dict: Dict[str, float]) -> None:
        """Update the differential diagnosis probabilities"""
        self.differential_diagnosis.update(diagnosis_dict)
    
    def add_test(self, test_name: str) -> None:
        """Add a test to the performed tests list"""
        if test_name not in self.tests_performed:
            self.tests_performed.append(test_name)
    
    def add_question(self, question: str) -> None:
        """Add a question to the asked questions list"""
        if question not in self.questions_asked:
            self.questions_asked.append(question)
    
    def is_stagnating(self, new_action: DiagnosticAction, lookback: int = 3) -> bool:
        """Detect if the diagnostic process is stagnating by checking recent actions"""
        if len(self.action_history) < lookback:
            return False
        
        recent_actions = [str(action) for action in self.action_history[-lookback:]]
        new_action_str = str(new_action)
        
        # Check if this action is too similar to recent actions
        similar_count = sum(1 for action_str in recent_actions if action_str == new_action_str)
        return similar_count >= 2
    
    def add_action(self, action: DiagnosticAction) -> None:
        """Add an action to the history for stagnation detection"""
        action.round_number = self.current_round
        self.action_history.append(action)
        
        # Keep only last 10 actions to prevent memory bloat
        if len(self.action_history) > 10:
            self.action_history = self.action_history[-10:]
    
    def get_max_confidence(self) -> float:
        """Get the highest confidence diagnosis probability"""
        return max(self.differential_diagnosis.values()) if self.differential_diagnosis else 0.0
    
    def get_leading_diagnosis(self) -> str:
        """Get the diagnosis with highest probability"""
        if not self.differential_diagnosis:
            return "No leading diagnosis"
        return max(self.differential_diagnosis.items(), key=lambda x: x[1])[0]
    
    def summarize_evidence(self) -> str:
        """Create a summary of all accumulated evidence"""
        if not self.evidence_log:
            return "No evidence accumulated yet."
        return " | ".join(self.evidence_log[-5:])  # Last 5 pieces of evidence

class BayesianReasoningHelper:
    """Helper class for Bayesian probability calculations and reasoning"""
    
    @staticmethod
    def calculate_posterior(prior: float, likelihood_ratio: float) -> float:
        """Calculate posterior probability using Bayes' theorem with likelihood ratio"""
        # P(H|E) = P(E|H) * P(H) / P(E)
        # Using likelihood ratio: LR = P(E|H) / P(E|not H)
        odds_prior = prior / (1 - prior) if prior < 1.0 else 1.0
        odds_posterior = odds_prior * likelihood_ratio
        posterior = odds_posterior / (1 + odds_posterior)
        return min(max(posterior, 0.001), 0.999)  # Clamp between bounds
    
    @staticmethod
    def estimate_likelihood_ratio(evidence_strength: float, evidence_type: str) -> float:
        """Estimate likelihood ratio based on evidence strength and type"""
        base_ratios = {
            "pathognomonic": 20.0,  # Nearly diagnostic
            "highly_specific": 10.0,  # Very strong evidence
            "specific": 5.0,  # Strong evidence
            "suggestive": 2.0,  # Moderate evidence
            "nonspecific": 1.2,  # Weak evidence
            "contradictory": 0.5,  # Evidence against
            "neutral": 1.0  # No impact
        }
        base_lr = base_ratios.get(evidence_type, 2.0)
        # Adjust by evidence strength
        if base_lr > 1.0:
            return 1.0 + (base_lr - 1.0) * evidence_strength
        else:
            return 1.0 - (1.0 - base_lr) * evidence_strength
    
    @staticmethod
    def assess_confidence_level(max_probability: float, evidence_quality: float, 
                               num_hypotheses: int) -> Tuple[str, float]:
        """Assess confidence level based on probability and evidence quality"""
        # Calculate base confidence from leading probability
        prob_confidence = max_probability
        
        # Adjust for evidence quality
        quality_factor = evidence_quality
        
        # Adjust for number of competing hypotheses
        competition_factor = 1.0 - (num_hypotheses - 1) * 0.05  # Decrease with more hypotheses
        competition_factor = max(competition_factor, 0.7)
        
        # Combined confidence
        overall_confidence = prob_confidence * quality_factor * competition_factor
        
        # Categorical levels with thresholds
        if overall_confidence >= 0.8:
            return "high", overall_confidence
        elif overall_confidence >= 0.5:
            return "medium", overall_confidence
        else:
            return "low", overall_confidence
    
    @staticmethod
    def calculate_evidence_weight(reliability: float, specificity: str, 
                                 clinical_context: str = "general") -> float:
        """Calculate overall evidence weight considering multiple factors"""
        # Base weight from reliability
        base_weight = reliability
        
        # Specificity multiplier
        specificity_multipliers = {
            "pathognomonic": 1.0,
            "highly_specific": 0.9,
            "moderately_specific": 0.7,
            "nonspecific": 0.4,
            "contradictory": 0.8  # High weight for contradictory evidence
        }
        
        specificity_mult = specificity_multipliers.get(specificity, 0.6)
        
        # Context adjustment
        context_adjustments = {
            "emergency": 1.1,  # Higher weight in emergency settings
            "outpatient": 0.9,  # Lower weight in outpatient settings
            "icu": 1.0,
            "general": 1.0
        }
        
        context_adj = context_adjustments.get(clinical_context, 1.0)
        
        return min(base_weight * specificity_mult * context_adj, 1.0)

class TestSelectionHelper:
    """Advanced helper class for sophisticated test selection and discriminative value scoring"""
    
    # Test categories and their characteristics
    TEST_CATEGORIES = {
        "blood_tests": {
            "accessibility": "immediate",
            "cost_tier": "low",
            "diagnostic_tier": "first_line",
            "turnaround_time": "hours"
        },
        "imaging_basic": {
            "accessibility": "same_day", 
            "cost_tier": "moderate",
            "diagnostic_tier": "first_line",
            "turnaround_time": "hours"
        },
        "imaging_advanced": {
            "accessibility": "within_week",
            "cost_tier": "high", 
            "diagnostic_tier": "second_line",
            "turnaround_time": "days"
        },
        "biopsy": {
            "accessibility": "referral_needed",
            "cost_tier": "very_high",
            "diagnostic_tier": "third_line", 
            "turnaround_time": "weeks"
        },
        "specialty_tests": {
            "accessibility": "referral_needed",
            "cost_tier": "high",
            "diagnostic_tier": "specialized",
            "turnaround_time": "days_to_weeks"
        }
    }
    
    # Common test redundancies and overlaps
    TEST_REDUNDANCIES = {
        "cbc": ["complete_blood_count", "full_blood_count", "hemogram"],
        "chemistry_panel": ["basic_metabolic_panel", "comprehensive_metabolic_panel", "chemistry_7", "chemistry_14"],
        "chest_imaging": ["chest_xray", "chest_ct", "chest_mri"],
        "cardiac_enzymes": ["troponin", "ck_mb", "myoglobin"],
        "liver_function": ["alt", "ast", "bilirubin", "alkaline_phosphatase"],
        "coagulation": ["pt", "ptt", "inr", "coagulation_studies"]
    }
    
    @staticmethod
    def calculate_discriminative_score(test_name: str, hypotheses: List[DiagnosticHypothesis]) -> DiscriminativeValue:
        """Calculate discriminative value score for a test given current hypotheses"""
        if not hypotheses or len(hypotheses) < 2:
            # Limited discriminative value with few hypotheses
            return DiscriminativeValue(
                hypothesis_separation_score=0.3,
                sensitivity_score=0.6,
                specificity_score=0.6, 
                diagnostic_yield_score=0.4
            )
        
        # Calculate separation score based on hypothesis probabilities
        probs = [h.probability for h in hypotheses]
        prob_range = max(probs) - min(probs)
        separation_score = min(prob_range * 2.0, 1.0)  # Scale to 0-1
        
        # Estimate sensitivity/specificity based on test type and condition match
        sensitivity_score = TestSelectionHelper._estimate_test_performance(test_name, hypotheses, "sensitivity")
        specificity_score = TestSelectionHelper._estimate_test_performance(test_name, hypotheses, "specificity")
        
        # Overall diagnostic yield
        diagnostic_yield = (separation_score + sensitivity_score + specificity_score) / 3
        
        return DiscriminativeValue(
            hypothesis_separation_score=separation_score,
            sensitivity_score=sensitivity_score,
            specificity_score=specificity_score,
            diagnostic_yield_score=diagnostic_yield
        )
    
    @staticmethod
    def _estimate_test_performance(test_name: str, hypotheses: List[DiagnosticHypothesis], metric: str) -> float:
        """Estimate test performance based on test name and target conditions"""
        test_lower = test_name.lower()
        
        # High-performance test patterns
        high_performance_patterns = {
            "biopsy": 0.95,
            "pathology": 0.95,
            "culture": 0.90,
            "pcr": 0.90,
            "genetic": 0.85
        }
        
        # Medium-performance test patterns  
        medium_performance_patterns = {
            "ct": 0.80,
            "mri": 0.80,
            "echo": 0.75,
            "ultrasound": 0.70,
            "xray": 0.65
        }
        
        # Basic test patterns
        basic_performance_patterns = {
            "blood": 0.60,
            "urine": 0.55,
            "cbc": 0.50,
            "chemistry": 0.50
        }
        
        # Check test patterns
        for pattern, score in high_performance_patterns.items():
            if pattern in test_lower:
                return min(score + 0.05, 1.0) if metric == "specificity" else score
        
        for pattern, score in medium_performance_patterns.items():
            if pattern in test_lower:
                return score
        
        for pattern, score in basic_performance_patterns.items():
            if pattern in test_lower:
                return score
        
        # Default moderate performance
        return 0.65
    
    @staticmethod
    def assess_test_redundancy(test_name: str, performed_tests: List[str]) -> RedundancyAssessment:
        """Assess redundancy of proposed test with already performed tests"""
        test_lower = test_name.lower()
        performed_lower = [t.lower() for t in performed_tests]
        
        overlapping_tests = []
        redundancy_score = 0.0
        
        # Check for direct redundancy
        if test_lower in performed_lower:
            redundancy_score = 1.0
            overlapping_tests.append(test_name)
        else:
            # Check for category overlap
            for category, synonyms in TestSelectionHelper.TEST_REDUNDANCIES.items():
                test_matches = any(synonym in test_lower for synonym in synonyms)
                performed_matches = [t for t in performed_lower if any(synonym in t for synonym in synonyms)]
                
                if test_matches and performed_matches:
                    redundancy_score = max(redundancy_score, 0.7)
                    overlapping_tests.extend([t for t in performed_tests if t.lower() in performed_matches])
        
        # Calculate incremental value
        incremental_value = max(0.1, 1.0 - redundancy_score)
        
        unique_info = TestSelectionHelper._generate_unique_information_description(test_name, overlapping_tests)
        
        return RedundancyAssessment(
            redundancy_score=redundancy_score,
            overlapping_tests=overlapping_tests,
            unique_information=unique_info,
            incremental_value=incremental_value
        )
    
    @staticmethod
    def _generate_unique_information_description(test_name: str, overlapping_tests: List[str]) -> str:
        """Generate description of unique diagnostic information"""
        if not overlapping_tests:
            return f"{test_name} provides novel diagnostic information not available from previous tests"
        
        test_lower = test_name.lower()
        
        # Specific unique information patterns
        if "ct" in test_lower and any("xray" in t.lower() for t in overlapping_tests):
            return "CT provides cross-sectional anatomy and better soft tissue detail than X-ray"
        elif "mri" in test_lower and any("ct" in t.lower() for t in overlapping_tests):
            return "MRI offers superior soft tissue contrast and no radiation exposure compared to CT"
        elif "echo" in test_lower and any("chest" in t.lower() for t in overlapping_tests):
            return "Echocardiogram provides detailed cardiac function assessment beyond chest imaging"
        elif "culture" in test_lower and any("blood" in t.lower() for t in overlapping_tests):
            return "Culture provides organism identification and antibiotic sensitivity beyond basic blood tests"
        
        return f"{test_name} offers additional specific diagnostic information complementing {', '.join(overlapping_tests)}"
    
    @staticmethod
    def determine_stepwise_tier(test_name: str, cost: float, hypotheses: List[DiagnosticHypothesis]) -> StepwiseReasoning:
        """Determine appropriate diagnostic tier and stepwise reasoning"""
        test_lower = test_name.lower()
        
        # Categorize test
        category = TestSelectionHelper._categorize_test(test_name)
        category_info = TestSelectionHelper.TEST_CATEGORIES.get(category, TestSelectionHelper.TEST_CATEGORIES["specialty_tests"])
        
        # Determine prerequisites
        prerequisites = TestSelectionHelper._determine_prerequisites(test_name, hypotheses)
        
        return StepwiseReasoning(
            diagnostic_tier=category_info["diagnostic_tier"],
            prerequisite_tests=prerequisites,
            accessibility=category_info["accessibility"],
            cost_tier=category_info["cost_tier"]
        )
    
    @staticmethod
    def _categorize_test(test_name: str) -> str:
        """Categorize test into predefined categories"""
        test_lower = test_name.lower()
        
        blood_patterns = ["blood", "cbc", "chemistry", "troponin", "glucose", "electrolyte", "liver", "kidney"]
        basic_imaging_patterns = ["xray", "ultrasound", "echo"]
        advanced_imaging_patterns = ["ct", "mri", "pet", "angiogram"]
        biopsy_patterns = ["biopsy", "aspiration", "cytology"]
        
        if any(pattern in test_lower for pattern in blood_patterns):
            return "blood_tests"
        elif any(pattern in test_lower for pattern in basic_imaging_patterns):
            return "imaging_basic"
        elif any(pattern in test_lower for pattern in advanced_imaging_patterns):
            return "imaging_advanced"
        elif any(pattern in test_lower for pattern in biopsy_patterns):
            return "biopsy"
        else:
            return "specialty_tests"
    
    @staticmethod
    def _determine_prerequisites(test_name: str, hypotheses: List[DiagnosticHypothesis]) -> List[str]:
        """Determine prerequisite tests that should be done first"""
        test_lower = test_name.lower()
        prerequisites = []
        
        # Advanced imaging usually requires basic workup first
        if any(pattern in test_lower for pattern in ["ct", "mri", "pet"]):
            prerequisites.extend(["CBC", "Basic metabolic panel"])
            if "chest" in test_lower:
                prerequisites.append("Chest X-ray")
        
        # Invasive procedures require imaging first
        if any(pattern in test_lower for pattern in ["biopsy", "aspiration"]):
            prerequisites.extend(["CBC", "Coagulation studies"])
            if "liver" in test_lower:
                prerequisites.extend(["Ultrasound abdomen", "CT abdomen"])
        
        # Specialty tests often require basic workup
        if any(pattern in test_lower for pattern in ["angiogram", "catheter"]):
            prerequisites.extend(["CBC", "Chemistry panel", "Coagulation studies", "ECG"])
        
        return prerequisites

@dataclass
class DeliberationState:
    """Structured state for panel deliberation coordination"""
    hypothesis_analysis: str = ""
    test_chooser_analysis: str = ""
    challenger_analysis: str = ""
    stewardship_analysis: str = ""
    checklist_analysis: str = ""
    consensus_reasoning: str = ""
    stagnation_detected: bool = False
    retry_count: int = 0
    round_start_time: datetime = field(default_factory=datetime.now)
    
    def reset_for_new_round(self) -> None:
        """Reset state for a new deliberation round"""
        self.hypothesis_analysis = ""
        self.test_chooser_analysis = ""
        self.challenger_analysis = ""
        self.stewardship_analysis = ""
        self.checklist_analysis = ""
        self.consensus_reasoning = ""
        self.stagnation_detected = False
        self.retry_count = 0
        self.round_start_time = datetime.now()
    
    def has_all_agent_inputs(self) -> bool:
        """Check if all required agent analyses are completed"""
        return all([
            self.hypothesis_analysis,
            self.test_chooser_analysis,
            self.challenger_analysis,
            self.stewardship_analysis,
            self.checklist_analysis
        ])

class CaseExecutionSession:
    """Manages a single diagnostic case execution session with enhanced state tracking"""
    
    def __init__(self, case_id: str, initial_case_info: str):
        self.case_id = case_id
        self.session_id = str(uuid.uuid4())
        self.initial_case_info = initial_case_info
        
        # Legacy attributes for backward compatibility
        self.traces: List[ExecutionTrace] = []
        self.agent_messages: List[AgentMessage] = []
        self.current_round = 0
        self.total_cost = 0.0
        self.final_diagnosis: Optional[str] = None
        self.confidence_score: Optional[float] = None
        self.created_at = datetime.now()
        
        # Enhanced state management
        self.case_state = CaseState(
            initial_case_info=initial_case_info,
            current_round=0,
            cumulative_cost=0.0
        )
        self.deliberation_state = DeliberationState()
        
    def add_trace(self, action_type: ActionType, actor: str, content: str, 
                  structured_data: Optional[Dict[str, Any]] = None, 
                  cost_impact: Optional[float] = None):
        """Add an execution trace entry"""
        trace = ExecutionTrace(
            case_id=self.case_id,
            session_id=self.session_id,
            timestamp=datetime.now(),
            round_number=self.current_round,
            action_type=action_type,
            actor=actor,
            content=content,
            structured_data=structured_data,
            cost_impact=cost_impact
        )
        self.traces.append(trace)
        
        if cost_impact:
            self.total_cost += cost_impact
            # Keep case_state in sync
            self.case_state.cumulative_cost += cost_impact
            
        # Create diagnostic action for stagnation detection
        if action_type in [ActionType.ASK_QUESTIONS, ActionType.ORDER_TESTS, ActionType.MAKE_DIAGNOSIS]:
            diagnostic_action = DiagnosticAction(
                action_type=action_type.value,
                content=content,
                reasoning=f"Action by {actor}",
                round_number=self.current_round
            )
            self.case_state.add_action(diagnostic_action)
            
    def add_agent_message(self, agent_role: str, message_type: str, content: str,
                         structured_data: Optional[Dict[str, Any]] = None):
        """Add a message from one of the specialized agents"""
        message = AgentMessage(
            agent_role=agent_role,
            timestamp=datetime.now(),
            message_type=message_type,
            content=content,
            structured_data=structured_data
        )
        self.agent_messages.append(message)
        
    def increment_round(self):
        """Move to the next diagnostic round"""
        self.current_round += 1
        # Keep case_state in sync
        self.case_state.current_round = self.current_round
        # Reset deliberation state for new round
        self.deliberation_state.reset_for_new_round()
    
    def check_stagnation(self, proposed_action: DiagnosticAction) -> bool:
        """Check if the proposed action would cause stagnation"""
        return self.case_state.is_stagnating(proposed_action)
    
    def add_evidence(self, evidence: str) -> None:
        """Add evidence to the case state"""
        self.case_state.add_evidence(evidence)
    
    def update_differential_diagnosis(self, diagnosis_dict: Dict[str, float]) -> None:
        """Update differential diagnosis probabilities"""
        self.case_state.update_differential(diagnosis_dict)
    
    def get_leading_diagnosis_confidence(self) -> Tuple[str, float]:
        """Get the leading diagnosis and its confidence"""
        diagnosis = self.case_state.get_leading_diagnosis()
        confidence = self.case_state.get_max_confidence()
        return diagnosis, confidence
    
    def get_structured_hypotheses(self) -> List[DiagnosticHypothesis]:
        """Extract structured hypotheses from agent messages with enhanced evidence tracking"""
        hypotheses = []
        for msg in reversed(self.agent_messages):
            if msg.agent_role == "Dr. Hypothesis" and msg.structured_data:
                try:
                    hyp_data = msg.structured_data.get('hypotheses', [])
                    for h in hyp_data:
                        if isinstance(h, dict):
                            # Enhanced evidence extraction with proper handling of new format
                            supporting_evidence = []
                            contradictory_evidence = []
                            
                            # Handle both old list format and new structured format
                            supp_ev_data = h.get('supporting_evidence', [])
                            for ev in supp_ev_data:
                                if isinstance(ev, dict):
                                    supporting_evidence.append(ev.get('description', str(ev)))
                                else:
                                    supporting_evidence.append(str(ev))
                            
                            contra_ev_data = h.get('contradictory_evidence', [])
                            for ev in contra_ev_data:
                                if isinstance(ev, dict):
                                    contradictory_evidence.append(ev.get('description', str(ev)))
                                else:
                                    contradictory_evidence.append(str(ev))
                            
                            hypotheses.append(DiagnosticHypothesis(
                                condition=h.get('condition', 'Unknown'),
                                probability=float(h.get('probability', 0.0)),
                                reasoning=h.get('reasoning', 'No reasoning'),
                                supporting_evidence=supporting_evidence,
                                contradictory_evidence=contradictory_evidence
                            ))
                    if hypotheses:
                        return hypotheses[:3]  # Return top 3
                except Exception:
                    continue
        return []
    
    def get_confidence_assessment(self) -> Dict[str, Any]:
        """Extract latest confidence assessment from Dr. Hypothesis"""
        for msg in reversed(self.agent_messages):
            if msg.agent_role == "Dr. Hypothesis" and msg.structured_data:
                confidence_data = msg.structured_data.get('confidence_assessment', {})
                if confidence_data:
                    return confidence_data
                # Fallback to legacy format
                confidence_level = msg.structured_data.get('confidence_level', 'medium')
                return {
                    "overall_confidence": 0.5,
                    "leading_hypothesis_strength": 0.5,
                    "evidence_completeness": 0.5,
                    "diagnostic_clarity": 0.5,
                    "uncertainty_factors": ["Legacy format"],
                    "confidence_level": confidence_level
                }
        return {
            "overall_confidence": 0.3,
            "leading_hypothesis_strength": 0.3,
            "evidence_completeness": 0.3,
            "diagnostic_clarity": 0.3,
            "uncertainty_factors": ["No assessment available"],
            "confidence_level": "low"
        }
    
    def get_contradictory_evidence_impact(self) -> Dict[str, Any]:
        """Analyze impact of contradictory evidence on diagnostic confidence"""
        contradictory_evidence = []
        total_impact = 0.0
        
        for msg in reversed(self.agent_messages):
            if msg.agent_role == "Dr. Hypothesis" and msg.structured_data:
                hypotheses_data = msg.structured_data.get('hypotheses', [])
                for hyp_data in hypotheses_data:
                    if isinstance(hyp_data, dict):
                        contra_ev_data = hyp_data.get('contradictory_evidence', [])
                        for ev in contra_ev_data:
                            if isinstance(ev, dict):
                                contradictory_evidence.append({
                                    "condition": hyp_data.get('condition', 'Unknown'),
                                    "evidence": ev.get('description', str(ev)),
                                    "strength": ev.get('strength', 0.5),
                                    "reliability": ev.get('reliability', 0.5),
                                    "impact": ev.get('strength', 0.5) * ev.get('reliability', 0.5)
                                })
                                total_impact += ev.get('strength', 0.5) * ev.get('reliability', 0.5)
                            else:
                                contradictory_evidence.append({
                                    "condition": hyp_data.get('condition', 'Unknown'),
                                    "evidence": str(ev),
                                    "strength": 0.5,
                                    "reliability": 0.5,
                                    "impact": 0.25
                                })
                                total_impact += 0.25
                break
        
        return {
            "contradictory_evidence_count": len(contradictory_evidence),
            "total_contradictory_impact": total_impact,
            "average_impact": total_impact / len(contradictory_evidence) if contradictory_evidence else 0.0,
            "evidence_details": contradictory_evidence,
            "confidence_reduction": min(total_impact * 0.2, 0.5)  # Max 50% confidence reduction
        }

class BaseSpecializedAgent:
    """Base class for all specialized diagnostic agents"""
    
    def __init__(self, role_name: str, client: AsyncOpenAI):
        self.role_name = role_name
        self.client = client
        self.model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        
    async def _call_llm(self, system_prompt: str, user_message: str, 
                       temperature: float = 0.7) -> str:
        """Make an async call to the language model"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in LLM call: {str(e)}"
    
    async def contribute(self, case_info: str, previous_findings: List[str], 
                        current_hypotheses: List[DiagnosticHypothesis],
                        session: CaseExecutionSession) -> Dict[str, Any]:
        """Override this method in specialized agents"""
        raise NotImplementedError("Subclasses must implement contribute method")

class DrHypothesis(BaseSpecializedAgent):
    """
    Enhanced Dr. Hypothesis - Maintains probability-ranked differential diagnosis
    with structured Bayesian reasoning, function calling, and advanced confidence assessment
    """
    
    def __init__(self, client: AsyncOpenAI):
        super().__init__("Dr. Hypothesis", client)
        self.bayesian_helper = BayesianReasoningHelper()
        
    def _calculate_confidence_factors(self, hypotheses: List[Dict], evidence_items: List[str]) -> Dict[str, float]:
        """Calculate confidence factors based on hypothesis and evidence characteristics"""
        if not hypotheses:
            return {"evidence_completeness": 0.0, "diagnostic_clarity": 0.0, "leading_hypothesis_strength": 0.0}
        
        # Leading hypothesis strength
        max_prob = max(h.get("probability", 0.0) for h in hypotheses)
        leading_strength = max_prob
        
        # Evidence completeness (based on number and type of evidence)
        evidence_count = len(evidence_items)
        evidence_completeness = min(evidence_count / 5.0, 1.0)  # Normalize to 0-1
        
        # Diagnostic clarity (probability separation between top hypotheses)
        if len(hypotheses) >= 2:
            probs = sorted([h.get("probability", 0.0) for h in hypotheses], reverse=True)
            prob_separation = probs[0] - probs[1] if len(probs) > 1 else probs[0]
            diagnostic_clarity = min(prob_separation * 2.0, 1.0)  # Scale separation
        else:
            diagnostic_clarity = max_prob
        
        return {
            "evidence_completeness": evidence_completeness,
            "diagnostic_clarity": diagnostic_clarity,
            "leading_hypothesis_strength": leading_strength
        }
        
    def _apply_bayesian_updates(self, current_hypotheses: List[DiagnosticHypothesis], 
                               new_evidence: List[str]) -> List[Dict[str, Any]]:
        """Apply Bayesian updates to existing hypotheses based on new evidence"""
        updated_hypotheses = []
        
        for hyp in current_hypotheses:
            # Start with current probability as prior
            prior_prob = hyp.probability
            
            # Analyze new evidence impact
            supporting_evidence = []
            contradictory_evidence = []
            
            for evidence in new_evidence:
                # Simple heuristic to categorize evidence (in real implementation, this would be more sophisticated)
                if any(keyword in evidence.lower() for keyword in ["consistent", "supports", "confirms"]):
                    supporting_evidence.append({
                        "description": evidence,
                        "impact_type": "supporting",
                        "strength": 0.7,
                        "reliability": 0.8,
                        "source": "clinical"
                    })
                elif any(keyword in evidence.lower() for keyword in ["rules out", "negative", "inconsistent"]):
                    contradictory_evidence.append({
                        "description": evidence,
                        "impact_type": "contradictory", 
                        "strength": 0.6,
                        "reliability": 0.8,
                        "source": "clinical"
                    })
            
            # Calculate likelihood ratio and posterior probability
            net_lr = 1.0
            for supp_ev in supporting_evidence:
                lr = self.bayesian_helper.estimate_likelihood_ratio(supp_ev["strength"], "suggestive")
                net_lr *= lr
                
            for contra_ev in contradictory_evidence:
                lr = self.bayesian_helper.estimate_likelihood_ratio(contra_ev["strength"], "contradictory")
                net_lr *= lr
            
            # Calculate posterior probability
            posterior_prob = self.bayesian_helper.calculate_posterior(prior_prob, net_lr)
            
            updated_hypotheses.append({
                "condition": hyp.condition,
                "probability": posterior_prob,
                "reasoning": f"Updated from {prior_prob:.3f} to {posterior_prob:.3f} based on new evidence. {hyp.reasoning}",
                "supporting_evidence": supporting_evidence,
                "contradictory_evidence": contradictory_evidence,
                "bayesian_updates": [{
                    "prior_probability": prior_prob,
                    "posterior_probability": posterior_prob,
                    "likelihood_ratio": net_lr,
                    "evidence_weight": 0.8,
                    "reasoning": f"Applied likelihood ratio {net_lr:.2f} based on {len(new_evidence)} new evidence items"
                }],
                "confidence_factors": {}
            })
        
        return updated_hypotheses
        
    async def contribute(self, case_info: str, previous_findings: List[str], 
                        current_hypotheses: List[DiagnosticHypothesis],
                        session: CaseExecutionSession) -> Dict[str, Any]:
        
        # Apply Bayesian updates if we have current hypotheses
        if current_hypotheses and previous_findings:
            updated_hypotheses = self._apply_bayesian_updates(current_hypotheses, previous_findings[-3:])
        else:
            updated_hypotheses = []
        
        system_prompt = """You are Dr. Hypothesis, an expert in differential diagnosis and advanced Bayesian reasoning.

Your enhanced capabilities include:
1. Structured Bayesian probability updates with explicit prior/posterior calculations
2. Evidence classification and impact assessment (supporting vs contradictory)
3. Numerical confidence assessment with detailed factors
4. Function-calling approach for reliable differential diagnosis output

You must respond with a complete JSON structure that follows this exact format:
{
    "hypotheses": [
        {
            "condition": "Primary condition name",
            "probability": 0.XX,
            "reasoning": "Detailed clinical reasoning with evidence analysis",
            "supporting_evidence": [
                {
                    "description": "Evidence description", 
                    "impact_type": "supporting",
                    "strength": 0.X,
                    "reliability": 0.X,
                    "source": "clinical/lab/imaging/history"
                }
            ],
            "contradictory_evidence": [
                {
                    "description": "Contradictory evidence",
                    "impact_type": "contradictory", 
                    "strength": 0.X,
                    "reliability": 0.X,
                    "source": "clinical/lab/imaging/history"
                }
            ],
            "bayesian_updates": [
                {
                    "prior_probability": 0.XX,
                    "posterior_probability": 0.XX,
                    "likelihood_ratio": X.X,
                    "evidence_weight": 0.X,
                    "reasoning": "Explanation of Bayesian update"
                }
            ],
            "confidence_factors": {
                "evidence_completeness": 0.X,
                "diagnostic_clarity": 0.X
            }
        }
    ],
    "bayesian_updates": "Overall explanation of probability updates and reasoning",
    "confidence_assessment": {
        "overall_confidence": 0.XX,
        "leading_hypothesis_strength": 0.XX,
        "evidence_completeness": 0.XX,
        "diagnostic_clarity": 0.XX,
        "uncertainty_factors": ["factor1", "factor2"],
        "confidence_level": "low/medium/high"
    },
    "differential_reasoning": "Detailed reasoning for differential diagnosis ranking",
    "key_discriminating_features": ["feature1", "feature2", "feature3"]
}

CRITICAL: Provide numerical confidence assessments and explicit Bayesian reasoning."""

        # Prepare enhanced context
        findings_text = "\n".join(previous_findings) if previous_findings else "No additional findings yet."
        
        if updated_hypotheses:
            current_hyp_text = "\nPrevious hypotheses with Bayesian updates:\n" + \
                "\n".join([f"- {h['condition']} (Prior: {h['bayesian_updates'][0]['prior_probability']:.3f} → Posterior: {h['probability']:.3f}): {h['reasoning'][:100]}..." 
                          for h in updated_hypotheses])
        elif current_hypotheses:
            current_hyp_text = "\nCurrent hypotheses:\n" + \
                "\n".join([f"- {h.condition} ({h.probability:.3f}): {h.reasoning}" 
                          for h in current_hypotheses])
        else:
            current_hyp_text = "\nNo current hypotheses established."
        
        user_message = f"""
=== CLINICAL CASE ANALYSIS ===
Initial Case: {case_info}

Previous Findings and Evidence:
{findings_text}
{current_hyp_text}

TASK: Provide comprehensive differential diagnosis with:
1. Structured Bayesian probability updates
2. Evidence classification (supporting vs contradictory)
3. Numerical confidence assessment
4. Clear discriminating features

Please analyze and update the differential diagnosis using advanced Bayesian reasoning.
"""

        response = await self._call_llm(system_prompt, user_message, temperature=0.3)
        session.add_agent_message(self.role_name, "hypothesis_update", response)
        
        # Enhanced parsing with multiple fallback strategies
        return await self._parse_structured_response(response, session)
        
    async def _parse_structured_response(self, response: str, session: CaseExecutionSession) -> Dict[str, Any]:
        """Enhanced parsing with structured validation and graceful fallbacks"""
        import json
        import re
        
        try:
            # First, try to extract and parse JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group())
                
                # Try to validate with enhanced HypothesisUpdate model
                try:
                    # Handle legacy format compatibility
                    if "confidence_level" in parsed_json and "confidence_assessment" not in parsed_json:
                        parsed_json["confidence_assessment"] = {
                            "overall_confidence": 0.5,
                            "leading_hypothesis_strength": 0.5,
                            "evidence_completeness": 0.5,
                            "diagnostic_clarity": 0.5,
                            "uncertainty_factors": ["Legacy format conversion"],
                            "confidence_level": parsed_json.get("confidence_level", "medium")
                        }
                    
                    # Ensure required fields for enhanced model
                    if "differential_reasoning" not in parsed_json:
                        parsed_json["differential_reasoning"] = parsed_json.get("bayesian_updates", "Standard differential reasoning applied")
                    
                    if "key_discriminating_features" not in parsed_json:
                        parsed_json["key_discriminating_features"] = []
                    
                    # Validate with Pydantic model
                    structured_response = HypothesisUpdate(**parsed_json)
                    structured_data = structured_response.dict()
                    
                    # Store enhanced structured data
                    session.agent_messages[-1].structured_data = structured_data
                    return structured_data
                    
                except Exception as pydantic_error:
                    # Fallback to basic validation and correction
                    validated_json = self._validate_and_correct_json(parsed_json)
                    session.agent_messages[-1].structured_data = validated_json
                    return validated_json
                    
        except json.JSONDecodeError as json_error:
            # Final fallback - create minimal structure from text
            return self._create_fallback_structure(response, session)
    
    def _validate_and_correct_json(self, parsed_json: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and correct JSON structure to ensure compatibility"""
        corrected = {
            "hypotheses": [],
            "bayesian_updates": parsed_json.get("bayesian_updates", "Bayesian reasoning applied"),
            "confidence_assessment": {
                "overall_confidence": 0.5,
                "leading_hypothesis_strength": 0.5,
                "evidence_completeness": 0.5,
                "diagnostic_clarity": 0.5,
                "uncertainty_factors": ["JSON structure correction applied"],
                "confidence_level": parsed_json.get("confidence_level", "medium")
            },
            "differential_reasoning": parsed_json.get("differential_reasoning", parsed_json.get("bayesian_updates", "Differential reasoning applied")),
            "key_discriminating_features": parsed_json.get("key_discriminating_features", [])
        }
        
        # Process hypotheses with validation
        if "hypotheses" in parsed_json and isinstance(parsed_json["hypotheses"], list):
            for hyp in parsed_json["hypotheses"][:5]:  # Limit to 5
                if isinstance(hyp, dict):
                    corrected_hyp = {
                        "condition": str(hyp.get("condition", "Unknown condition")),
                        "probability": float(max(0.0, min(1.0, hyp.get("probability", 0.3)))),
                        "reasoning": str(hyp.get("reasoning", "Clinical reasoning applied")),
                        "supporting_evidence": [],
                        "contradictory_evidence": [],
                        "bayesian_updates": [],
                        "confidence_factors": {}
                    }
                    corrected["hypotheses"].append(corrected_hyp)
        
        return corrected
    
    def _create_fallback_structure(self, response: str, session: CaseExecutionSession) -> Dict[str, Any]:
        """Create minimal fallback structure when all parsing fails"""
        fallback = {
            "hypotheses": [],
            "bayesian_updates": f"Fallback parsing applied. Original response: {response[:200]}...",
            "confidence_assessment": {
                "overall_confidence": 0.3,
                "leading_hypothesis_strength": 0.3,
                "evidence_completeness": 0.2,
                "diagnostic_clarity": 0.2,
                "uncertainty_factors": ["Parsing failed", "Fallback structure used"],
                "confidence_level": "low"
            },
            "differential_reasoning": "Fallback differential reasoning due to parsing failure",
            "key_discriminating_features": []
        }
        
        session.agent_messages[-1].structured_data = fallback
        return fallback

class DrTestChooser(BaseSpecializedAgent):
    """
    Dr. Test-Chooser - Sophisticated diagnostic test selection with discriminative value scoring,
    redundancy detection, and stepwise diagnostic approach
    """
    
    def __init__(self, client: AsyncOpenAI):
        super().__init__("Dr. Test-Chooser", client)
        self.performed_tests = []
        
    async def contribute(self, case_info: str, previous_findings: List[str], 
                        current_hypotheses: List[DiagnosticHypothesis],
                        session: CaseExecutionSession) -> Dict[str, Any]:
        
        # Generate candidate tests using LLM
        candidate_tests = await self._generate_candidate_tests(case_info, previous_findings, current_hypotheses)
        
        # Apply sophisticated scoring and filtering
        scored_tests = []
        for test in candidate_tests:
            # Calculate discriminative value
            discriminative_value = TestSelectionHelper.calculate_discriminative_score(test["test_name"], current_hypotheses)
            
            # Assess redundancy with performed tests
            redundancy = TestSelectionHelper.assess_test_redundancy(test["test_name"], self.performed_tests)
            
            # Determine stepwise reasoning
            stepwise = TestSelectionHelper.determine_stepwise_tier(test["test_name"], test.get("estimated_cost", 100), current_hypotheses)
            
            # Calculate overall priority score
            priority_score = self._calculate_priority_score(discriminative_value, redundancy, stepwise, test.get("estimated_cost", 100))
            
            # Create enhanced test recommendation
            enhanced_test = TestRecommendationItem(
                test_name=test["test_name"],
                rationale=test.get("rationale", "Diagnostic test recommendation"),
                priority=test.get("priority", 2),  # Default to medium priority
                discriminative_value=discriminative_value,
                stepwise_reasoning=stepwise,
                redundancy_assessment=redundancy,
                estimated_cost=test.get("estimated_cost", 100.0)
            )
            
            scored_tests.append(enhanced_test)
        
        # Apply stepwise filtering and ranking
        final_recommendations = self._apply_stepwise_filtering(scored_tests)
        
        # Generate comprehensive reasoning
        reasoning = self._generate_comprehensive_reasoning(final_recommendations, current_hypotheses)
        
        # Generate additional required fields for TestRecommendations
        stepwise_strategy = self._generate_stepwise_strategy(final_recommendations)
        redundancy_analysis = self._generate_redundancy_analysis(final_recommendations)
        cost_optimization_notes = self._generate_cost_optimization_notes(final_recommendations)
        
        # Create structured response
        test_recommendations = TestRecommendations(
            recommended_tests=final_recommendations,
            reasoning=reasoning,
            stepwise_strategy=stepwise_strategy,
            redundancy_analysis=redundancy_analysis,
            cost_optimization_notes=cost_optimization_notes
        )
        
        # Create JSON response for consistency with other agents
        response_json = json.dumps(test_recommendations.dict(), indent=2)
        
        # Record message and structured data
        session.add_agent_message(self.role_name, "test_recommendation", response_json)
        session.agent_messages[-1].structured_data = test_recommendations.dict()
        
        return test_recommendations.dict()
    
    async def _generate_candidate_tests(self, case_info: str, previous_findings: List[str], 
                                      current_hypotheses: List[DiagnosticHypothesis]) -> List[Dict[str, Any]]:
        """Generate initial candidate tests using LLM"""
        
        system_prompt = """You are Dr. Test-Chooser, a specialist in diagnostic test selection and evidence-based medicine.

Your role:
1. Select up to 5-8 diagnostic tests that maximally discriminate between leading hypotheses
2. Prioritize tests with highest diagnostic yield
3. Consider test characteristics: sensitivity, specificity, cost-effectiveness
4. Avoid redundant or low-yield investigations
5. Follow stepwise diagnostic approach (basic tests before advanced ones)

Format your response as JSON:
{
    "recommended_tests": [
        {
            "test_name": "specific test name",
            "rationale": "why this test discriminates between hypotheses",
            "urgency": "routine|urgent|emergent",
            "discriminative_value": "which conditions this test helps distinguish",
            "estimated_cost": estimated_cost_in_dollars
        }
    ]
}"""

        hypotheses_text = "\n".join([f"- {h.condition} ({h.probability:.2f}): {h.reasoning[:150]}..." 
                                   for h in current_hypotheses[:5]]) if current_hypotheses else "No hypotheses available."
        findings_text = "\n".join(previous_findings[-5:]) if previous_findings else "No findings yet."
        
        performed_tests_info = ""
        if self.performed_tests:
            performed_tests_info = f"\nAlready performed tests: {', '.join(self.performed_tests)}"
        
        user_message = f"""
Case: {case_info}

Current Top Hypotheses:
{hypotheses_text}

Recent Findings:
{findings_text}{performed_tests_info}

Select the most discriminative diagnostic tests to differentiate between these hypotheses.
Consider cost-effectiveness and avoid redundancy with already performed tests.
"""

        response = await self._call_llm(system_prompt, user_message)
        
        # Parse JSON response
        try:
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group())
                return parsed_json.get("recommended_tests", [])
        except Exception as e:
            print(f"Error parsing test recommendations: {e}")
        
        # Fallback recommendations
        return [
            {
                "test_name": "Complete Blood Count (CBC)",
                "rationale": "Basic screening for systemic conditions",
                "urgency": "routine",
                "estimated_cost": 50.0
            },
            {
                "test_name": "Comprehensive Metabolic Panel", 
                "rationale": "Assess organ function and metabolic status",
                "urgency": "routine",
                "estimated_cost": 75.0
            }
        ]
    
    def _calculate_priority_score(self, discriminative_value: DiscriminativeValue, 
                                 redundancy: RedundancyAssessment, stepwise: StepwiseReasoning, 
                                 cost: float) -> float:
        """Calculate overall priority score for test recommendation"""
        
        # Discriminative value component (40% weight)
        discriminative_score = discriminative_value.diagnostic_yield_score * 0.4
        
        # Non-redundancy component (30% weight)  
        redundancy_score = redundancy.incremental_value * 0.3
        
        # Cost-effectiveness component (20% weight)
        # Normalize cost (assume $1000 as high cost reference)
        cost_effectiveness = max(0, 1.0 - (cost / 1000.0)) * 0.2
        
        # Accessibility/tier component (10% weight)
        tier_multiplier = {
            "first_line": 1.0,
            "second_line": 0.8, 
            "third_line": 0.6,
            "specialized": 0.7
        }
        tier_score = tier_multiplier.get(stepwise.diagnostic_tier, 0.5) * 0.1
        
        total_score = discriminative_score + redundancy_score + cost_effectiveness + tier_score
        return min(total_score, 1.0)
    
    def _apply_stepwise_filtering(self, scored_tests: List[TestRecommendationItem]) -> List[TestRecommendationItem]:
        """Apply stepwise diagnostic approach filtering"""
        
        # Sort by discriminative value and cost effectiveness
        scored_tests.sort(key=lambda x: (x.discriminative_value.diagnostic_yield_score, -x.priority), reverse=True)
        
        # Group by diagnostic tier
        tier_groups = {
            "first_line": [],
            "second_line": [], 
            "third_line": [],
            "specialized": []
        }
        
        for test in scored_tests:
            tier = test.stepwise_reasoning.diagnostic_tier if test.stepwise_reasoning else "specialized"
            tier_groups[tier].append(test)
        
        # Apply stepwise logic
        final_recommendations = []
        
        # Always include top first-line tests
        final_recommendations.extend(tier_groups["first_line"][:3])
        
        # Include second-line tests if first-line tests are high-value
        if len(tier_groups["first_line"]) > 0 and tier_groups["first_line"][0].discriminative_value.diagnostic_yield_score > 0.7:
            final_recommendations.extend(tier_groups["second_line"][:2])
        
        # Include specialized tests if highly discriminative
        specialized_high_value = [t for t in tier_groups["specialized"] if t.discriminative_value.diagnostic_yield_score > 0.8]
        final_recommendations.extend(specialized_high_value[:1])
        
        # Remove high-redundancy tests
        final_recommendations = [t for t in final_recommendations 
                               if t.redundancy_assessment and t.redundancy_assessment.redundancy_score < 0.8]
        
        return final_recommendations[:3]  # Limit to top 3 recommendations per model constraints
    
    def _generate_comprehensive_reasoning(self, recommendations: List[TestRecommendationItem], 
                                        hypotheses: List[DiagnosticHypothesis]) -> str:
        """Generate comprehensive reasoning for test selection"""
        
        if not recommendations:
            return "No suitable test recommendations could be generated based on current hypotheses."
        
        reasoning_parts = []
        
        # Overall approach
        reasoning_parts.append("Test selection using discriminative value scoring and stepwise diagnostic approach:")
        
        # Tier-based reasoning
        tiers = {}
        for test in recommendations:
            tier = test.stepwise_reasoning.diagnostic_tier if test.stepwise_reasoning else "specialized"
            if tier not in tiers:
                tiers[tier] = []
            tiers[tier].append(test)
        
        for tier, tests in tiers.items():
            reasoning_parts.append(f"\n{tier.replace('_', ' ').title()} tests:")
            for test in tests:
                score_info = f"(priority: {test.priority}, discrimination: {test.discriminative_value.diagnostic_yield_score:.2f})"
                reasoning_parts.append(f"  - {test.test_name}: {test.rationale} {score_info}")
        
        # Redundancy and discrimination insights
        high_discrimination = [t for t in recommendations 
                             if t.discriminative_value and t.discriminative_value.diagnostic_yield_score > 0.7]
        if high_discrimination:
            reasoning_parts.append(f"\nHighly discriminative tests: {', '.join([t.test_name for t in high_discrimination])}")
        
        # Cost considerations
        total_cost = sum(t.estimated_cost for t in recommendations)
        reasoning_parts.append(f"\nTotal estimated cost: ${total_cost:.2f}")
        
        return " ".join(reasoning_parts)
    
    def _calculate_recommendation_confidence(self, recommendations: List[TestRecommendationItem], 
                                          hypotheses: List[DiagnosticHypothesis]) -> float:
        """Calculate confidence in the test recommendations"""
        
        if not recommendations:
            return 0.1
        
        # Factor in hypothesis quality
        hypothesis_confidence = sum(h.probability for h in hypotheses[:3]) / 3 if hypotheses else 0.5
        
        # Factor in test quality (lower priority number = higher priority)
        avg_priority_score = 1.0 - (sum(t.priority for t in recommendations) / len(recommendations) / 3.0)
        
        # Factor in discriminative value
        avg_discriminative = 0.5
        if recommendations[0].discriminative_value:
            discriminative_scores = [t.discriminative_value.diagnostic_yield_score for t in recommendations 
                                   if t.discriminative_value]
            if discriminative_scores:
                avg_discriminative = sum(discriminative_scores) / len(discriminative_scores)
        
        # Combined confidence
        combined_confidence = (hypothesis_confidence * 0.4 + avg_priority_score * 0.4 + avg_discriminative * 0.2)
        return min(combined_confidence, 0.95)
    
    def _generate_stepwise_strategy(self, recommendations: List[TestRecommendationItem]) -> str:
        """Generate stepwise strategy explanation"""
        if not recommendations:
            return "No stepwise strategy available due to lack of recommendations."
        
        tiers = {}
        for test in recommendations:
            tier = test.stepwise_reasoning.diagnostic_tier
            if tier not in tiers:
                tiers[tier] = []
            tiers[tier].append(test.test_name)
        
        strategy_parts = []
        if "first_line" in tiers:
            strategy_parts.append(f"First-line tests: {', '.join(tiers['first_line'])} - immediate accessibility")
        if "second_line" in tiers:
            strategy_parts.append(f"Second-line tests: {', '.join(tiers['second_line'])} - if first-line inconclusive")
        if "specialized" in tiers:
            strategy_parts.append(f"Specialized tests: {', '.join(tiers['specialized'])} - for specific discrimination")
        
        return "; ".join(strategy_parts) if strategy_parts else "Standard diagnostic approach following clinical guidelines"
    
    def _generate_redundancy_analysis(self, recommendations: List[TestRecommendationItem]) -> str:
        """Generate redundancy analysis explanation"""
        if not recommendations:
            return "No redundancy analysis available."
        
        redundant_tests = [t for t in recommendations if t.redundancy_assessment.redundancy_score > 0.3]
        non_redundant = len(recommendations) - len(redundant_tests)
        
        if redundant_tests:
            return f"Minimal redundancy detected: {len(redundant_tests)} tests have moderate overlap, {non_redundant} provide unique information"
        else:
            return f"All {len(recommendations)} recommended tests provide complementary diagnostic information with minimal overlap"
    
    def _generate_cost_optimization_notes(self, recommendations: List[TestRecommendationItem]) -> str:
        """Generate cost optimization notes"""
        if not recommendations:
            return "No cost optimization analysis available."
        
        total_cost = sum(t.estimated_cost for t in recommendations if t.estimated_cost)
        avg_discrimination = sum(t.discriminative_value.diagnostic_yield_score for t in recommendations) / len(recommendations)
        
        cost_tier_counts = {}
        for test in recommendations:
            tier = test.stepwise_reasoning.cost_tier
            cost_tier_counts[tier] = cost_tier_counts.get(tier, 0) + 1
        
        notes = f"Total cost ~${total_cost:.0f}, avg discrimination {avg_discrimination:.2f}. "
        notes += f"Distribution: {', '.join([f'{k}: {v}' for k, v in cost_tier_counts.items()])}"
        
        return notes
    
    def add_performed_test(self, test_name: str):
        """Add a test to the performed tests list to avoid redundancy"""
        if test_name not in self.performed_tests:
            self.performed_tests.append(test_name)

class DrChallenger(BaseSpecializedAgent):
    """
    Dr. Challenger - Acts as devil's advocate, identifies anchoring bias,
    highlights contradictory evidence
    """
    
    def __init__(self, client: AsyncOpenAI):
        super().__init__("Dr. Challenger", client)
        
    async def contribute(self, case_info: str, previous_findings: List[str], 
                        current_hypotheses: List[DiagnosticHypothesis],
                        session: CaseExecutionSession) -> Dict[str, Any]:
        
        system_prompt = """You are Dr. Challenger, the devil's advocate who prevents diagnostic errors.

Your role:
1. Identify potential anchoring bias in current hypotheses
2. Highlight contradictory evidence that doesn't fit leading diagnoses
3. Propose alternative diagnoses that might be overlooked
4. Suggest tests that could falsify current leading diagnosis
5. Challenge assumptions and cognitive shortcuts

Format your response as JSON:
{
    "challenges": [
        {
            "target_hypothesis": "hypothesis being challenged",
            "challenge_type": "anchoring bias / contradictory evidence / alternative explanation",
            "reasoning": "detailed challenge reasoning",
            "alternative_hypothesis": "proposed alternative if applicable"
        }
    ],
    "falsifying_tests": ["tests that could disprove current leading diagnosis"],
    "overlooked_possibilities": ["diagnoses that might be missed"],
    "cognitive_bias_warnings": "warnings about potential reasoning errors"
}"""

        hypotheses_text = "\n".join([f"- {h.condition} ({h.probability:.2f}): {h.reasoning}" 
                                   for h in current_hypotheses[:3]]) if current_hypotheses else "No hypotheses to challenge."
        findings_text = "\n".join(previous_findings) if previous_findings else "No findings yet."
        
        user_message = f"""
Case: {case_info}

Current Leading Hypotheses:
{hypotheses_text}

Accumulated Findings:
{findings_text}

Challenge these hypotheses. What are we potentially missing or overlooking?
"""

        response = await self._call_llm(system_prompt, user_message)
        session.add_agent_message(self.role_name, "challenge", response)
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
            
        return {
            "challenges": [],
            "falsifying_tests": [],
            "overlooked_possibilities": [],
            "cognitive_bias_warnings": response
        }

class DrStewardship(BaseSpecializedAgent):
    """
    Dr. Stewardship - Enforces cost-conscious care, advocates for cheaper alternatives,
    vetoes low-yield expensive tests
    """
    
    def __init__(self, client: AsyncOpenAI):
        super().__init__("Dr. Stewardship", client)
        
    async def contribute(self, case_info: str, previous_findings: List[str], 
                        current_hypotheses: List[DiagnosticHypothesis],
                        session: CaseExecutionSession,
                        proposed_tests: List[TestRecommendation] = None) -> Dict[str, Any]:
        
        system_prompt = """You are Dr. Stewardship, the guardian of cost-effective and value-based care.

Your role:
1. Review proposed tests for cost-effectiveness
2. Suggest cheaper alternatives when diagnostically equivalent
3. Veto low-yield expensive tests
4. Advocate for step-wise diagnostic approach
5. Balance diagnostic yield against cost and patient burden

Format your response as JSON:
{
    "cost_analysis": [
        {
            "test_name": "test being reviewed",
            "approval_status": "approved / conditional / rejected",
            "reasoning": "cost-benefit analysis",
            "cheaper_alternative": "alternative test if applicable",
            "cost_category": "low / moderate / high / very high"
        }
    ],
    "budget_recommendation": "continue / proceed with caution / stop and reassess",
    "stewardship_notes": "overall cost-consciousness guidance"
}"""

        proposed_tests_text = ""
        if proposed_tests:
            proposed_tests_text = "\n".join([f"- {t.test_name}: {t.rationale} (Est. cost: ${t.estimated_cost or 'unknown'})" 
                                           for t in proposed_tests])
        
        current_cost = session.total_cost
        
        hypotheses_summary = "\n".join([f"- {h.condition} ({h.probability:.2f})" for h in current_hypotheses[:3]]) if current_hypotheses else "No hypotheses yet."
        
        user_message = f"""
Case: {case_info}

Current Cumulative Cost: ${current_cost:.2f}

Proposed Tests:
{proposed_tests_text or "No tests proposed yet."}

Current Hypotheses:
{hypotheses_summary}

Review these tests from a cost-effectiveness perspective. Are there cheaper alternatives?
"""

        response = await self._call_llm(system_prompt, user_message)
        session.add_agent_message(self.role_name, "stewardship_review", response)
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
            
        return {
            "cost_analysis": [],
            "budget_recommendation": "continue",
            "stewardship_notes": response
        }

class DrChecklist(BaseSpecializedAgent):
    """
    Dr. Checklist - Performs quality control, ensures valid test names,
    maintains internal consistency
    """
    
    def __init__(self, client: AsyncOpenAI):
        super().__init__("Dr. Checklist", client)
        
    async def contribute(self, case_info: str, previous_findings: List[str], 
                        current_hypotheses: List[DiagnosticHypothesis],
                        session: CaseExecutionSession) -> Dict[str, Any]:
        
        system_prompt = """You are Dr. Checklist, the quality control specialist ensuring systematic and thorough care.

Your role:
1. Assess completeness of current diagnostic workup
2. Identify missing critical information or assessments
3. Evaluate systematic approach to diagnosis
4. Flag any logical inconsistencies or gaps in reasoning
5. Provide quality assessment of current diagnostic process

Format your response as JSON:
{
    "missing_info": ["list of missing critical information"],
    "systematic_gaps": ["gaps in systematic approach"],
    "quality_concerns": ["any quality issues identified"],
    "recommended_next_steps": ["suggested next diagnostic steps"],
    "completeness_assessment": "overall assessment of diagnostic completeness",
    "quality_score": 1-10
}"""

        hypotheses_summary = "\n".join([f"- {h.condition} ({h.probability:.2f}): {h.reasoning}" for h in current_hypotheses]) if current_hypotheses else "No hypotheses available."
        findings_text = "\n".join(previous_findings) if previous_findings else "No additional findings yet."
        
        user_message = f"""
Case: {case_info}

Current Hypotheses:
{hypotheses_summary}

Accumulated Findings:
{findings_text}

Current Round: {session.current_round}
Total Cost So Far: ${session.total_cost:.2f}

Perform quality control assessment of the current diagnostic approach and identify any gaps or concerns.
"""

        response = await self._call_llm(system_prompt, user_message)
        session.add_agent_message(self.role_name, "quality_control", response)
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
            
        return {
            "missing_info": [],
            "systematic_gaps": [],
            "quality_concerns": [],
            "recommended_next_steps": [],
            "completeness_assessment": response,
            "quality_score": 5
        }

class ConsensusCoordinator(BaseSpecializedAgent):
    """
    Consensus Coordinator - Synthesizes all panel recommendations into a single consensus decision
    """
    
    def __init__(self, client: AsyncOpenAI):
        super().__init__("Consensus Coordinator", client)
        
    async def synthesize_consensus(self, case_info: str, previous_findings: List[str],
                                 session: CaseExecutionSession,
                                 panel_contributions: Dict[str, Any], 
                                 max_rounds: int = 10) -> Dict[str, Any]:
        
        # Check if this is the final round
        is_final_round = session.current_round >= max_rounds
        
        system_prompt = f"""You are the Consensus Coordinator, responsible for synthesizing the diagnostic panel's recommendations into a single consensus decision.

Your role:
1. Review all panel member contributions (Dr. Hypothesis, Dr. Test-Chooser, Dr. Challenger, Dr. Stewardship, Dr. Checklist)
2. Weigh the evidence and recommendations from each specialist
3. Make a consensus decision on the next action to take
4. Provide clear reasoning for the chosen action

You must choose exactly ONE of these three actions:
- ask_questions: When more clinical information is needed
- order_tests: When diagnostic tests will help differentiate hypotheses
- make_diagnosis: When confidence is sufficient for diagnosis

Decision Guidelines:
- Make Diagnosis: When diagnostic confidence is sufficiently high (≥85%)
- Order Tests: When tests can meaningfully differentiate between top hypotheses
- Ask Questions: When additional clinical information could be of high value to clarify or refine hypotheses

CRITICAL: {"This is the FINAL ROUND. You MUST make a diagnosis based on the best available information, regardless of confidence level. Provide the most likely diagnosis with clear reasoning about the diagnostic process and available evidence." if is_final_round else ""}

Format your response as JSON:
{{
    "consensus_action": "ask_questions | order_tests | make_diagnosis",
    "action_content": {{
        "questions": ["question1", "question2"] OR
        "tests": ["test1", "test2"] OR 
        "diagnosis": "final diagnosis",
        "confidence": 0.XX
    }},
    "reasoning": "detailed explanation of why this action was chosen{"; If this is the FINAL ROUND and the diagnosis decision is made because of it, make that clear." if is_final_round else ""}",
    "panel_synthesis": "how you weighed different panel member inputs",
    "confidence_assessment": "assessment of current diagnostic confidence"
}}"""

        # Extract key information from panel contributions with enhanced confidence assessment
        hypothesis_data = panel_contributions.get("hypothesis", {})
        test_data = panel_contributions.get("tests", {})
        challenge_data = panel_contributions.get("challenges", {})
        stewardship_data = panel_contributions.get("stewardship", {})
        checklist_data = panel_contributions.get("checklist", {})
        
        # Get enhanced confidence assessment from session
        confidence_assessment = session.get_confidence_assessment()
        contradictory_impact = session.get_contradictory_evidence_impact()
        
        # Calculate numerical confidence threshold for decision making
        overall_confidence = confidence_assessment.get("overall_confidence", 0.3)
        leading_strength = confidence_assessment.get("leading_hypothesis_strength", 0.3)
        evidence_completeness = confidence_assessment.get("evidence_completeness", 0.3)
        
        # Adjust confidence based on contradictory evidence
        confidence_reduction = contradictory_impact.get("confidence_reduction", 0.0)
        adjusted_confidence = max(overall_confidence - confidence_reduction, 0.1)
        
        # Format panel contributions for the LLM
        panel_summary = f"""
=== Dr. Hypothesis Assessment ===
{json.dumps(hypothesis_data, indent=2)}

=== Dr. Test-Chooser Recommendations ===
{json.dumps(test_data, indent=2)}

=== Dr. Challenger Analysis ===
{json.dumps(challenge_data, indent=2)}

=== Dr. Stewardship Review ===
{json.dumps(stewardship_data, indent=2)}

=== Dr. Checklist Quality Control ===
{json.dumps(checklist_data, indent=2)}
"""

        findings_text = "\n".join(previous_findings) if previous_findings else "No additional findings yet."
        
        # Decision guidance based on numerical thresholds
        if adjusted_confidence >= 0.85:
            confidence_guidance = "HIGH CONFIDENCE: Strong recommendation for diagnosis"
        elif adjusted_confidence >= 0.65:
            confidence_guidance = "MEDIUM CONFIDENCE: Consider diagnosis or targeted testing"
        elif adjusted_confidence >= 0.45:
            confidence_guidance = "LOW-MEDIUM CONFIDENCE: Additional testing likely needed"
        else:
            confidence_guidance = "LOW CONFIDENCE: More information gathering required"

        user_message = f"""
Case: {case_info}

Accumulated Findings:
{findings_text}

Current Round: {session.current_round} of {max_rounds} {"(FINAL ROUND - MUST DIAGNOSE)" if is_final_round else ""}
Total Cost So Far: ${session.total_cost:.2f}

=== ENHANCED CONFIDENCE ASSESSMENT ===
Overall Confidence: {overall_confidence:.3f}
Leading Hypothesis Strength: {leading_strength:.3f}
Evidence Completeness: {evidence_completeness:.3f}
Contradictory Evidence Impact: -{confidence_reduction:.3f}
ADJUSTED CONFIDENCE: {adjusted_confidence:.3f}
Guidance: {confidence_guidance}

Contradictory Evidence Summary:
- Count: {contradictory_impact.get('contradictory_evidence_count', 0)} items
- Total Impact: {contradictory_impact.get('total_contradictory_impact', 0.0):.3f}
- Confidence Reduction: {confidence_reduction:.3f}

Panel Member Contributions:
{panel_summary}

DECISION THRESHOLDS:
- Diagnose: Adjusted confidence ≥ 0.85 OR final round
- Test: Adjusted confidence 0.45-0.84 AND tests can discriminate
- Question: Adjusted confidence < 0.45 AND questions can clarify

Based on all panel member inputs and numerical confidence assessment, determine the consensus action. Consider:
1. Enhanced diagnostic confidence from Dr. Hypothesis (numerical assessment)
2. Available tests from Dr. Test-Chooser with discriminative value
3. Concerns and contradictory evidence from Dr. Challenger
4. Cost-effectiveness from Dr. Stewardship
5. Quality and completeness from Dr. Checklist

{"FINAL ROUND REQUIREMENT: You must provide a diagnosis based on the best available evidence, even if confidence is lower than ideal. Select the most probable diagnosis from Dr. Hypothesis's assessment and provide clear reasoning about the diagnostic reasoning process." if is_final_round else "Choose the most appropriate action using the numerical confidence thresholds above."}
"""

        response = await self._call_llm(system_prompt, user_message)
        session.add_agent_message(self.role_name, "consensus_decision", response)
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                # Force diagnosis on final round if not already chosen
                if is_final_round and result.get("consensus_action") != "make_diagnosis":
                    # Extract the most likely diagnosis from panel contributions
                    hypotheses = panel_contributions.get("hypothesis", {}).get("hypotheses", [])
                    if hypotheses:
                        best_hypothesis = hypotheses[0]
                        diagnosis = best_hypothesis.get("condition", "Unknown diagnosis")
                        confidence = best_hypothesis.get("probability", 0.5)
                    else:
                        diagnosis = "Unable to determine specific diagnosis based on available information"
                        confidence = 0.3
                    
                    return {
                        "consensus_action": "make_diagnosis",
                        "action_content": {
                            "diagnosis": diagnosis,
                            "confidence": confidence
                        },
                        "reasoning": f"FINAL ROUND: Making diagnosis based on best available evidence. Original consensus action was '{result.get('consensus_action')}', but final round requires diagnosis. {result.get('reasoning', '')}",
                        "panel_synthesis": result.get("panel_synthesis", response),
                        "confidence_assessment": f"Final round forced diagnosis with confidence {confidence:.2f}"
                    }
                
                return result
        except:
            pass
            
        # Fallback response - force diagnosis on final round, ask questions otherwise
        if is_final_round:
            # Extract diagnosis from panel contributions for fallback
            hypotheses = panel_contributions.get("hypothesis", {}).get("hypotheses", [])
            if hypotheses:
                diagnosis = hypotheses[0].get("condition", "Unknown diagnosis")
                confidence = hypotheses[0].get("probability", 0.3)
            else:
                diagnosis = "Unable to determine specific diagnosis - insufficient information"
                confidence = 0.2
                
            return {
                "consensus_action": "make_diagnosis", 
                "action_content": {
                    "diagnosis": diagnosis,
                    "confidence": confidence
                },
                "reasoning": "FINAL ROUND: JSON parsing failed, but final round requires diagnosis. Making best determination from available panel inputs.",
                "panel_synthesis": f"JSON parsing error, using fallback diagnosis: {response[:200]}...",
                "confidence_assessment": f"Low confidence fallback diagnosis ({confidence:.2f}) due to parsing error"
            }
        else:
            return {
                "consensus_action": "ask_questions",
                "action_content": {
                    "questions": ["What additional clinical information would be most helpful for diagnosis?"]
                },
                "reasoning": "JSON parsing failed, defaulting to request for more information",
                "panel_synthesis": response,
                "confidence_assessment": "Unable to assess"
            }

class DiagnosticOrchestrator:
    """
    Main orchestrator that coordinates the multi-agent diagnostic process
    following the MAI-DxO pattern
    """
    
    def __init__(self, azure_openai_endpoint: str = None, azure_openai_key: str = None):
        # Initialize Azure OpenAI client
        endpoint = azure_openai_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = azure_openai_key or os.getenv("AZURE_OPENAI_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    
        # Debug output (remove after testing)
        print(f"Debug - OpenAI version: {openai.__version__}")
        print(f"Debug - Endpoint: {endpoint}")
        print(f"Debug - API Key: {'***' + api_key[-4:] if api_key else 'None'}")
        print(f"Debug - API Version: {api_version}")
        
        if not endpoint or not api_key:
            raise ValueError("Azure OpenAI endpoint and key must be provided")
            
        # Ensure endpoint has https:// prefix
        if not endpoint.startswith('https://'):
            endpoint = f"https://{endpoint}"
            
        # Use base_url approach for Azure OpenAI with standard openai library
        base_url = f"{endpoint.rstrip('/')}/openai/v1/"
        
        print(f"Debug - Constructed base_url: {base_url}")

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            # default_headers={
            #     "api-version": api_version
            # }
        )

        print(f"Debug - Using base_url initialization: {base_url}")
        
        # Initialize specialized agents
        self.dr_hypothesis = DrHypothesis(self.client)
        self.dr_test_chooser = DrTestChooser(self.client)
        self.dr_challenger = DrChallenger(self.client)
        self.dr_stewardship = DrStewardship(self.client)
        self.dr_checklist = DrChecklist(self.client)
        self.consensus_coordinator = ConsensusCoordinator(self.client)
        
        # Execution sessions
        self.active_sessions: Dict[str, CaseExecutionSession] = {}
        
    async def run_diagnostic_case(self, case_info: str, max_rounds: int = 10,
                                 budget_limit: Optional[float] = None,
                                 execution_mode: str = "unconstrained") -> CaseExecutionSession:
        """
        Execute a complete diagnostic case using the MAI-DxO orchestration pattern
        
        Args:
            case_info: Initial case presentation text
            max_rounds: Maximum number of diagnostic rounds
            budget_limit: Optional budget constraint
            execution_mode: "instant", "questions_only", "unconstrained"
        
        Returns:
            CaseExecutionSession with complete execution trace
        """
        case_id = str(uuid.uuid4())
        session = CaseExecutionSession(case_id, case_info)
        self.active_sessions[case_id] = session
        self._current_session = session  # Store for cost tracking
        
        # Diagnostic orchestration started - no separate trace needed
        
        current_hypotheses: List[DiagnosticHypothesis] = []
        accumulated_findings: List[str] = []
        
        # Handle different execution modes
        if execution_mode == "instant":
            return await self._instant_diagnosis(session, case_info)
        elif execution_mode == "questions_only":
            return await self._questions_only_mode(session, case_info)
        
        # Main diagnostic loop - each round results in exactly one of three actions
        for round_num in range(max_rounds):
            session.increment_round()
            
            # Check budget constraints before starting round
            if budget_limit and session.total_cost >= budget_limit:
                # Force diagnosis due to budget constraints
                if current_hypotheses:
                    session.final_diagnosis = current_hypotheses[0].condition
                    session.confidence_score = current_hypotheses[0].probability
                else:
                    session.final_diagnosis = "Insufficient data - budget limit reached"
                    session.confidence_score = 0.3
                
                session.add_trace(
                    ActionType.MAKE_DIAGNOSIS,
                    "Panel Consensus", 
                    f"Final Diagnosis (Budget Limited): {session.final_diagnosis}",
                    {"confidence": session.confidence_score, "reason": "budget_limit"}
                )
                break
            
            # Execute panel deliberation - each agent contributes once
            panel_contributions = await self._execute_panel_deliberation(
                session, case_info, accumulated_findings, current_hypotheses
            )
            
            # Update current hypotheses from Dr. Hypothesis contribution
            current_hypotheses = self._parse_hypotheses_from_response(
                panel_contributions.get("hypothesis", {})
            )
                
            # Consensus Coordinator synthesizes panel input into final decision
            consensus_result = await self.consensus_coordinator.synthesize_consensus(
                case_info, accumulated_findings, session, panel_contributions, max_rounds
            )
            
            # Execute the consensus decision
            consensus_action = consensus_result.get("consensus_action")
            action_content = consensus_result.get("action_content", {})
            reasoning = consensus_result.get("reasoning", "")
            
            # Check for stagnation before executing action
            if consensus_action in [ActionType.ASK_QUESTIONS.value, ActionType.ORDER_TESTS.value]:
                content = action_content.get("questions" if consensus_action == ActionType.ASK_QUESTIONS.value else "tests", [])
                proposed_action = DiagnosticAction(
                    action_type=consensus_action,
                    content=content,
                    reasoning=reasoning,
                    round_number=session.current_round
                )
                
                if session.check_stagnation(proposed_action):
                    # Force diagnosis if stagnation detected
                    session.deliberation_state.stagnation_detected = True
                    if current_hypotheses:
                        session.final_diagnosis = current_hypotheses[0].condition
                        session.confidence_score = current_hypotheses[0].probability
                    else:
                        session.final_diagnosis = "Insufficient data - stagnation detected"
                        session.confidence_score = 0.2
                    
                    session.add_trace(
                        ActionType.MAKE_DIAGNOSIS,
                        "Panel Consensus",
                        f"Final Diagnosis (Stagnation Detected): {session.final_diagnosis}",
                        {"confidence": session.confidence_score, "reason": "stagnation_detected", "stagnated_action": str(proposed_action)}
                    )
                    break
            
            if consensus_action == ActionType.MAKE_DIAGNOSIS.value:
                session.final_diagnosis = action_content.get("diagnosis", "Unknown diagnosis")
                session.confidence_score = action_content.get("confidence", 0.0)
                session.add_trace(
                    ActionType.MAKE_DIAGNOSIS,
                    "Consensus Coordinator",
                    f"Final Diagnosis: {session.final_diagnosis}",
                    {
                        "confidence": session.confidence_score,
                        "reasoning": reasoning,
                        "panel_synthesis": consensus_result.get("panel_synthesis", ""),
                        "round": round_num + 1
                    }
                )
                break
                
            elif consensus_action == ActionType.ORDER_TESTS.value:
                # Execute ordered tests and incorporate results for next round
                tests_to_order = action_content.get("tests", [])
                test_results, test_costs = await self._simulate_test_execution(tests_to_order)
                accumulated_findings.extend(test_results)
                
                # Update enhanced state tracking
                for test in tests_to_order:
                    session.case_state.add_test(test)
                for result in test_results:
                    session.add_evidence(result)
                session.add_trace(
                    ActionType.ORDER_TESTS,
                    "Consensus Coordinator",
                    f"Ordered tests: {', '.join(tests_to_order)} (Total cost: ${test_costs:.2f})",
                    {
                        "tests": tests_to_order,
                        "reasoning": reasoning,
                        "panel_synthesis": consensus_result.get("panel_synthesis", ""),
                        "round": round_num + 1,
                        "test_costs": test_costs
                    },
                    cost_impact=test_costs
                )
                
            elif consensus_action == ActionType.ASK_QUESTIONS.value:
                # Ask questions and incorporate answers for next round
                questions_to_ask = action_content.get("questions", [])
                question_results, visit_cost = await self._simulate_question_answers(questions_to_ask)
                accumulated_findings.extend(question_results)
                
                # Update enhanced state tracking
                for question in questions_to_ask:
                    session.case_state.add_question(question)
                for result in question_results:
                    session.add_evidence(result)
                session.add_trace(
                    ActionType.ASK_QUESTIONS,
                    "Consensus Coordinator",
                    f"Asked questions: {'; '.join(questions_to_ask)}" + (f" (Visit cost: ${visit_cost:.2f})" if visit_cost > 0 else ""),
                    {
                        "questions": questions_to_ask,
                        "reasoning": reasoning,
                        "panel_synthesis": consensus_result.get("panel_synthesis", ""),
                        "round": round_num + 1,
                        "visit_cost": visit_cost
                    },
                    cost_impact=visit_cost if visit_cost > 0 else None
                )
            
            else:
                # Fallback - if consensus action is not recognized, default to ask questions
                fallback_results, fallback_cost = await self._simulate_question_answers([
                    "What additional information would help with diagnosis?"
                ])
                accumulated_findings.extend(fallback_results)
                session.add_trace(
                    ActionType.ASK_QUESTIONS,
                    "Consensus Coordinator",
                    f"Unrecognized consensus action '{consensus_action}', defaulting to questions" + (f" (Visit cost: ${fallback_cost:.2f})" if fallback_cost > 0 else ""),
                    {
                        "questions": ["What additional information would help with diagnosis?"],
                        "reasoning": f"Consensus coordinator returned unrecognized action: {consensus_action}",
                        "consensus_result": consensus_result,
                        "round": round_num + 1,
                        "visit_cost": fallback_cost
                    },
                    cost_impact=fallback_cost if fallback_cost > 0 else None
                )
            
            # Round complete - new findings will be processed in next round's deliberation
        
        # Session completed - final diagnosis should have been made in the loop
        
        return session
    
    async def _execute_panel_deliberation(self, session: CaseExecutionSession, 
                                        case_info: str, findings: List[str],
                                        hypotheses: List[DiagnosticHypothesis]) -> Dict[str, Any]:
        """Execute single-stage panel deliberation where each agent contributes once"""
        
        contributions = {}
        
        # Each agent contributes once with full analysis and recommendations
        
        # Dr. Hypothesis provides differential diagnosis with probabilities
        hypothesis_contrib = await self.dr_hypothesis.contribute(
            case_info, findings, hypotheses, session
        )
        contributions["hypothesis"] = hypothesis_contrib
        current_hypotheses = self._parse_hypotheses_from_response(hypothesis_contrib)
        
        # Dr. Test-Chooser recommends tests based on current hypotheses
        test_contrib = await self.dr_test_chooser.contribute(
            case_info, findings, current_hypotheses, session
        )
        contributions["tests"] = test_contrib
        
        # Dr. Challenger identifies potential issues with current thinking
        challenge_contrib = await self.dr_challenger.contribute(
            case_info, findings, current_hypotheses, session
        )
        contributions["challenges"] = challenge_contrib
        
        # Dr. Stewardship reviews cost-effectiveness
        stewardship_contrib = await self.dr_stewardship.contribute(
            case_info, findings, current_hypotheses, session,
            self._parse_test_recommendations(test_contrib)
        )
        contributions["stewardship"] = stewardship_contrib
        
        # Dr. Checklist performs quality control assessment
        checklist_contrib = await self.dr_checklist.contribute(
            case_info, findings, current_hypotheses, session
        )
        contributions["checklist"] = checklist_contrib
        
        return contributions


    
    def _format_hypotheses_for_context(self, hypotheses: List[DiagnosticHypothesis]) -> str:
        """Format hypotheses for context in agent deliberation"""
        if not hypotheses:
            return "No current hypotheses"
        
        formatted = []
        for i, hyp in enumerate(hypotheses[:3], 1):
            formatted.append(f"{i}. {hyp.condition} ({hyp.probability:.2f}) - {hyp.reasoning[:100]}...")
        
        return "\n".join(formatted)
    
    async def _simulate_test_execution(self, tests: List[Union[str, Dict[str, Any]]]) -> Tuple[List[str], float]:
        """Simulate execution of diagnostic tests and return mock results with cost tracking"""
        results = []
        total_round_cost = 0.0
        
        for test in tests:
            # Handle both string test names and dictionary test objects
            if isinstance(test, str):
                test_name = test
            else:
                test_name = test.get("test_name", "Unknown test")
            
            # Calculate and track cost
            test_cost = cost_estimator.estimate_test_cost(test_name)
            total_round_cost += test_cost.total_cost
            
            # Mock test result - in real implementation, this would interface with actual systems
            result = f"{test_name}: [Simulated result - would be actual lab/imaging result] (Cost: ${test_cost.total_cost:.2f})"
            results.append(result)
        
        # Return both results and total cost for proper session tracking
        return results, total_round_cost
    
    async def _simulate_question_answers(self, questions: List[str]) -> Tuple[List[str], float]:
        """Simulate answers to patient questions with visit cost tracking"""
        answers = []
        visit_cost = 0.0
        
        # Questions are part of physician visit - add visit cost only once per case
        if hasattr(self, '_current_session') and not hasattr(self._current_session, '_visit_cost_added'):
            visit_cost = cost_estimator.PHYSICIAN_VISIT_COST
            self._current_session._visit_cost_added = True
        
        for question in questions:
            # Mock answer - in real implementation, this would interface with patient records
            answer = f"Q: {question} A: [Simulated patient response]"
            answers.append(answer)
        
        return answers, visit_cost
    
    def _parse_hypotheses_from_response(self, response: Dict[str, Any]) -> List[DiagnosticHypothesis]:
        """Parse agent response into DiagnosticHypothesis objects"""
        hypotheses = []
        for hyp_data in response.get("hypotheses", []):
            hypothesis = DiagnosticHypothesis(
                condition=hyp_data.get("condition", "Unknown"),
                probability=hyp_data.get("probability", 0.0),
                reasoning=hyp_data.get("reasoning", ""),
                supporting_evidence=hyp_data.get("supporting_evidence", []),
                contradictory_evidence=hyp_data.get("contradictory_evidence", [])
            )
            hypotheses.append(hypothesis)
        return sorted(hypotheses, key=lambda x: x.probability, reverse=True)
    
    def _parse_test_recommendations(self, test_response: Dict[str, Any]) -> List[TestRecommendation]:
        """Parse test recommendations from agent response"""
        recommendations = []
        for test_data in test_response.get("recommended_tests", []):
            rec = TestRecommendation(
                test_name=test_data.get("test_name", ""),
                rationale=test_data.get("rationale", ""),
                estimated_cost=test_data.get("estimated_cost"),
                priority=test_data.get("priority", 1),
                discriminative_value=test_data.get("discriminative_value", "")
            )
            recommendations.append(rec)
        return recommendations
    
    async def _instant_diagnosis(self, session: CaseExecutionSession, case_info: str) -> CaseExecutionSession:
        """Instant diagnosis mode - diagnosis based solely on initial vignette"""
        hypothesis_result = await self.dr_hypothesis.contribute(case_info, [], [], session)
        hypotheses = self._parse_hypotheses_from_response(hypothesis_result)
        
        if hypotheses:
            session.final_diagnosis = hypotheses[0].condition
            session.confidence_score = hypotheses[0].probability
        else:
            session.final_diagnosis = "Insufficient information for diagnosis"
            session.confidence_score = 0.1
            
        session.add_trace(
            ActionType.MAKE_DIAGNOSIS,
            "Dr. Hypothesis",
            f"Instant diagnosis: {session.final_diagnosis}",
            {"confidence": session.confidence_score}
        )
        
        return session
    
    async def _questions_only_mode(self, session: CaseExecutionSession, case_info: str) -> CaseExecutionSession:
        """Questions-only mode - can ask questions but cannot order diagnostic tests"""
        # Simulate asking questions and getting responses
        questions = [
            "Can you provide more details about the patient's symptoms?",
            "What is the patient's relevant medical history?",
            "What are the current vital signs and physical exam findings?"
        ]
        
        # Simulate getting additional information
        findings, visit_cost = await self._simulate_question_answers(questions)
        
        # Add trace with proper cost tracking
        session.add_trace(
            ActionType.ASK_QUESTIONS, 
            "System", 
            f"Questions-only mode: gathering additional history (Cost: ${visit_cost:.2f})",
            {"questions": questions, "visit_cost": visit_cost},
            cost_impact=visit_cost
        )
        
        # Generate diagnosis based on questions
        hypothesis_result = await self.dr_hypothesis.contribute(case_info, findings, [], session)
        hypotheses = self._parse_hypotheses_from_response(hypothesis_result)
        
        if hypotheses:
            session.final_diagnosis = hypotheses[0].condition
            session.confidence_score = hypotheses[0].probability
        else:
            session.final_diagnosis = "Insufficient information for diagnosis"
            session.confidence_score = 0.1
            
        session.add_trace(
            ActionType.MAKE_DIAGNOSIS,
            "Panel Consensus",
            f"Questions-only diagnosis: {session.final_diagnosis}"
        )
        
        return session
    
    def get_session_traces(self, case_id: str) -> List[ExecutionTrace]:
        """Get execution traces for a specific case"""
        if case_id in self.active_sessions:
            return self.active_sessions[case_id].traces
        return []
    
    def get_session_summary(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a diagnostic session"""
        if case_id not in self.active_sessions:
            return None
            
        session = self.active_sessions[case_id]
        return {
            "case_id": case_id,
            "session_id": session.session_id,
            "final_diagnosis": session.final_diagnosis,
            "confidence_score": session.confidence_score,
            "total_cost": session.total_cost,
            "rounds_completed": session.current_round,
            "created_at": session.created_at.isoformat(),
            "trace_count": len(session.traces),
            "agent_message_count": len(session.agent_messages)
        }