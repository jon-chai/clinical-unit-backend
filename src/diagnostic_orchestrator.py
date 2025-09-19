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

# Removed complex evidence and Bayesian models - simplified approach

class HypothesisItem(BaseModel):
    """Individual hypothesis with probability, reasoning, and supporting/contradictory evidence"""
    condition: str = Field(..., min_length=1, description="Medical condition name")
    probability: float = Field(..., ge=0.0, le=1.0, description="Current probability between 0 and 1")
    reasoning: str = Field(..., min_length=10, description="Clinical reasoning for this hypothesis")
    supporting_evidence: List[str] = Field(default_factory=list, max_items=5, description="Key evidence supporting this hypothesis")
    contradictory_evidence: List[str] = Field(default_factory=list, max_items=3, description="Evidence that challenges or contradicts this hypothesis")

# Removed ConfidenceAssessment - simplified to basic confidence level only

class HypothesisUpdate(BaseModel):
    """Structured output from Dr. Hypothesis"""
    hypotheses: List[HypothesisItem] = Field(..., max_items=5, description="Top hypotheses ranked by probability")
    differential_reasoning: str = Field(..., min_length=20, description="Overall reasoning for differential diagnosis")
    confidence_level: Literal["low", "medium", "high"] = Field(..., description="Overall confidence in assessment")
    
    @validator('hypotheses')
    def validate_probabilities_sum(cls, v):
        """Ensure at least one hypothesis is provided"""
        if len(v) == 0:
            raise ValueError("At least one hypothesis must be provided")
        return v

# Simplified test recommendation models - removed complex mathematical scoring

class TestRecommendationItem(BaseModel):
    """Simplified test recommendation with essential information"""
    test_name: str = Field(..., min_length=2, description="Specific diagnostic test name")
    rationale: str = Field(..., min_length=10, description="Clinical rationale for test selection")
    priority: int = Field(..., ge=1, le=3, description="Priority level (1=highest, 3=lowest)")
    estimated_cost: Optional[float] = Field(None, ge=0, description="Estimated cost in USD")
    discriminative_value: str = Field(..., min_length=5, description="Description of what conditions this test helps distinguish")

class TestRecommendations(BaseModel):
    """Simplified structured output from Dr. Test-Chooser"""
    recommended_tests: List[TestRecommendationItem] = Field(..., max_items=3, description="Up to 3 recommended tests")
    reasoning: str = Field(..., min_length=10, description="Overall test selection reasoning")

class ChallengeItem(BaseModel):
    """Individual challenge to current thinking with explicit cognitive bias identification"""
    target_hypothesis: str = Field(..., min_length=1, description="Hypothesis being challenged")
    challenge_type: Literal["anchoring_bias", "confirmation_bias", "availability_bias", "representativeness_bias", 
                           "contradictory_evidence", "alternative_explanation", "premature_closure"] = Field(..., description="Type of cognitive bias or challenge")
    reasoning: str = Field(..., min_length=10, description="Detailed challenge reasoning")
    alternative_hypothesis: Optional[str] = Field(None, description="Proposed alternative if applicable")

class ChallengeResponse(BaseModel):
    """Enhanced structured output from Dr. Challenger with explicit bias detection"""
    challenges: List[ChallengeItem] = Field(default_factory=list, max_items=5, description="List of challenges raised with specific bias types")
    falsifying_tests: List[str] = Field(default_factory=list, max_items=3, description="Tests that could disprove leading diagnosis")
    overlooked_possibilities: List[str] = Field(default_factory=list, max_items=3, description="Potentially missed diagnoses")
    cognitive_bias_warnings: str = Field(..., min_length=5, description="Overall warnings about reasoning errors and biases")

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

# Removed BayesianReasoningHelper - simplified approach without complex mathematical calculations

# Removed complex TestSelectionHelper - using simplified approach

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
    Dr. Hypothesis - Maintains probability-ranked differential diagnosis with clinical reasoning
    """
    
    def __init__(self, client: AsyncOpenAI):
        super().__init__("Dr. Hypothesis", client)
        
    # Removed complex calculation methods - using simplified clinical reasoning approach
        
    async def contribute(self, case_info: str, previous_findings: List[str], 
                        current_hypotheses: List[DiagnosticHypothesis],
                        session: CaseExecutionSession) -> Dict[str, Any]:
        """Generate differential diagnosis with structured clinical reasoning"""
        
        # Build context from case and findings
        findings_text = "\n".join(previous_findings) if previous_findings else "No additional findings yet."
        
        # Format current hypotheses if available
        if current_hypotheses:
            current_hyp_text = "\nPrevious hypotheses:\n" + \
                "\n".join([f"- {h.condition} ({h.probability:.2f}): {h.reasoning}" 
                          for h in current_hypotheses])
        else:
            current_hyp_text = "\nNo current hypotheses established."
        
        # Create the structured prompt following the new format
        system_prompt = f"""You are Dr. Hypothesis, a clinical expert specializing in differential diagnosis and clinical reasoning.

Core responsibilities:
- Generate ranked differential diagnoses based on clinical evidence
- Assign probability estimates to each diagnostic possibility  
- Provide clear clinical reasoning for each hypothesis
- Identify specific supporting evidence for each hypothesis
- Acknowledge contradictory evidence that challenges each hypothesis
- Update probabilities using Bayesian reasoning based on new evidence
- Maintain appropriate clinical confidence levels
- Explicitly detail how new findings affect your diagnostic thought process

Clinical expertise:
Your reasoning should integrate symptoms, signs, patient demographics, risk factors, and available test results to formulate the most likely diagnoses. Consider epidemiology, pathophysiology, and clinical patterns when ranking hypotheses.

Evidence analysis:
For each hypothesis, explicitly identify:
- Supporting evidence: Clinical findings, symptoms, demographics, or test results that support this diagnosis
- Contradictory evidence: Any findings or factors that argue against this diagnosis or reduce its likelihood
- Be objective and acknowledge uncertainty when evidence is mixed or incomplete

Approach:
- Systematically analyze all available clinical information
- Generate 3-5 most likely diagnoses ranked by probability
- Provide clear, evidence-based reasoning for each hypothesis
- Explicitly list supporting and contradictory evidence for each hypothesis
- Assign realistic probability estimates (totaling ≤1.0)
- Explain your Bayesian reasoning clearly and understandably
- Assess overall confidence in the differential diagnosis

Output format:
Respond with a JSON structure containing:
{{
    "hypotheses": [
        {{
            "condition": "Condition name",
            "probability": 0.XX,
            "reasoning": "Clinical reasoning for this diagnosis",
            "supporting_evidence": ["Evidence point 1", "Evidence point 2", "Evidence point 3"],
            "contradictory_evidence": ["Contradictory finding 1", "Contradictory finding 2"]
        }}
    ],
    "differential_reasoning": "Overall reasoning for the differential diagnosis ranking and analysis",
    "confidence_level": "low/medium/high"
}}

Case: {case_info}

Accumulated findings: {findings_text}{current_hyp_text}

Current Cumulative Cost: ${session.total_cost:.2f}

Provide your differential diagnosis analysis based on the available information."""

        response = await self._call_llm(system_prompt, "", temperature=0.3)
        session.add_agent_message(self.role_name, "hypothesis_update", response)
        
        # Parse structured response
        return self._parse_structured_response(response, session)
        
    def _parse_structured_response(self, response: str, session: CaseExecutionSession) -> Dict[str, Any]:
        """Parse structured response with validation and update differential diagnosis"""
        import json
        import re
        
        try:
            # Extract and parse JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group())
                
                # Validate with simplified model
                structured_response = HypothesisUpdate(**parsed_json)
                structured_data = structured_response.dict()
                
                # Update case state differential diagnosis with new probabilities
                if 'hypotheses' in structured_data:
                    differential_dict = {}
                    for hypothesis in structured_data['hypotheses']:
                        condition = hypothesis.get('condition', '')
                        probability = hypothesis.get('probability', 0.0)
                        if condition:
                            differential_dict[condition] = probability
                    
                    # Update the session's case state
                    session.update_differential_diagnosis(differential_dict)
                
                # Store structured data
                session.agent_messages[-1].structured_data = structured_data
                return structured_data
                    
        except (json.JSONDecodeError, Exception) as error:
            # Fallback - create minimal structure
            return self._create_fallback_structure(response, session)
    
    def _create_fallback_structure(self, response: str, session: CaseExecutionSession) -> Dict[str, Any]:
        """Create fallback structure when parsing fails"""
        fallback = {
            "hypotheses": [
                {
                    "condition": "Unable to parse diagnosis",
                    "probability": 0.3,
                    "reasoning": "Response parsing failed - manual review needed"
                }
            ],
            "differential_reasoning": f"Parsing failed. Original response: {response[:200]}...",
            "confidence_level": "low"
        }
        
        session.agent_messages[-1].structured_data = fallback
        return fallback

class DrTestChooser(BaseSpecializedAgent):
    """
    Dr. Test-Chooser - Intelligent diagnostic test selection specialist
    """
    
    def __init__(self, client: AsyncOpenAI):
        super().__init__("Dr. Test-Chooser", client)
        self.performed_tests = []
        
    async def contribute(self, case_info: str, previous_findings: List[str], 
                        current_hypotheses: List[DiagnosticHypothesis],
                        session: CaseExecutionSession) -> Dict[str, Any]:
        
        # Generate the Dr. Test-Chooser prompt
        system_prompt = self._create_test_chooser_prompt()
        
        # Prepare case context
        user_message = self._prepare_case_context(case_info, previous_findings, current_hypotheses)
        
        # Get LLM response
        response = await self._call_llm(system_prompt, user_message)
        
        # Parse response into structured format
        test_recommendations = self._parse_test_response(response)
        
        # Create JSON response for consistency with other agents
        response_json = json.dumps(test_recommendations, indent=2)
        
        # Record message and structured data
        session.add_agent_message(self.role_name, "test_recommendation", response_json)
        session.agent_messages[-1].structured_data = test_recommendations
        
        return test_recommendations
    
    def _create_test_chooser_prompt(self) -> str:
        """Create the Dr. Test-Chooser system prompt"""
        
        return """You are Dr. Test-Chooser, a diagnostic test selection specialist with expertise in evidence-based medicine and cost-effective diagnostic workflows.

Core responsibilities:
- Select the most discriminative diagnostic tests for differentiating between competing hypotheses
- Prioritize tests with high diagnostic yield and clinical utility
- Consider cost-effectiveness, accessibility, and patient safety
- Avoid redundant or low-value testing
- Think about test sequencing, like following a stepwise diagnostic approach (basic tests before advanced ones)

Approach:
- Focus on tests that will most effectively rule in or rule out the leading diagnostic hypotheses
- Consider test characteristics: sensitivity, specificity, positive/negative predictive values
- Balance comprehensive evaluation with efficient resource utilization
- Prioritize tests that change management decisions
- Account for patient factors: age, comorbidities, clinical stability

Output format:
Provide your test recommendations as a JSON object with exactly 1-3 tests (maximum 3):
{
    "recommended_tests": [
        {
            "test_name": "specific diagnostic test name",
            "rationale": "clear explanation of why this test is recommended and how it discriminates between hypotheses",
            "priority": 1-3 (1=high, 2=medium, 3=low priority),
            "estimated_cost": estimated_cost_in_dollars,
            "discriminative_value": "brief description of which conditions this test helps distinguish"
        }
    ],
    "reasoning": "comprehensive explanation of the test selection strategy, considering the differential diagnosis, patient factors, and diagnostic efficiency"
}"""

    def _prepare_case_context(self, case_info: str, previous_findings: List[str], 
                            current_hypotheses: List[DiagnosticHypothesis]) -> str:
        """Prepare the case context for the LLM"""
        
        # Format current hypotheses
        if current_hypotheses:
            hypotheses_text = "\n".join([
                f"- {h.condition} (probability: {h.probability:.2f}): {h.reasoning[:200]}..."
                for h in current_hypotheses[:5]
            ])
        else:
            hypotheses_text = "No specific hypotheses generated yet."
        
        # Separate test results from panel insights  
        test_results = [f for f in previous_findings if not f.startswith("Panel Discussion")]
        panel_insights = [f for f in previous_findings if f.startswith("Panel Discussion")]
        
        findings_text = "\n".join(test_results[-3:]) if test_results else "No test results yet."
        
        # Format panel discussion history
        panel_history = ""
        if panel_insights:
            panel_history = f"\n\nPrevious Panel Discussions:\n" + "\n".join(panel_insights[-3:])
        
        # Format performed tests
        performed_tests_info = ""
        if self.performed_tests:
            performed_tests_info = f"\n\nAlready performed tests (avoid redundancy): {', '.join(self.performed_tests)}"
        
        # Cost information
        cost_info = """
Cost considerations:
- Basic labs (CBC, CMP): $50-100
- Imaging (X-ray): $100-300
- Advanced imaging (CT, MRI): $500-2000
- Specialized tests (genetic, biopsy): $200-1000+
- Consider stepwise approach: basic tests before expensive ones
"""
        
        user_message = f"""Case Information:
{case_info}

Current Leading Hypotheses:
{hypotheses_text}

Recent Diagnostic Findings:
{findings_text}{panel_history}{performed_tests_info}

{cost_info}

Based on the current hypotheses and any previous panel discussions, select the most appropriate diagnostic tests to differentiate between these hypotheses. Consider input from other panel members and focus on tests with maximum discriminatory power while avoiding redundancy with already performed tests."""
        
        return user_message
    
    def _parse_test_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured test recommendations"""
        
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group())
                
                # Ensure required fields exist
                if "recommended_tests" in parsed_json:
                    # Validate and clean test items - limit to top 3 tests
                    cleaned_tests = []
                    for test in parsed_json["recommended_tests"][:3]:  # Limit to 3 tests
                        if isinstance(test, dict) and "test_name" in test:
                            cleaned_test = TestRecommendationItem(
                                test_name=test.get("test_name", "Unknown Test"),
                                rationale=test.get("rationale", "Diagnostic evaluation"),
                                priority=test.get("priority", 2),
                                estimated_cost=test.get("estimated_cost", 100.0),
                                discriminative_value=test.get("discriminative_value", "General diagnostic utility")
                            )
                            cleaned_tests.append(cleaned_test.model_dump())
                    
                    return TestRecommendations(
                        recommended_tests=cleaned_tests,
                        reasoning=parsed_json.get("reasoning", "Test recommendations based on current differential diagnosis.")
                    ).model_dump()
        
        except Exception as e:
            print(f"Error parsing test recommendations: {e}")
        
        # Fallback recommendations
        fallback_tests = [
            TestRecommendationItem(
                test_name="Complete Blood Count (CBC)",
                rationale="Basic screening for infectious, hematologic, and inflammatory conditions",
                priority=1,
                estimated_cost=50.0,
                discriminative_value="Infection vs. non-infectious causes"
            ),
            TestRecommendationItem(
                test_name="Comprehensive Metabolic Panel (CMP)",
                rationale="Assess organ function and metabolic status",
                priority=1,
                estimated_cost=75.0,
                discriminative_value="Organ dysfunction vs. systemic conditions"
            )
        ]
        
        return TestRecommendations(
            recommended_tests=[test.model_dump() for test in fallback_tests],
            reasoning="Standard diagnostic workup with basic laboratory studies to establish baseline and screen for common conditions."
        ).model_dump()
    
    def add_performed_test(self, test_name: str):
        """Add a test to the performed tests list to avoid redundancy"""
        if test_name not in self.performed_tests:
            self.performed_tests.append(test_name)

class DrChallenger(BaseSpecializedAgent):
    """
    Dr. Challenger - Enhanced cognitive bias detection specialist and devil's advocate
    """
    
    def __init__(self, client: AsyncOpenAI):
        super().__init__("Dr. Challenger", client)
        
    async def contribute(self, case_info: str, previous_findings: List[str], 
                        current_hypotheses: List[DiagnosticHypothesis],
                        session: CaseExecutionSession) -> Dict[str, Any]:
        
        system_prompt = """You are Dr. Challenger, an expert in cognitive bias detection and diagnostic error prevention. Your critical role is to identify reasoning errors and challenge assumptions that could lead to misdiagnosis.

Core responsibilities:
- Identify specific cognitive biases affecting current reasoning (anchoring, confirmation, availability, representativeness)
- Highlight contradictory evidence that doesn't support leading hypotheses
- Propose alternative diagnoses that may be overlooked due to bias
- Suggest falsifying tests that could disprove current leading diagnosis
- Challenge premature closure and ensure thorough differential consideration
- Assess overall bias risk in the diagnostic reasoning process

Cognitive Bias Expertise:
- Anchoring Bias: Over-reliance on first information received or early hypotheses
- Confirmation Bias: Seeking evidence that confirms preconceptions while ignoring contradictory data
- Availability Bias: Judging probability by ease of recalling similar cases
- Representativeness Bias: Assuming symptoms match typical presentation patterns
- Premature Closure: Accepting diagnosis before adequate verification

Approach:
- Systematically examine each hypothesis for cognitive bias vulnerabilities
- Identify evidence that contradicts or doesn't fit current hypotheses
- Consider rare but serious diagnoses that may be dismissed too quickly
- Challenge assumptions and think about alternative explanations
- Propose specific tests that could definitively rule out leading diagnoses

Output format:
{
    "challenges": [
        {
            "target_hypothesis": "specific hypothesis being challenged",
            "challenge_type": "anchoring_bias|confirmation_bias|availability_bias|representativeness_bias|contradictory_evidence|alternative_explanation|premature_closure",
            "reasoning": "detailed explanation of the bias or challenge",
            "alternative_hypothesis": "proposed alternative diagnosis if applicable"
        }
    ],
    "falsifying_tests": ["specific tests that could disprove leading diagnosis"],
    "overlooked_possibilities": ["diagnoses that might be missed due to bias"],
    "cognitive_bias_warnings": "overall assessment of reasoning errors and biases present"
}"""

        hypotheses_text = "\n".join([f"- {h.condition} ({h.probability:.2f}): {h.reasoning}" 
                                   for h in current_hypotheses[:3]]) if current_hypotheses else "No hypotheses to challenge."
        findings_text = "\n".join(previous_findings) if previous_findings else "No findings yet."
        
        # Calculate current cost for context
        current_cost = session.total_cost if hasattr(session, 'total_cost') else 0
        
        user_message = f"""Case: {case_info}

Case state: Round {session.current_round}, {len(previous_findings)} findings accumulated
Accumulated findings: {findings_text}

Current hypotheses to challenge:
{hypotheses_text}

Current cumulative cost: ${current_cost:.2f}

Systematically challenge current reasoning and identify cognitive biases. Focus on:
1. What biases might be affecting the current hypotheses?
2. What contradictory evidence is being overlooked?
3. What alternative diagnoses should be considered?
4. What tests could definitively rule out the leading diagnosis?

Be rigorous and challenge assumptions and biases in order to drive towards improved diagnostic accuracy."""

        response = await self._call_llm(system_prompt, user_message)
        session.add_agent_message(self.role_name, "challenge", response)
        
        try:
            import re
            import json
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            # Fallback parsing if JSON fails
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
                    # CRITICAL FIX: Also add to Dr. Test-Chooser's performed tests list
                    self.dr_test_chooser.add_performed_test(test)
                for result in test_results:
                    session.add_evidence(result)
                    
                # Add panel insights from this round to findings for next round
                panel_insights = self._extract_panel_insights_from_round(session, round_num + 1)
                accumulated_findings.extend(panel_insights)
                    
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
                    
                # Add panel insights from this round to findings for next round
                panel_insights = self._extract_panel_insights_from_round(session, round_num + 1)
                accumulated_findings.extend(panel_insights)
                    
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
    
    def _extract_panel_insights_from_round(self, session: CaseExecutionSession, round_num: int) -> List[str]:
        """Extract key insights from agents in current round to inform next round"""
        insights = []
        
        # Get messages from current round (last few messages)
        current_round_messages = session.agent_messages[-15:] if session.agent_messages else []
        
        for msg in current_round_messages:
            if msg.agent_role == "Dr. Challenger":
                insights.append(f"Panel Discussion Round {round_num}: Dr. Challenger raised concerns about potential diagnostic biases and alternative diagnoses")
            elif msg.agent_role == "Dr. Stewardship" and "conditional" in msg.content.lower():
                insights.append(f"Panel Discussion Round {round_num}: Dr. Stewardship flagged cost concerns and suggested more targeted testing approach")
            elif msg.agent_role == "Dr. Checklist" and "gap" in msg.content.lower():
                insights.append(f"Panel Discussion Round {round_num}: Dr. Checklist identified quality gaps requiring attention before proceeding")
        
        return insights[:3]  # Limit to top 3 insights
    
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