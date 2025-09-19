# test_run_diagnostic_orchestrator.py
import asyncio
import os
import re
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from diagnostic_orchestrator import DiagnosticOrchestrator

# Load environment variables
load_dotenv()

class MarkdownLogger:
    """Utility class to log output to both console and markdown file"""
    
    def __init__(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"diagnostic_orchestrator_test_{timestamp}.md"
        
        self.filename = filename
        self.md_content = []
        
        # Initialize markdown file with header
        self.add_md_header()
    
    def add_md_header(self):
        """Add markdown document header"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"""# MAI Diagnostic Orchestrator Test Results

**Test Run:** {timestamp}  
**OpenAI Version:** Latest  
**Test Script:** `test_run_diagnostic_orchestrator.py`

---

"""
        self.md_content.append(header)
    
    def print_and_log(self, text, md_text=None):
        """Print to console and add to markdown"""
        print(text)
        if md_text is None:
            md_text = text
        self.md_content.append(md_text + "\n")
    
    def log_heading(self, text, level=1):
        """Add a markdown heading"""
        md_heading = "#" * level + " " + text
        self.print_and_log(text, md_heading)
    
    def log_text(self, text):
        """Add regular text to markdown"""
        print(text)
        self.md_content.append(f"{text}\n\n")
    
    def log_list(self, items):
        """Add a bulleted list to markdown"""
        for item in items:
            print(f"   • {item}")
            self.md_content.append(f"- {item}\n")
        self.md_content.append("\n")
    
    def log_code_block(self, text, language=""):
        """Add a code block to markdown"""
        print(text)
        md_code = f"```{language}\n{text}\n```"
        self.md_content.append(md_code + "\n")
    
    def log_agent_json(self, timestamp, agent_role, content):
        """Special formatting for agent JSON communications with enhanced test selection display"""
        print(f"   [{timestamp}] {agent_role}:")
        
        # Enhanced display for Dr. Test-Chooser structured output
        if agent_role == "Dr. Test-Chooser" and self._is_structured_json(content):
            try:
                import json
                parsed = json.loads(content)
                if "recommended_tests" in parsed:
                    print(f"   📋 Recommended Tests: {len(parsed['recommended_tests'])}")
                    for i, test in enumerate(parsed['recommended_tests'], 1):
                        print(f"     {i}. {test.get('test_name', 'Unknown Test')}")
                        print(f"        Rationale: {test.get('rationale', 'No rationale')[:80]}...")
                        if test.get('discriminative_value'):
                            # Handle both old complex format and new simplified string format
                            disc_value = test['discriminative_value']
                            if isinstance(disc_value, dict) and 'diagnostic_yield_score' in disc_value:
                                # Old complex format
                                disc_score = disc_value.get('diagnostic_yield_score', 0)
                                print(f"        Discriminative Value: {disc_score:.3f}")
                            else:
                                # New simplified string format
                                print(f"        Discriminative Value: {disc_value}")
                        if test.get('estimated_cost'):
                            print(f"        Cost: ${test['estimated_cost']:.2f}")
                    
                    if parsed.get('cost_optimization_notes'):
                        print(f"   💰 Cost Summary: {parsed['cost_optimization_notes']}")
                else:
                    print(f"   {content}")
            except (json.JSONDecodeError, KeyError):
                print(f"   {content}")
        else:
            print(f"   {content}")
        
        # For markdown, ensure the JSON block starts on a new line
        # Check if content already starts with ```json to avoid duplicates
        if content.strip().startswith('```json'):
            self.md_content.append(f"**[{timestamp}] {agent_role}:**\n\n{content}\n\n")
        else:
            self.md_content.append(f"**[{timestamp}] {agent_role}:**\n\n```json\n{content}\n```\n\n")

    
    def log_action(self, round_num, action_type, action_data):
        """Log diagnostic actions taken in each round"""
        action_title = f"🎯 Round {round_num} Action: {action_type.upper()}"
        print(f"\n{action_title}")
        self.md_content.append(f"### {action_title}\n\n")
        
        if isinstance(action_data, dict):
            if action_data.get('action') == 'order_tests':
                tests = action_data.get('tests', [])
                print(f"   📋 Tests Ordered: {len(tests)}")
                self.md_content.append(f"**Tests Ordered:** {len(tests)}\n\n")
                for i, test in enumerate(tests, 1):
                    print(f"   {i}. {test}")
                    self.md_content.append(f"{i}. {test}\n")
                self.md_content.append("\n")
            elif action_data.get('action') == 'ask_questions':
                questions = action_data.get('questions', [])
                print(f"   ❓ Questions Asked: {len(questions)}")
                self.md_content.append(f"**Questions Asked:** {len(questions)}\n\n")
                for i, question in enumerate(questions, 1):
                    print(f"   {i}. {question}")
                    self.md_content.append(f"{i}. {question}\n")
                self.md_content.append("\n")
            elif action_data.get('action') == 'make_diagnosis':
                diagnosis = action_data.get('diagnosis', 'Unknown')
                confidence = action_data.get('confidence', 'N/A')
                print(f"   🎯 Final Diagnosis: {diagnosis}")
                print(f"   🎲 Confidence: {confidence}")
                self.md_content.append(f"**Final Diagnosis:** {diagnosis}\n\n")
                self.md_content.append(f"**Confidence:** {confidence}\n\n")
        else:
            print(f"   📄 Action Data: {str(action_data)}")
            self.md_content.append(f"**Action Data:** {str(action_data)}\n\n")
    
    def log_round_hypotheses(self, round_num, hypotheses):
        """Log the top 3 hypotheses at the end of each round with evidence"""
        print(f"\n📋 Round {round_num} - Top 3 Hypotheses:")
        self.md_content.append(f"#### 📋 Round {round_num} - Top 3 Hypotheses\n\n")
        
        if not hypotheses:
            print("   No hypotheses available")
            self.md_content.append("*No hypotheses available*\n\n")
            return
        
        for i, hyp in enumerate(hypotheses[:3], 1):
            # Handle both dictionary format and object format
            if isinstance(hyp, dict):
                condition = hyp.get('condition', f'Hypothesis {i}')
                confidence = hyp.get('confidence', hyp.get('probability', 0))
                reasoning = hyp.get('reasoning', 'No reasoning provided')
                supporting_evidence = hyp.get('supporting_evidence', [])
                contradictory_evidence = hyp.get('contradictory_evidence', [])
            else:
                # Object format (has attributes)
                condition = getattr(hyp, 'condition', str(hyp))
                confidence = getattr(hyp, 'probability', getattr(hyp, 'confidence', 0))
                reasoning = getattr(hyp, 'reasoning', 'No reasoning provided')
                supporting_evidence = getattr(hyp, 'supporting_evidence', [])
                contradictory_evidence = getattr(hyp, 'contradictory_evidence', [])
            
            confidence_pct = f"{confidence * 100:.1f}%" if confidence else "N/A"
            
            print(f"   {i}. {condition} ({confidence_pct})")
            print(f"      Reasoning: {reasoning[:100]}...")
            
            if supporting_evidence:
                print(f"      Supporting: {', '.join(supporting_evidence[:2])}{'...' if len(supporting_evidence) > 2 else ''}")
            
            if contradictory_evidence:
                print(f"      Challenges: {', '.join(contradictory_evidence[:2])}{'...' if len(contradictory_evidence) > 2 else ''}")
            
            # Markdown formatting
            self.md_content.append(f"{i}. **{condition}** - {confidence_pct}\n")
            self.md_content.append(f"   - *Reasoning:* {reasoning}\n")
            
            if supporting_evidence:
                self.md_content.append(f"   - *Supporting Evidence:*\n")
                for evidence in supporting_evidence:
                    self.md_content.append(f"     • {evidence}\n")
            
            if contradictory_evidence:
                self.md_content.append(f"   - *Contradictory Evidence:*\n")
                for evidence in contradictory_evidence:
                    self.md_content.append(f"     • {evidence}\n")
            
            self.md_content.append("\n")
        
        self.md_content.append("\n")
    
    def log_final_diagnosis_rationale(self, diagnosis, confidence, rationale):
        """Log the final diagnosis with detailed rationale"""
        print(f"\n🎯 Final Diagnosis Details:")
        print(f"   Diagnosis: {diagnosis}")
        print(f"   Confidence: {confidence}")
        print(f"   Rationale: {rationale}")
        
        self.md_content.append(f"#### 🎯 Final Diagnosis Details\n\n")
        self.md_content.append(f"**Diagnosis:** {diagnosis}\n\n")
        self.md_content.append(f"**Confidence:** {confidence}\n\n")
        self.md_content.append(f"**Rationale:** {rationale}\n\n")
    
    def log_table_row(self, *columns):
        """Add a table row to markdown"""
        text = " | ".join(str(col) for col in columns)
        print(f"   {text}")
        md_row = "| " + " | ".join(str(col) for col in columns) + " |"
        self.md_content.append(md_row + "\n")
    
    def log_table_header(self, *headers):
        """Add a table header to markdown"""
        self.log_table_row(*headers)
        separator = "|" + "|".join([" --- " for _ in headers]) + "|"
        self.md_content.append(separator + "\n")
    
    def _is_structured_json(self, content):
        """Check if content is valid JSON"""
        try:
            import json
            json.loads(content)
            return True
        except (json.JSONDecodeError, TypeError):
            return False
    
    def save_markdown(self):
        """Save markdown content to file"""
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                f.writelines(self.md_content)
            print(f"\n📄 Test results saved to: {self.filename}")
        except Exception as e:
            print(f"\n❌ Failed to save markdown file: {e}")

# Global markdown logger
md_logger = None



def _group_messages_by_round(agent_messages, traces):
    """
    Group agent messages by round using chronological boundaries.
    All messages that occur before a trace belong to that trace's round.
    """
    if not traces:
        return {1: agent_messages}
    
    # Sort traces by timestamp to establish round boundaries
    sorted_traces = sorted(traces, key=lambda t: t.timestamp)
    messages_by_round = {}
    
    for msg in agent_messages:
        msg_round = None
        
        # Find which round this message belongs to
        for trace in sorted_traces:
            # If message comes before this trace, it belongs to this round
            if msg.timestamp <= trace.timestamp:
                msg_round = trace.round_number
                break
        
        # If message comes after all traces, assign to the last round
        if msg_round is None:
            msg_round = sorted_traces[-1].round_number
        
        # Add message to its round
        if msg_round not in messages_by_round:
            messages_by_round[msg_round] = []
        messages_by_round[msg_round].append(msg)
    
    return messages_by_round

def _parse_hypotheses_from_message(content):
    """Parse hypotheses with confidence scores from Dr. Hypothesis message content"""
    hypotheses = []
    
    # Try to find JSON structure in the message - now with enhanced model support
    import re
    import json
    
    # Look for both new 'hypotheses' format and legacy 'differential_diagnoses' format
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            
            # New structured format (from enhanced Pydantic models)
            if 'hypotheses' in parsed:
                for i, hypothesis in enumerate(parsed['hypotheses'][:3]):  # Top 3
                    if isinstance(hypothesis, dict):
                        condition = hypothesis.get('condition', f'Hypothesis {i+1}')
                        confidence = hypothesis.get('probability', 0)  # New field name
                        reasoning = hypothesis.get('reasoning', 'No reasoning provided')
                        supporting_evidence = hypothesis.get('supporting_evidence', [])
                        contradictory_evidence = hypothesis.get('contradictory_evidence', [])
                        
                        hypotheses.append({
                            'condition': condition,
                            'confidence': confidence,
                            'reasoning': reasoning,
                            'supporting_evidence': supporting_evidence,
                            'contradictory_evidence': contradictory_evidence
                        })
            
            # Legacy format support
            elif 'differential_diagnoses' in parsed:
                for i, diagnosis in enumerate(parsed['differential_diagnoses'][:3]):  # Top 3
                    if isinstance(diagnosis, dict):
                        condition = diagnosis.get('condition', f'Hypothesis {i+1}')
                        confidence = diagnosis.get('confidence_score', diagnosis.get('likelihood', 0))
                        reasoning = diagnosis.get('reasoning', diagnosis.get('rationale', 'No reasoning provided'))
                        # Legacy format may not have evidence fields
                        supporting_evidence = diagnosis.get('supporting_evidence', [])
                        contradictory_evidence = diagnosis.get('contradictory_evidence', [])
                        
                        hypotheses.append({
                            'condition': condition,
                            'confidence': confidence,
                            'reasoning': reasoning,
                            'supporting_evidence': supporting_evidence,
                            'contradictory_evidence': contradictory_evidence
                        })
        except json.JSONDecodeError:
            pass
    
    # If JSON parsing failed, try to extract from text
    if not hypotheses:
        lines = content.split('\n')
        current_hypothesis = None
        
        for line in lines:
            # Look for numbered diagnoses or conditions
            if re.match(r'^\d+\.\s+', line.strip()):
                if current_hypothesis:
                    hypotheses.append(current_hypothesis)
                
                condition = re.sub(r'^\d+\.\s+', '', line.strip())
                current_hypothesis = {
                    'condition': condition,
                    'confidence': 0.5,  # Default confidence
                    'reasoning': 'Extracted from text format'
                }
            elif current_hypothesis and ('confidence' in line.lower() or 'likelihood' in line.lower()):
                # Try to extract confidence score
                conf_match = re.search(r'(\d+\.?\d*)%?', line)
                if conf_match:
                    conf_val = float(conf_match.group(1))
                    if conf_val > 1:
                        conf_val = conf_val / 100  # Convert percentage to decimal
                    current_hypothesis['confidence'] = conf_val
        
        if current_hypothesis:
            hypotheses.append(current_hypothesis)
    
    return hypotheses[:3]  # Return top 3

def _extract_final_diagnosis_rationale(session):
    """Extract the final diagnosis rationale from the session"""
    
    # Look for the most recent Dr. Hypothesis message that contains the final diagnosis
    final_messages = []
    
    for msg in reversed(session.agent_messages):
        if msg.agent_role == "Dr. Hypothesis":
            final_messages.append(msg.content)
            # Look for the message that mentions the final diagnosis
            if session.final_diagnosis.lower() in msg.content.lower():
                break
    
    if not final_messages:
        return "No rationale available for final diagnosis."
    
    # Extract reasoning from the final hypothesis message
    final_content = final_messages[0]
    
    # Try to parse JSON structure for reasoning
    import re
    import json
    
    json_match = re.search(r'\{.*\}', final_content, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            
            # Look for final diagnosis reasoning
            if 'final_diagnosis' in parsed:
                final_diag = parsed['final_diagnosis']
                if isinstance(final_diag, dict):
                    rationale = final_diag.get('rationale', final_diag.get('reasoning', ''))
                    if rationale:
                        return rationale
            
            # Look for reasoning in differential diagnoses (if final diagnosis matches top one)
            if 'differential_diagnoses' in parsed and parsed['differential_diagnoses']:
                top_diagnosis = parsed['differential_diagnoses'][0]
                if isinstance(top_diagnosis, dict):
                    condition = top_diagnosis.get('condition', '')
                    if condition and condition.lower() in session.final_diagnosis.lower():
                        return top_diagnosis.get('reasoning', top_diagnosis.get('rationale', ''))
                        
        except json.JSONDecodeError:
            pass
    
    # If JSON parsing failed, try to extract reasoning from text
    lines = final_content.split('\n')
    reasoning_lines = []
    in_reasoning_section = False
    
    for line in lines:
        line = line.strip()
        if any(keyword in line.lower() for keyword in ['rationale', 'reasoning', 'because', 'evidence', 'supports']):
            in_reasoning_section = True
            reasoning_lines.append(line)
        elif in_reasoning_section and line and not line.startswith('{'):
            reasoning_lines.append(line)
        elif in_reasoning_section and (line.startswith('{') or not line):
            break
    
    if reasoning_lines:
        return ' '.join(reasoning_lines)
    
    # Default to first few sentences of the message
    sentences = final_content.split('.')[:3]
    return '. '.join(sentences) + '.'

async def run_real_test_case():
    """Run a real diagnostic case with live LLM calls"""
    
    global md_logger
    md_logger = MarkdownLogger()
    
    # Sample clinical case (you can replace with any case)
    test_case = """
    A 45-year-old male presents to the emergency department with acute onset of severe chest pain 
    that started 2 hours ago while climbing stairs. The pain is described as crushing, radiates to 
    the left arm and jaw, and is associated with shortness of breath and nausea. He has a history 
    of hypertension and smoking. Vital signs: BP 160/95, HR 110, RR 22, O2 sat 94% on room air.
    """
    
    try:
        # Initialize the orchestrator with Azure OpenAI
        orchestrator = DiagnosticOrchestrator()
        
        md_logger.log_heading("🏥 MAI Diagnostic Orchestrator Test", 1)
        md_logger.log_heading("📋 Clinical Case", 2)
        md_logger.log_code_block(test_case.strip(), "text")
        
        # Run different execution modes
        modes_to_test = [
            ("instant", "Instant diagnosis from vignette only")
            # ("questions_only", "Questions only, no diagnostic tests"),
            # ("budgeted", "Full orchestration with $2000 budget"), # Mode not fully implemented
            # ("unconstrained", "Full orchestration, no budget limits")
        ]
        
        for mode, description in modes_to_test:
            print(f"\n🔄 Testing {mode} mode: {description}")
            print("-" * 50)
            
            # Log to markdown
            md_logger.log_heading(f"🔄 Testing {mode} mode", 2)
            md_logger.log_text(f"**Description:** {description}")
            
            # Configure parameters based on mode
            budget_limit = 2000.0 if mode == "budgeted" else None
            max_rounds = 5  # Limit rounds for testing
            
            # Execute the diagnostic session
            session = await orchestrator.run_diagnostic_case(
                case_info=test_case,
                max_rounds=max_rounds,
                budget_limit=budget_limit,
                execution_mode=mode
            )
            
            # Print results
            print_session_results(session, mode)
            
            # Wait between modes to avoid rate limiting
            await asyncio.sleep(2)
        
        # Save the markdown file at the end
        md_logger.log_heading("✅ Test Completed Successfully!", 2)
        md_logger.save_markdown()
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("1. Check your Azure OpenAI credentials are correct")
        print("2. Ensure your deployment name matches your Azure OpenAI model")
        print("3. Verify your Azure OpenAI resource has quota available")
        
        # Log error to markdown
        if md_logger:
            md_logger.log_heading("❌ ERROR OCCURRED", 2)
            md_logger.log_text(f"**Error:** {str(e)}")
            md_logger.log_text(f"**Error type:** {type(e).__name__}")
            
            import traceback
            md_logger.log_text("**Traceback:**")
            md_logger.log_code_block(traceback.format_exc(), "python")
            
            md_logger.save_markdown()

def print_session_results(session, mode):
    """Print comprehensive session results with enhanced confidence assessment"""
    global md_logger
    
    print(f"\n📊 Results for {mode} mode:")
    print(f"   🔍 Final Diagnosis: {session.final_diagnosis}")
    print(f"   🎯 Confidence: {session.confidence_score:.2f}" if session.confidence_score else "   🎯 Confidence: N/A")
    print(f"   💰 Total Cost: ${session.total_cost:.2f}")
    print(f"   🔄 Rounds Completed: {session.current_round}")
    print(f"   📝 Trace Entries: {len(session.traces)}")
    print(f"   🤖 Agent Messages: {len(session.agent_messages)}")
    
    # Enhanced confidence assessment display
    try:
        confidence_assessment = session.get_confidence_assessment()
        contradictory_impact = session.get_contradictory_evidence_impact()
        
        print(f"\n🎯 Enhanced Confidence Assessment:")
        print(f"   📈 Overall Confidence: {confidence_assessment['overall_confidence']:.3f}")
        print(f"   🎪 Leading Hypothesis Strength: {confidence_assessment['leading_hypothesis_strength']:.3f}")
        print(f"   📋 Evidence Completeness: {confidence_assessment['evidence_completeness']:.3f}")
        print(f"   🔍 Diagnostic Clarity: {confidence_assessment['diagnostic_clarity']:.3f}")
        print(f"   ❌ Contradictory Evidence Impact: -{contradictory_impact['confidence_reduction']:.3f}")
        print(f"   📊 Confidence Level: {confidence_assessment['confidence_level'].upper()}")
        
        if confidence_assessment.get('uncertainty_factors'):
            print(f"   ⚠️  Uncertainty Factors: {', '.join(confidence_assessment['uncertainty_factors'])}")
            
    except (AttributeError, KeyError) as e:
        print(f"   ⚠️  Enhanced confidence assessment not available: {e}")
    
    # Log to markdown
    if md_logger:
        md_logger.log_heading(f"📊 Results for {mode} mode", 3)
        results_list = [
            f"🔍 **Final Diagnosis:** {session.final_diagnosis}",
            f"🎯 **Confidence:** {session.confidence_score:.2f}" if session.confidence_score else "🎯 **Confidence:** N/A",
            f"💰 **Total Cost:** ${session.total_cost:.2f}",
            f"🔄 **Rounds Completed:** {session.current_round}",
            f"📝 **Trace Entries:** {len(session.traces)}",
            f"🤖 **Agent Messages:** {len(session.agent_messages)}"
        ]
        md_logger.log_list(results_list)
        
        # Enhanced confidence assessment in markdown
        try:
            confidence_assessment = session.get_confidence_assessment()
            contradictory_impact = session.get_contradictory_evidence_impact()
            
            md_logger.log_heading("🎯 Enhanced Confidence Assessment", 4)
            confidence_list = [
                f"📈 **Overall Confidence:** {confidence_assessment['overall_confidence']:.3f}",
                f"🎪 **Leading Hypothesis Strength:** {confidence_assessment['leading_hypothesis_strength']:.3f}",
                f"📋 **Evidence Completeness:** {confidence_assessment['evidence_completeness']:.3f}",
                f"🔍 **Diagnostic Clarity:** {confidence_assessment['diagnostic_clarity']:.3f}",
                f"❌ **Contradictory Evidence Impact:** -{contradictory_impact['confidence_reduction']:.3f}",
                f"📊 **Confidence Level:** {confidence_assessment['confidence_level'].upper()}"
            ]
            
            if confidence_assessment.get('uncertainty_factors'):
                confidence_list.append(f"⚠️ **Uncertainty Factors:** {', '.join(confidence_assessment['uncertainty_factors'])}")
                
            md_logger.log_list(confidence_list)
            
        except (AttributeError, KeyError):
            md_logger.log_text("*Enhanced confidence assessment not available*")
        
        # Log final diagnosis rationale if available
        if session.final_diagnosis and session.final_diagnosis != "No diagnosis reached":
            final_reasoning = _extract_final_diagnosis_rationale(session)
            if final_reasoning:
                confidence_score = session.confidence_score if session.confidence_score else 0.0
                md_logger.log_final_diagnosis_rationale(session.final_diagnosis, confidence_score, final_reasoning)
    
    # Show actions taken in each round and hypotheses
    print(f"\n🎯 Actions Taken by Round:")
    if md_logger:
        md_logger.log_heading("🎯 Actions Taken by Round", 4)
    
    # Group actions and agent messages by round
    actions_by_round = {}
    hypotheses_by_round = {}
    
    for trace in session.traces:
        round_num = trace.round_number
        if round_num not in actions_by_round:
            actions_by_round[round_num] = []
        actions_by_round[round_num].append(trace)
    
    # Extract hypotheses - try structured method first, then parse from messages
    try:
        # Use new structured method if available
        structured_hypotheses = session.get_structured_hypotheses()
        if structured_hypotheses:
            # Assign to current round
            current_round = session.current_round or 1
            hypotheses_by_round[current_round] = [
                {
                    'condition': h.condition,
                    'confidence': h.probability,
                    'reasoning': h.reasoning
                } for h in structured_hypotheses[:3]
            ]
    except (AttributeError, Exception):
        # Fallback to message parsing for older sessions or if method fails
        pass
    
    # Also extract from messages by round for comprehensive logging
    all_messages_by_round = _group_messages_by_round(session.agent_messages, session.traces)
    for round_num, round_messages in all_messages_by_round.items():
        for msg in round_messages:
            if msg.agent_role == "Dr. Hypothesis":
                hypotheses = _parse_hypotheses_from_message(msg.content)
                if hypotheses and round_num not in hypotheses_by_round:
                    hypotheses_by_round[round_num] = hypotheses
                    break  # Only use the first Dr. Hypothesis message per round
    
    for round_num in sorted(actions_by_round.keys()):
        print(f"\n   Round {round_num}:")
        if md_logger:
            md_logger.md_content.append(f"**Round {round_num}:**\n\n")
        
        # Show actions for this round
        for trace in actions_by_round[round_num]:
            timestamp = trace.timestamp.strftime("%H:%M:%S")
            action_desc = f"{trace.action_type.value.replace('_', ' ').title()}"
            content_preview = trace.content[:100] + "..." if len(trace.content) > 100 else trace.content
            print(f"     [{timestamp}] {action_desc}: {content_preview}")
            
            if md_logger:
                md_logger.md_content.append(f"- **[{timestamp}] {action_desc}:** {content_preview}\n")
        
        # Show hypotheses for this round
        if round_num in hypotheses_by_round and md_logger:
            md_logger.log_round_hypotheses(round_num, hypotheses_by_round[round_num])
        
        if md_logger:
            md_logger.md_content.append("\n")
    
    # Show panel decisions (all traces are now panel decisions)
    print(f"\n📋 Panel Decisions:")
    decision_points = []
    for trace in session.traces:
        timestamp = trace.timestamp.strftime("%H:%M:%S")
        action_name = trace.action_type.value.replace('_', ' ').title()
        print(f"   [{timestamp}] {action_name}: {trace.content}")
        decision_points.append(f"**[{timestamp}] {action_name}:** {trace.content}")
    
    if md_logger and decision_points:
        md_logger.log_heading("📋 Panel Decisions", 4)
        md_logger.log_list(decision_points)
    
    # Show agent communications grouped by round with proper JSON formatting
    print(f"\n🤖 Agent Communications by Round:")
    if md_logger:
        md_logger.log_heading("🤖 Agent Communications by Round", 4)
    
    # Group agent messages by round using improved logic
    messages_by_round = _group_messages_by_round(session.agent_messages, session.traces)
    
    # Display messages grouped by round
    for round_num in sorted(messages_by_round.keys()):
        print(f"\n   === Round {round_num} Discussions ===")
        if md_logger:
            md_logger.log_heading(f"Round {round_num} Discussions", 5)
        
        for msg in messages_by_round[round_num]:
            timestamp = msg.timestamp.strftime("%H:%M:%S")
            if md_logger:
                md_logger.log_agent_json(timestamp, msg.agent_role, msg.content)
            else:
                print(f"   [{timestamp}] {msg.agent_role}:")
                print(f"   {msg.content}")
        
        if md_logger:
            md_logger.md_content.append("\n")

async def run_detailed_single_case():
    """Run a single case and show detailed agent interactions"""
    
    global md_logger
    md_logger = MarkdownLogger()
    
    case = """
    A 28-year-old female presents with a 3-day history of fever, headache, and neck stiffness. 
    She reports photophobia and has developed a petechial rash on her trunk and extremities. 
    Temperature is 39.2°C, blood pressure 90/60 mmHg, heart rate 120 bpm.
    """
    
    orchestrator = DiagnosticOrchestrator()
    
    print("🔬 Detailed Single Case Analysis")
    print("=" * 60)
    
    md_logger.log_heading("🔬 Detailed Single Case Analysis", 1)
    md_logger.log_heading("📋 Clinical Case", 2)
    md_logger.log_code_block(case.strip(), "text")
    
    session = await orchestrator.run_diagnostic_case(
        case_info=case,
        max_rounds=3,
        execution_mode="unconstrained"
    )
    
    # Print session results using consistent format
    print_session_results(session, "unconstrained")
    
    # Save markdown at the end
    md_logger.log_heading("✅ Detailed Analysis Complete", 2)
    md_logger.save_markdown()

if __name__ == "__main__":
    # Choose which test to run:
    
    print("Select test mode:")
    print("1. Quick multi-mode test")
    print("2. Detailed single case analysis")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(run_real_test_case())
    elif choice == "2":
        asyncio.run(run_detailed_single_case())
    else:
        print("Running default multi-mode test...")
        asyncio.run(run_real_test_case())