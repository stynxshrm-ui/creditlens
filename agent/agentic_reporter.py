"""
Agentic Insight Reporter for CreditLens.

Uses Anthropic API with tool calling to:
1. Investigate drift patterns from Week 7 pipeline
2. Analyze model performance and portfolio health
3. Generate plain-English monthly summaries for stakeholders

No LangChain - raw Anthropic API to keep all logic visible and debuggable.
"""

import anthropic
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import os


class CreditLensAgent:
    """
    Agentic reporter for credit risk portfolio insights.
    
    Uses Anthropic Claude with tool calling to investigate data
    and generate narrative reports.
    """
    
    def __init__(
        self,
        api_key: str,
        tools_instance,  # CreditLensTools instance
        model: str = "claude-sonnet-4-20250514"
    ):
        """
        Initialize agent.
        
        Args:
            api_key: Anthropic API key
            tools_instance: Instance of CreditLensTools
            model: Claude model to use (Sonnet 4 recommended)
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.tools = tools_instance
        self.model = model
        self.conversation_history = []
        
    def _execute_tool(self, tool_name: str, tool_input: Dict) -> Any:
        """
        Execute a tool and return results.
        
        Args:
            tool_name: Name of the tool to call
            tool_input: Input parameters for the tool
            
        Returns:
            Tool execution result
        """
        # Map tool names to tool methods
        tool_map = {
            'get_psi_report': self.tools.get_psi_report,
            'get_model_performance': self.tools.get_model_performance,
            'get_portfolio_summary': self.tools.get_portfolio_summary,
            'get_vintage_cohort': self.tools.get_vintage_cohort,
            'get_champion_challenger_status': self.tools.get_champion_challenger_status,
            'query_portfolio': self.tools.query_portfolio
        }
        
        if tool_name not in tool_map:
            return {'error': f'Unknown tool: {tool_name}'}
            
        try:
            # Call the tool with unpacked input
            result = tool_map[tool_name](**tool_input)
            return result
        except Exception as e:
            return {'error': str(e)}
    
    def generate_monthly_report(
        self,
        investigation_prompt: Optional[str] = None
    ) -> str:
        """
        Generate monthly portfolio insight report.
        
        The agent decides what to investigate based on drift reports
        and generates a plain-English summary.
        
        Args:
            investigation_prompt: Optional specific investigation focus
            
        Returns:
            Plain-English monthly report
        """
        # Default system prompt for monthly reporting
        system_prompt = """You are a credit risk analyst generating monthly portfolio insights.

Your task:
1. Investigate recent drift detection results
2. Check model performance trends
3. Analyze portfolio risk composition
4. Identify any concerning cohorts
5. Check champion/challenger status
6. Synthesize findings into a clear narrative

Your audience is non-technical stakeholders (executives, business owners).
Use plain English. Highlight actionable insights. Be concise but complete.

Investigation strategy:
- Start by checking PSI reports for any features in RETRAIN or MONITOR status
- If drift detected, investigate which segments/cohorts are affected
- Check model performance to see if drift correlates with degradation
- Always check champion/challenger status for context
- Use SQL queries only if standard tools don't answer your question

Output format:
Executive Summary (2-3 sentences)
Key Findings (3-5 bullets)
Detailed Analysis (2-3 paragraphs)
Recommendations (actionable next steps)
"""

        # Start conversation
        user_message = investigation_prompt or """Generate this month's portfolio risk report. 

Start by checking the drift detection results to see if any features are flagged. 
Then assess model performance and portfolio health. 
Conclude with recommendations."""

        messages = [
            {"role": "user", "content": user_message}
        ]
        
        # Import tool definitions
        from agent_tools import TOOL_DEFINITIONS
        
        print("=" * 70)
        print("CreditLens Agentic Reporter - Monthly Analysis")
        print("=" * 70)
        print(f"\nStarting investigation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 70)
        
        max_turns = 10  # Prevent infinite loops
        turn = 0
        
        while turn < max_turns:
            turn += 1
            print(f"\n[Turn {turn}] Calling Claude...")
            
            # Call Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                system=system_prompt,
                messages=messages,
                tools=TOOL_DEFINITIONS
            )
            
            # Check stop reason
            if response.stop_reason == "end_turn":
                # Claude is done - extract final text
                final_text = ""
                for block in response.content:
                    if block.type == "text":
                        final_text += block.text
                        
                print(f"\n[Turn {turn}] Claude finished analysis")
                print("=" * 70)
                return final_text
                
            elif response.stop_reason == "tool_use":
                # Claude wants to use tools
                print(f"[Turn {turn}] Claude is using tools...")
                
                # Process tool calls
                tool_results = []
                
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input
                        tool_id = block.id
                        
                        print(f"  └─> {tool_name}({json.dumps(tool_input)})")
                        
                        # Execute tool
                        result = self._execute_tool(tool_name, tool_input)
                        
                        # Format tool result for Claude
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": json.dumps(result)
                        })
                
                # Add assistant message to history
                messages.append({
                    "role": "assistant",
                    "content": response.content
                })
                
                # Add tool results
                messages.append({
                    "role": "user",
                    "content": tool_results
                })
                
            else:
                # Unexpected stop reason
                print(f"\n[Turn {turn}] Unexpected stop reason: {response.stop_reason}")
                break
                
        # If we hit max turns, return what we have
        print(f"\n[Warning] Reached maximum turns ({max_turns})")
        
        final_text = "Analysis incomplete - reached turn limit."
        for msg in messages:
            if msg["role"] == "assistant":
                for block in msg.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        final_text = block["text"]
                        
        return final_text
    
    def investigate_drift(self, feature_name: str) -> str:
        """
        Deep dive into specific feature drift.
        
        Args:
            feature_name: Feature to investigate
            
        Returns:
            Analysis of why the feature drifted
        """
        system_prompt = f"""You are investigating PSI drift on the feature: {feature_name}

Your task:
1. Get the PSI report for {feature_name}
2. If PSI is elevated, investigate which portfolio segments are affected
3. Check if specific vintages show unusual patterns
4. Determine if this explains any model performance degradation
5. Provide a clear explanation of what caused the drift

Be specific - use data from tool calls to support your conclusions.
"""

        messages = [
            {"role": "user", "content": f"Why did {feature_name} drift? Investigate the root cause."}
        ]
        
        from agent_tools import TOOL_DEFINITIONS
        
        # Run investigation (similar to monthly report)
        print(f"\nInvestigating drift on: {feature_name}")
        print("-" * 70)
        
        max_turns = 8
        turn = 0
        
        while turn < max_turns:
            turn += 1
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=system_prompt,
                messages=messages,
                tools=TOOL_DEFINITIONS
            )
            
            if response.stop_reason == "end_turn":
                final_text = ""
                for block in response.content:
                    if block.type == "text":
                        final_text += block.text
                return final_text
                
            elif response.stop_reason == "tool_use":
                tool_results = []
                
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result)
                        })
                
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                
        return "Investigation incomplete."
    
    def save_report(self, report: str, output_dir: Path, report_type: str = 'monthly'):
        """
        Save generated report to file.
        
        Args:
            report: Generated report text
            output_dir: Directory to save report
            report_type: 'monthly' or 'drift_investigation'
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{report_type}_report_{timestamp}.md"
        
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(f"# CreditLens {report_type.title()} Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            f.write(report)
            
        print(f"\nReport saved: {filepath}")
        return str(filepath)


def demo_monthly_report():
    """
    Demonstrate monthly report generation.
    
    Note: Requires ANTHROPIC_API_KEY environment variable.
    """
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        print("\nTo use the agent:")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        print("  python agentic_reporter.py")
        return
    
    # Initialize tools
    from agent_tools import CreditLensTools
    
    tools = CreditLensTools(
        db_path='/tmp/demo.duckdb',
        drift_reports_dir=Path('/home/claude/drift_test_output')
    )
    
    # Initialize agent
    agent = CreditLensAgent(
        api_key=api_key,
        tools_instance=tools
    )
    
    # Generate monthly report
    report = agent.generate_monthly_report()
    
    print("\n" + "=" * 70)
    print("GENERATED REPORT")
    print("=" * 70)
    print(report)
    print("=" * 70)
    
    # Save report
    agent.save_report(
        report,
        output_dir=Path('/home/claude/agent_reports'),
        report_type='monthly'
    )


if __name__ == '__main__':
    # Check if API key is available
    if os.environ.get('ANTHROPIC_API_KEY'):
        demo_monthly_report()
    else:
        print("=" * 70)
        print("CreditLens Agentic Reporter")
        print("=" * 70)
        print("\nThis module requires an Anthropic API key.")
        print("\nSetup:")
        print("  1. Get API key from: https://console.anthropic.com/")
        print("  2. export ANTHROPIC_API_KEY='your-key-here'")
        print("  3. python agentic_reporter.py")
        print("\nThe agent will:")
        print("  - Investigate drift detection results")
        print("  - Check model performance trends")
        print("  - Analyze portfolio composition")
        print("  - Generate plain-English monthly report")
        print("=" * 70)
