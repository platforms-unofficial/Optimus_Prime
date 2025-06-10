from typing import Any, Dict, List, Literal, TypedDict, Union, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import langgraph.graph as lg
from pydantic import BaseModel, Field

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
import json


from llm_client import LLMClient
from models import MathematicalModel, Variable, Constraint, Objective, ValidationResult

# Initialize LLM client
llm_client = LLMClient()

# Update the AgentState definition
class AgentState(TypedDict):
    # The original problem description from the user
    problem_description: str
    # The mathematical model (variables, constraints, objective)
    math_model: Optional[MathematicalModel]
    # The generated solver code
    solver_code: Optional[str]
    # The solution to the LP problem
    solution: Optional[Dict[str, Any]]
    # Track the number of retries for each stage
    model_generation_attempts: int
    code_generation_attempts: int
    # Messages for the conversation
    messages: List[BaseMessage]
    # Current error if any
    error: Optional[str]
    # Detailed execution error information for code fixing
    execution_error: Optional[Dict[str, Any]]


### TODO: Have a higher level agent to generate high level scenarios 
# and then have these agents work on those scenarios
# after this agent or another will then compare the scenarios
# we can return all the options with the best one selected

# Agent 1: Mathematical Model Generator
def generate_mathematical_model(state: AgentState) -> AgentState:
    """Generate a mathematical model from the problem description."""
    try:
        # Create a parser for the MathematicalModel
        parser = PydanticOutputParser(pydantic_object=MathematicalModel)
        
        # Check if we have validation errors from a previous attempt
        previous_error = state.get("error", "")
        previous_model = state.get("math_model")
        error_context = ""

        if previous_error and previous_model:
            # Provide the previous model for refinement with clearer instructions
            error_context = f"""
            IMPORTANT: Your previous model had these specific issues:
            {previous_error}
            
            Here is your previous model:
            Variables: {[v.model_dump() for v in previous_model.variables]}
            Constraints: {[c.model_dump() for c in previous_model.constraints]}
            Objective: {previous_model.objective.model_dump()}
            
            INSTRUCTIONS FOR REFINEMENT:
            1. PRESERVE ALL CORRECT ELEMENTS from the previous model
            2. ADD the missing variables identified in the error
            3. ADD the missing constraints identified in the error
            4. CORRECT the objective function as needed
            5. DO NOT REMOVE any variables, constraints, or objective terms that were not specifically mentioned as problematic
            
            Build upon the previous model - do not start from scratch.
            """

        # Create a prompt template with the parser's format instructions
        prompt_template = PromptTemplate(
            template="""
            Create a comprehensive mathematical model for the following optimization problem 
            using linear or mixed integer programming:
            {problem_description}
            
            {error_context}
            
            ## PRELIMINARY ANALYSIS (Complete all steps before formulating the model)
            
            1. EXTRACT KEY ELEMENTS:
               - List all resources/items mentioned (e.g., products, machines, locations)
               - List all activities/decisions mentioned (e.g., production amounts, assignments)
               - List all time periods if applicable
               - Identify what is being optimized (minimized or maximized)
            
            2. IDENTIFY DECISION VARIABLES:
               - For each decision that needs to be made, create a variable
               - Specify variable type (continuous, binary, integer)
               - Include indexes needed (e.g., for different products, locations, time periods)
               - Consider if auxiliary variables are needed for complex constraints
            
            3. IDENTIFY ALL CONSTRAINTS:
               - Resource limitations (e.g., capacity, budget, time)
               - Requirements (e.g., demand must be met, minimum service levels)
               - Logical relationships (e.g., if X happens, then Y must happen)
               - Physical limitations (e.g., non-negativity, upper bounds)
               - Balance equations (e.g., flow in = flow out)
            
            4. IDENTIFY OBJECTIVE COMPONENTS:
               - All costs/profits associated with decision variables
               - Fixed costs/revenues
               - Penalties or bonuses
            
            ## MATHEMATICAL FORMULATION
            
            5. DEFINE VARIABLES FORMALLY:
               - Use meaningful, descriptive names that directly reference the problem domain
               - For roles, locations, products, etc., use their actual names (e.g., x_CustomerSupport_Philippines instead of x_r1_l2)
               - For indexed variables, use descriptive prefixes with specific entity names (e.g., assign_Engineer_London instead of a_1_3)
               - Specify domains and bounds precisely
               - Group related variables together
            
            6. FORMULATE CONSTRAINTS EXPLICITLY:
               - Express each constraint mathematically
               - Use the exact operators (≤, ≥, =)
               - Label each constraint or group of constraints
               - Ensure every constraint identified in step 3 is formulated
            
            7. CONSTRUCT THE OBJECTIVE FUNCTION:
               - Include ALL components identified in step 4
               - Ensure correct signs (+ for maximization terms, - for minimization terms)
               - Double-check coefficients against the problem description
            
            8. VERIFICATION CHECKLIST:
               - Every statement in the problem description is reflected in the model
               - All variables mentioned in constraints and objective are defined
               - No "dangling" variables (defined but unused)
               - Units are consistent across the model
               - The model is mathematically coherent
            
            {format_instructions}
            """,
            input_variables=["problem_description"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        # Generate the prompt
        prompt = prompt_template.format(
            problem_description=state['problem_description'],
            error_context=error_context
        )
        
        # Get structured output from LLM
        math_model = llm_client.send_structured_prompt(prompt, MathematicalModel)
        
        return {
            **state,
            "math_model": math_model,
            "messages": state["messages"] + [
                AIMessage(content=f"Generated mathematical model: {math_model.model_dump_json(indent=2)}")
            ],
            "model_generation_attempts": state["model_generation_attempts"] + 1,
            "error": None
        }
    except Exception as e:
        return {
            **state,
            "error": f"Error generating mathematical model: {str(e)}",
            "messages": state["messages"] + [
                AIMessage(content=f"Error generating mathematical model: {str(e)}")
            ],
            "model_generation_attempts": state["model_generation_attempts"] + 1
        }

# Agent 2: Model Validator
def validate_mathematical_model(state: AgentState) -> AgentState:
    """Validate the mathematical model against the original problem description."""
    try:
        model = state["math_model"]
        problem_description = state["problem_description"]
        
        # Check if we have a model
        if not model:
            raise ValueError("Model is missing")
            
        # First perform basic validation of the structure
        defined_vars = {v.name for v in model.variables}
        used_vars = set()
        
        # Extract variables from constraints
        for constraint in model.constraints:
            expr = constraint.expression
            for var in defined_vars:
                if var in expr:
                    used_vars.add(var)
        
        # Extract variables from objective
        obj_expr = model.objective.expression
        for var in defined_vars:
            if var in obj_expr:
                used_vars.add(var)
        
        # Check if any undefined variables are used
        if not defined_vars.issuperset(used_vars):
            raise ValueError(f"Some variables are used but not defined: {used_vars - defined_vars}")
        
        # Now use LLM to validate the model against the problem description
        # Use structured output parser instead of regex
        parser = PydanticOutputParser(pydantic_object=ValidationResult)
        
        validation_prompt = f"""
        Carefully validate if the mathematical model correctly captures the ESSENTIAL elements from the original problem description.

        Original problem description:
        {problem_description}

        Mathematical model:
        Variables: {[v.model_dump() for v in model.variables]}
        Constraints: {[c.model_dump() for c in model.constraints]}
        Objective: {model.objective.model_dump()}

        SYSTEMATIC VALIDATION:
        1. Extract the key decision elements and resources that directly affect the optimization outcome
        2. Distinguish between essential elements (required for the model) and contextual information
        3. For each essential entity, verify appropriate variables exist
        4. Extract every critical limitation or requirement mentioned
        5. Verify each critical constraint is properly represented
        6. Identify all components that affect the objective value
        7. Verify each component is represented in the objective function

        Check specifically for:
        1. Missing ESSENTIAL variables: List only variables that are REQUIRED but not defined
        2. Missing CRITICAL constraints: List only constraints that are NECESSARY but not represented
        3. Incorrect objective: Note if ANY part of the objective function is missing or incorrect

        {parser.get_format_instructions()}

        Only mark the model as invalid if it's missing elements that would materially affect the optimization result.
        DO NOT flag variables or constraints as missing if they're mentioned in the problem but not actually needed for a correct mathematical model.
        """
            
        # Get structured validation result
        validation_result = llm_client.send_structured_prompt(validation_prompt, ValidationResult)
        
        # Check if validation passed
        if validation_result.valid:
            return {
                **state,
                "error": None,
                "messages": state["messages"] + [
                    AIMessage(content="The mathematical model has been validated successfully against the problem description.")
                ]
            }
        else:
            # Create detailed error message from structured result
            error_msg = f"Model validation error: {validation_result.explanation}"
            if validation_result.missing_variables:
                error_msg += f"\nMissing variables: {', '.join(validation_result.missing_variables)}"
            if validation_result.missing_constraints:
                error_msg += f"\nMissing constraints: {', '.join(validation_result.missing_constraints)}"
            if validation_result.incorrect_objective:
                error_msg += f"\nIncorrect objective: {validation_result.incorrect_objective}"
            
            return {
                **state,
                "error": error_msg,
                "messages": state["messages"] + [
                    AIMessage(content=error_msg)
                ]
            }
        
    except Exception as e:
        return {
            **state,
            "error": f"Model validation error: {str(e)}",
            "messages": state["messages"] + [
                AIMessage(content=f"Model validation error: {str(e)}")
            ]
        }

# Agent 3: Code Generator
def generate_solver_code(state: AgentState) -> AgentState:
    """Generate PuLP solver code from the mathematical model."""
    try:
        model = state["math_model"]
        
        # Base prompt for initial code generation
        base_prompt = f"""
        Generate Python code using PuLP to solve the following linear programming problem:
        
        Variables: {[v.model_dump() for v in model.variables]}
        Constraints: {[c.model_dump() for c in model.constraints]}
        Objective: {model.objective.model_dump()}
        
        Your code MUST include a function called solve_problem() that:
        1. Takes NO PARAMETERS - all data must be defined within the function
        2. Defines all variables with appropriate bounds
        3. Implements all constraints
        4. Sets up the objective function
        5. Solves the model
        6. Returns a JSON-compatible dictionary with this exact structure:
        
        
        {{
          "status": "Optimal",  # or other PuLP status like "Infeasible", "Unbounded", etc.
          "objective_value": 15.0,  # the numeric value of the objective function
          "variables": {{
            "x1": 2.5,  # variable name and its optimal value
            "x2": 3.75,
            # ... all variables and their values
          }}
        }}
        """
        
        # If we have execution errors, augment the prompt with error information
        if state.get("execution_error"):
            error_info = state["execution_error"]
            prompt = f"""
            {base_prompt}
            
            The previous code had the following error:
            Error Type: {error_info["error_type"]}
            Error Message: {error_info["error_message"]}
            
            Here's the code that needs to be fixed:
            ```python
            {error_info["code"]}
            ```
            
            Please fix the issues and provide corrected code.
            """
        else:
            # Example function for the initial generation
            example = """
            Example of correct function definition:
            ```python
            def solve_problem():
                # Define the problem
                prob = LpProblem("LP Problem", LpMaximize)
                
                # Create variables
                x = LpVariable("x", lowBound=0)
                y = LpVariable("y", lowBound=0)
                
                # Add constraints
                prob += x + y <= 10
                
                # Set objective
                prob += 2*x + 3*y
                
                # Solve
                prob.solve()
                
                # Return solution as JSON-compatible dictionary
                return {
                    "status": LpStatus[prob.status],
                    "objective_value": value(prob.objective),
                    "variables": {
                        "x": value(x),
                        "y": value(y)
                    }
                }
            ```
            DO NOT define solve_problem() with any parameters. All data must be defined directly in the function.
            """
            prompt = f"{base_prompt}\n\n{example}\n\nOnly return code, no explanations."
        
        response = llm_client.send_prompt(prompt)
        solver_code = response.get("content", "")
        
        # Clean up code if it's wrapped in markdown
        if "```python" in solver_code:
            start = solver_code.find("```python") + 9
            end = solver_code.rfind("```")
            solver_code = solver_code[start:end].strip()
        elif "```" in solver_code:
            start = solver_code.find("```") + 3
            end = solver_code.rfind("```")
            solver_code = solver_code[start:end].strip()
        
        return {
            **state,
            "solver_code": solver_code,
            "error": None,
            "execution_error": None,  # Clear any previous execution errors
            "messages": state["messages"] + [
                AIMessage(content=f"Generated solver code:\n```python\n{solver_code}\n```")
            ],
            "code_generation_attempts": state["code_generation_attempts"] + 1
        }
    except Exception as e:
        return {
            **state,
            "error": f"Error generating solver code: {str(e)}",
            "messages": state["messages"] + [
                AIMessage(content=f"Error generating solver code: {str(e)}")
            ],
            "code_generation_attempts": state["code_generation_attempts"] + 1
        }

# Agent 4: Code Executor
def execute_solver_code(state: AgentState) -> AgentState:
    """Execute the generated solver code."""
    try:
        code = state["solver_code"]
        
        # Create a namespace for execution
        local_vars = {}
        
        # Import necessary libraries in the namespace
        exec("from pulp import *", local_vars)
        
        # Execute the generated code
        exec(code, local_vars)
        
        # Look for a function that returns the solution
        solution = None
        if "solve_problem" in local_vars and callable(local_vars["solve_problem"]):
            solution = local_vars["solve_problem"]()
        elif "main" in local_vars and callable(local_vars["main"]):
            solution = local_vars["main"]()
        elif "solution" in local_vars:
            solution = local_vars["solution"]
        else:
            # Extract variable values and objective from PuLP objects if present
            # This is a fallback if the code doesn't return a solution dict
            LpVariable = local_vars["LpVariable"]
            LpStatus = local_vars["LpStatus"]
            value = local_vars["value"]
            
            model_vars = {k: v for k, v in local_vars.items() 
                         if isinstance(v, LpVariable)}
            
            if model_vars and "prob" in local_vars:
                prob = local_vars["prob"]
                solution = {
                    "status": LpStatus[prob.status],
                    "objective_value": value(prob.objective),
                    "variables": {name: value(var) for name, var in model_vars.items()}
                }
        
        # Ensure numeric values are Python native types (not PuLP-specific types)
        if solution and isinstance(solution, dict):
            if "objective_value" in solution:
                solution["objective_value"] = float(solution["objective_value"])
            
            if "variables" in solution and isinstance(solution["variables"], dict):
                for var_name, var_value in solution["variables"].items():
                    if var_value is not None:
                        solution["variables"][var_name] = float(var_value)
        
        if not solution:
            raise ValueError("Could not extract solution from executed code")
            
        return {
            **state,
            "solution": solution,
            "error": None,
            "execution_error": None,  # Clear any previous execution errors
            "messages": state["messages"] + [
                AIMessage(content=f"Executed solver code successfully. Solution: {json.dumps(solution, indent=2)}")
            ]
        }
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        
        # Extract the most relevant error information
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": error_traceback,
            "code": state["solver_code"]
        }
        
        return {
            **state,
            "error": f"Error executing solver code: {str(e)}",
            "execution_error": error_details,  # Store detailed error info for the code generator
            "messages": state["messages"] + [
                AIMessage(content=f"Error executing solver code: {str(e)}\n\nTraceback:\n{error_traceback}")
            ]
        }

# Define the decision function for the model generation branch
def should_retry_model_generation(state: AgentState) -> Union[Literal["generate_model"], Literal["validate_model"]]:
    """Decide whether to retry model generation or proceed to validation."""
    if state["error"] and state["model_generation_attempts"] < 3:
        return "generate_model"
    return "validate_model"

# Define the decision function after validation
def after_validation(state: AgentState) -> Union[Literal["generate_model"], Literal["generate_code"], Literal["end"]]:
    """Decide what to do after validation."""
    if state["error"]:
        if state["model_generation_attempts"] < 5:
            return "generate_model"
        else:
            # We've tried 3 times and still have errors
            return "end"
    return "generate_code"

# Define the decision function after code generation
def after_code_generation(state: AgentState) -> Union[Literal["execute_code"], Literal["generate_code"], Literal["end"]]:
    """Decide what to do after code generation."""
    if state["error"]:
        if state["code_generation_attempts"] < 3:
            return "generate_code"
        else:
            return "end"
    return "execute_code"

# Define the decision function after code execution
def after_execution(state: AgentState) -> str:
    """Decide what to do after code execution."""
    if state["error"]:
        if state["code_generation_attempts"] < 3:
            return "generate_code"  # Go back to code generation with the error details
        else:
            return "end"  # Maximum attempts reached
    return "end"  # Execution was successful

# Build the graph
def build_agent_graph():
    #checkpoint = MemorySaver()
    workflow = StateGraph(AgentState)
    
    # Add nodes for each agent
    workflow.add_node("generate_model", generate_mathematical_model)
    workflow.add_node("validate_model", validate_mathematical_model)
    workflow.add_node("generate_code", generate_solver_code)
    workflow.add_node("execute_code", execute_solver_code)
    
    # Add edges to define the flow
    workflow.add_conditional_edges(
        "generate_model",
        should_retry_model_generation,
        {
            "generate_model": "generate_model",
            "validate_model": "validate_model"
        }
    )
    
    workflow.add_conditional_edges(
        "validate_model",
        after_validation,
        {
            "generate_model": "generate_model",
            "generate_code": "generate_code",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "generate_code",
        after_code_generation,
        {
            "execute_code": "execute_code",
            "generate_code": "generate_code",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "execute_code",
        after_execution,
        {
            "generate_code": "generate_code",
            "end": END
        }
    )
    
    # Set the entry point
    workflow.set_entry_point("generate_model")
    
    #return workflow.compile(checkpointer=checkpoint)
    return workflow.compile()

# Entry point function to run the agent workflow
def solve_lp_problem(problem_description: str) -> Dict[str, Any]:
    """
    Solve a linear programming problem described in natural language.
    
    Args:
        problem_description: Natural language description of the LP problem
        
    Returns:
        Dictionary containing the solution and other information
    """
    graph = build_agent_graph()
    
    # Initialize the state
    initial_state = AgentState(
        problem_description=problem_description,
        math_model=None,
        solver_code=None,
        solution=None,
        model_generation_attempts=0,
        code_generation_attempts=0,
        messages=[HumanMessage(content=problem_description)],
        error=None,
        execution_error=None
    )
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    return {
        "solution": result["solution"],
        "math_model": result["math_model"],
        "solver_code": result["solver_code"],
        "conversation": [m.content for m in result["messages"]]
    }