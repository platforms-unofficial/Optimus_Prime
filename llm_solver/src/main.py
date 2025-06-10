import os
import sys
import json
import datetime
from dotenv import load_dotenv
from agent_flow import solve_lp_problem

# Load environment variables from .env file
load_dotenv()

def main():
    """Main entry point for the LP solver."""
    
    # Check if API key is set
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("Error: AZURE_OPENAI_API_KEY environment variable is not set.")
        print("Please set it in a .env file or directly in your environment.")
        sys.exit(1)
    
    # Get problem description from command line arguments or prompt user
    if len(sys.argv) > 1:
        problem_description = " ".join(sys.argv[1:])
    else:
        print("Please describe the linear programming problem you want to solve:")
        problem_description = input("> ")
    
    print(f"\nSolving problem: {problem_description}\n")
    
    # Run the agent workflow
    result = solve_lp_problem(problem_description)
    
    # Display the mathematical model
    print("\n" + "=" * 50)
    print("MATHEMATICAL MODEL:")
    print("=" * 50)
    if result["math_model"]:
        model = result["math_model"]
        print("\nVariables:")
        for var in model.variables:
            bounds = []
            if var.lower_bound is not None:
                bounds.append(f">= {var.lower_bound}")
            if var.upper_bound is not None:
                bounds.append(f"<= {var.upper_bound}")
            
            bounds_str = ", ".join(bounds)
            description = var.description or ""
            print(f"  {var.name} {bounds_str} - {description}")
        
        print("\nConstraints:")
        for constraint in model.constraints:
            print(f"  {constraint.name}: {constraint.expression}")
            if constraint.description:
                print(f"    {constraint.description}")
        
        print(f"\nObjective: {model.objective.direction} {model.objective.expression}")
        if model.objective.description:
            print(f"  {model.objective.description}")
    else:
        print("No mathematical model was generated.")
    
    # Display the solution
    print("\n" + "=" * 50)
    print("SOLUTION:")
    print("=" * 50)
    if result["solution"]:
        print(f"Status: {result['solution'].get('status', 'Unknown')}")
        print(f"Objective value: {result['solution'].get('objective_value', 'Unknown')}")
        print("\nVariable values:")
        for var, value in result["solution"].get("variables", {}).items():
            print(f"  {var} = {value}")
    else:
        print("No solution was found.")
    
    # Display the conversation history
    print("\n" + "=" * 50)
    print("AGENT CONVERSATION:")
    print("=" * 50)
    if "conversation" in result and result["conversation"]:
        for i, message in enumerate(result["conversation"]):
            # Limit message length for display
            if len(message) > 500:
                message = message[:500] + "... (truncated)"
            print(f"\nMessage {i+1}:")
            print(f"{message}")
    else:
        print("No conversation history available.")
    
    # Generate timestamp for filenames
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Optionally save the generated code to a file
    if result["solver_code"]:
        solver_filename = f"results/solver_{timestamp}.py"
        with open(solver_filename, "w") as f:
            f.write(result["solver_code"])
        print(f"\nSolver code has been saved to {solver_filename}")
    
    # Save the full result (need to convert Pydantic models to dict)
    result_dict = result.copy()
    if result_dict["math_model"]:
        result_dict["math_model"] = result_dict["math_model"].model_dump()
    
    results_filename = f"results/lp_solver_result_{timestamp}.json"
    with open(results_filename, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"\nFull result has been saved to {results_filename}")

if __name__ == "__main__":
    main()