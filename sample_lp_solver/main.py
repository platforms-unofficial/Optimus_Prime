from pulp import LpMaximize, LpProblem, LpStatus, LpVariable


def solve_example_lp():
    """
    Solve a simple linear programming problem:
    Maximize: 3x + 5y
    Subject to:
        2x + 3y <= 12
        x + y <= 5
        x, y >= 0
    """
    # Create the model
    model = LpProblem(name="simple-lp-problem", sense=LpMaximize)
    
    # Define the variables
    x = LpVariable(name="x", lowBound=0)
    y = LpVariable(name="y", lowBound=0)
    
    # Add the constraints
    model += (2 * x + 3 * y <= 12, "constraint1")
    model += (x + y <= 5, "constraint2")
    
    # Set the objective
    model += 3 * x + 5 * y
    
    # Solve the problem
    model.solve()
    
    # Print the results
    print(f"Status: {model.status} ({LpStatus[model.status]})")
    print(f"Objective value: {model.objective.value()}")
    print(f"x = {x.value()}")
    print(f"y = {y.value()}")
    
    return {
        "status": LpStatus[model.status],
        "objective": model.objective.value(),
        "variables": {"x": x.value(), "y": y.value()}
    }


def main():
    print("Solving a linear programming example with PuLP:")
    result = solve_example_lp()
    print("\nSolution found!")


if __name__ == "__main__":
    main()
