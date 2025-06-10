# Linear Programming Solver

A Python package for solving linear programming problems using open source solvers.

## Features

- Uses PuLP to model and solve linear programming problems
- Compatible with multiple solvers (CBC, GLPK, and commercial solvers like Gurobi)
- Simple API for defining and solving LP problems

## Installation

```bash
uv sync
```

## Usage

```python
from pulp import LpMaximize, LpProblem, LpVariable

# Create the model
model = LpProblem(name="simple-problem", sense=LpMaximize)

# Define variables
x = LpVariable(name="x", lowBound=0)
y = LpVariable(name="y", lowBound=0)

# Add constraints
model += (2 * x + 3 * y <= 12, "constraint1")
model += (x + y <= 5, "constraint2")

# Set objective
model += 3 * x + 5 * y

# Solve
model.solve()

# Access results
print(f"x = {x.value()}, y = {y.value()}")
print(f"Objective value: {model.objective.value()}")
```

## Switching to Gurobi

To switch to Gurobi later, you only need to install Gurobi and modify the solver:

```python
# Install gurobi: pip install gurobipy
model.solve(solver=pulp.GUROBI())
```