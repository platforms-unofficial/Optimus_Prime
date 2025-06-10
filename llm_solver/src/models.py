from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field

class Variable(BaseModel):
    """A variable in a linear programming problem."""
    name: str = Field(..., description="The name of the variable")
    description: str = Field(default="", description="Description of what the variable represents")
    lower_bound: Optional[float] = Field(default=0, description="Lower bound of the variable (default: 0)")
    upper_bound: Optional[float] = Field(default=None, description="Upper bound of the variable (None if unbounded)")

class Constraint(BaseModel):
    """A constraint in a linear programming problem."""
    name: str = Field(..., description="The name of the constraint")
    expression: str = Field(..., description="The mathematical expression of the constraint (e.g., '2*x + 3*y <= 10')")
    description: str = Field(default="", description="Description of what the constraint represents")

class Objective(BaseModel):
    """The objective function in a linear programming problem."""
    expression: str = Field(..., description="The mathematical expression to optimize (e.g., '5*x + 3*y')")
    direction: Literal["maximize", "minimize"] = Field(..., description="Whether to maximize or minimize the objective")
    description: str = Field(default="", description="Description of what the objective represents")

class MathematicalModel(BaseModel):
    """A mathematical model for a linear programming problem."""
    variables: List[Variable] = Field(..., description="The variables in the problem")
    constraints: List[Constraint] = Field(..., description="The constraints in the problem")
    objective: Objective = Field(..., description="The objective function to optimize")
    explanation: str = Field(description="Explanation of the mathematical model construction process")
class ValidationResult(BaseModel):
    valid: bool = Field(description="Whether the model is valid or not")
    missing_variables: List[str] = Field(default_factory=list, description="List of missing variables")
    missing_constraints: List[str] = Field(default_factory=list, description="Description of missing constraints")
    incorrect_objective: Optional[str] = Field(default=None, description="Description if objective is wrong")
    explanation_feedback: Optional[str] = Field(default=None, description="Feedback on the model explanation")
    explanation: str = Field(default="", description="Detailed explanation of issues")
