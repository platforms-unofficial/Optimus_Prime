# LLM LP Solver

## Overview
The LLM LP Solver is a Python project designed to facilitate the creation and solving of linear programming (LP) problems using a natural language interface. By leveraging a large language model (LLM) API, users can describe their LP problems in plain English, and the system will parse these descriptions to generate mathematical models, which can then be solved using the PuLP library.

## Project Structure
```
llm_lp_solver
├── src
│   ├── agent_flow.py                   # LangGraph agent workflow implementation
│   ├── llm_client.py                   # Azure OpenAI API client
│   ├── models.py                       # Pydantic models for structured data
│   ├── main.py                         # Entry point with CLI interface
│   └── arbitrage_data_generator.py     # Data generator for example problem 
|   ├── inputs
|       └── sample_input.txt    # Sample LP problem descriptions
├── results                     # Generated solutions and code
│   ├── solver_[timestamp].py   # Generated PuLP solver code
│   └── lp_solver_result_[timestamp].json  # Complete results including model and solution
├── pyproject.toml              # Project metadata and dependencies
└── README.md
```

## Example: How to run
uv run src/main.py "$(cat src/inputs/arbitrage.txt)" src/data/labor_arbitrage_data.json