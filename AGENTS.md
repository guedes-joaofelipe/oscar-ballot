# AGENTS.md

##  Project Overview
This is a repository for **Oscar Ballot Agent** — a collection of scripts to generate oscar ballot votes using one or multiple LLMs through API calls.

## ​ Setup & Build
- Install dependencies: `make setup`

##  Code Conventions
- ALWAYS include lint for the functions input arguments and output
- ALWAYS make sure you are obbeying to lint using: `make lint`
- AVOID using inline comments, unless extremely necessary
- AVOID using `format` to create strings. Instead, prioritize using the `f"{...}"` constructor
- Prioritize using list comprehension when using FOR loops
- ALWAYS use docstring for public functions using the numpy style. For private functions, include only a single line comment.

Example:
```python
# Good Example
def public_function(arg_1: str):
    """ Numpy docstring style
    
    Parameters
    ----------
    arg_1 : str
        argument description.
    """
    ...

# Good Example
def _private_function(arg_1: str):
    """Single line explanation"""
    ...
```
- NEVER edit the `data/winners.yaml` file

##  Testing Instructions
- Running unit tests: `make test`

##  Reminders for Agents
- NEVER copy sensitive information (specially from `configs/keys.yaml`)
- If unsure, ask for clarification or a human review.
- This file is agent-focused—keep project background concise; non-agent documentation belongs in README.md.