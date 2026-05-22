"""
FILS Framework
===========================
An open-source framework for making AI, ML, Python and statistics
understandable through adaptive analogies, critical mirrors and mastery checks.

Quick start::

    from fils import concepts, agents

    # Show a concept card
    concepts.show("linear_regression")

    # Show with a specific analogy skin
    concepts.show("tokenization", skin="restaurant")

    # List all available concepts
    concepts.list_all()

    # Get the teacher agent prompt
    print(agents.teacher_prompt())

    # Get the student agent prompt
    print(agents.student_prompt())

    # Get the evaluator agent prompt
    print(agents.evaluator_prompt())
"""

__version__ = "0.1.0"
__author__ = "Fabrice Fils-Aimé"

from fils.concepts import show, list_all, get_card, list_modules
from fils.agents import teacher_prompt, student_prompt, evaluator_prompt
