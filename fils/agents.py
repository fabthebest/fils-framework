"""
Agent prompt loader for the FILS Framework.

Provides ready-to-use system prompts for the three agents:
- Professor Bridge (Master Teacher)
- Fabrice the Skeptic (Struggling Learner)
- The Evaluator (Neutral Scorer)

Copy any of these prompts into an LLM chat to activate the agent.
"""

from pathlib import Path

_AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"


def _load_prompt(filename):
    """Load an agent prompt file."""
    path = _AGENTS_DIR / filename
    if not path.exists():
        return f"Agent prompt not found: {path}\nMake sure the agents/ directory is present."
    return path.read_text(encoding="utf-8")


def teacher_prompt():
    """Return the Professor Bridge (Master Teacher) system prompt.

    Usage:
        Copy the output into any LLM (ChatGPT, Claude, etc.) as a system prompt
        to activate the Socratic teaching agent.

    Returns:
        str: The complete system prompt.
    """
    return _load_prompt("teacher.md")


def student_prompt():
    """Return the Fabrice the Skeptic (Struggling Learner) system prompt.

    Usage:
        Load this into a second LLM instance. Feed it the teacher's explanations
        and it will respond with realistic beginner confusions, forcing the
        teacher (or you) to clarify.

    Returns:
        str: The complete system prompt.
    """
    return _load_prompt("student.md")


def evaluator_prompt():
    """Return The Evaluator (Neutral Scorer) system prompt.

    Usage:
        Load this into a third LLM instance. Feed it the student's responses
        and it will score comprehension on a 0-6 scale with pass/fail decisions.

    Returns:
        str: The complete system prompt.
    """
    return _load_prompt("evaluator.md")


def list_agents():
    """List available agent prompts.

    Returns:
        list[dict]: Each dict has 'name', 'file', and 'description'.
    """
    return [
        {
            "name": "Professor Bridge",
            "file": "teacher.md",
            "role": "Master Teacher — Socratic, adaptive, multi-analogy explainer",
        },
        {
            "name": "Fabrice the Skeptic",
            "file": "student.md",
            "role": "Struggling Learner — stress-tests explanations with real beginner confusions",
        },
        {
            "name": "The Evaluator",
            "file": "evaluator.md",
            "role": "Neutral Scorer — 6-point mastery assessment with pass/fail gate",
        },
    ]
