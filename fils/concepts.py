"""
Concept loader for the FILS Framework.

Reads BRIDGIA concept cards from the modules/ directory and presents them
in a readable format, optionally filtered by analogy skin.
"""

import os
import re
from pathlib import Path

# Resolve the project root (one level up from this file's package)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODULES_DIR = _PROJECT_ROOT / "modules"

# Module display names
_MODULE_NAMES = {
    "01-python-basics": "Python Basics",
    "02-data-pandas": "Data & Pandas",
    "03-machine-learning": "Machine Learning",
    "04-deep-learning": "Deep Learning",
    "05-nlp-llms": "NLP & LLMs",
    "06-statistics": "Statistics",
}


def list_modules():
    """List all available modules.

    Returns:
        list[dict]: Each dict has 'id', 'name', and 'path' keys.
    """
    modules = []
    if not _MODULES_DIR.exists():
        return modules
    for d in sorted(_MODULES_DIR.iterdir()):
        if d.is_dir() and (d / "README.md").exists():
            modules.append({
                "id": d.name,
                "name": _MODULE_NAMES.get(d.name, d.name),
                "path": str(d / "README.md"),
            })
    return modules


def _parse_cards(module_path):
    """Parse a module README.md into individual concept cards.

    Returns:
        list[dict]: Each dict has 'title', 'number', 'content', and 'sections'.
    """
    text = Path(module_path).read_text(encoding="utf-8")
    # Split on card headers (## Card N or ## Concept N or ### Card N)
    card_pattern = re.compile(
        r'^#{2,3}\s+(?:Card|Concept)\s+(\d+)\s*[:\-—]\s*(.+)$',
        re.MULTILINE
    )
    matches = list(card_pattern.finditer(text))
    cards = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        cards.append({
            "number": int(match.group(1)),
            "title": match.group(2).strip(),
            "content": content,
        })
    return cards


def list_all():
    """List all available concept cards across all modules.

    Returns:
        list[dict]: Each dict has 'number', 'title', and 'module' keys.
    """
    all_cards = []
    for module in list_modules():
        cards = _parse_cards(module["path"])
        for card in cards:
            all_cards.append({
                "number": card["number"],
                "title": card["title"],
                "module": module["name"],
            })
    return all_cards


def get_card(query):
    """Find a concept card by number or name fragment.

    Args:
        query: An integer (card number) or string (partial title match).

    Returns:
        dict with 'number', 'title', 'module', 'content' or None.
    """
    for module in list_modules():
        cards = _parse_cards(module["path"])
        for card in cards:
            if isinstance(query, int) and card["number"] == query:
                card["module"] = module["name"]
                return card
            if isinstance(query, str) and query.lower() in card["title"].lower():
                card["module"] = module["name"]
                return card
    return None


def show(query, skin=None):
    """Display a concept card in the terminal.

    Args:
        query: Card number (int) or partial title (str).
        skin: Optional analogy skin to highlight (e.g., "restaurant").
              If None, shows the full card.
    """
    card = get_card(query)
    if card is None:
        print(f"Concept not found: {query}")
        print("Use fils.list_all() to see available concepts.")
        return

    print(f"{'=' * 60}")
    print(f"  Card {card['number']}: {card['title']}")
    print(f"  Module: {card['module']}")
    print(f"{'=' * 60}")

    content = card["content"]

    if skin:
        # Highlight lines mentioning the requested skin
        lines = content.split("\n")
        skin_lower = skin.lower()
        filtered = []
        in_relevant_section = False
        for line in lines:
            if skin_lower in line.lower():
                in_relevant_section = True
                filtered.append(f">>> {line}")
            elif line.startswith("#"):
                in_relevant_section = False
                filtered.append(line)
            elif in_relevant_section:
                filtered.append(f"    {line}")
            else:
                filtered.append(line)
        print("\n".join(filtered))
    else:
        print(content)

    print(f"\n{'=' * 60}")
