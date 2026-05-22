# Contributing to the FILS Framework

Thank you for your interest in contributing. The FILS Framework is an open-source pedagogy project. Contributions that improve clarity, expand coverage, or fix errors are welcome.

This guide explains the four main contribution types and the process for submitting each.

---

## Code of Conduct

This project follows a standard contributor code of conduct. In short: be direct, be constructive, and assume good faith. Criticism of ideas is welcome. Personal attacks are not. If you encounter behavior that violates these principles, open an issue with the label `conduct`.

---

## How to Add a New Concept Card

Concept cards use the BRIDGIA format (Bridge, Reality, Insight, Debug, Gesture, Integrate, Apply). A blank template is available at `templates/concept-card-template.md`.

**Steps:**

1. Copy `templates/concept-card-template.md` to the appropriate module folder under `concepts/`.
2. Name the file using lowercase and hyphens, e.g. `attention-mechanism.md`.
3. Fill in all seven BRIDGIA sections. Do not leave placeholder text in a submitted card.
4. Fill in the metadata block at the top: concept name, difficulty (1-5), impact (1-5), module, and which analogy skins are used.
5. Add a row for the concept to the module index file (`concepts/<module>/index.md`), including its direct dependencies.
6. Open a pull request with the label `new-concept`.

A concept card will not be merged if:
- Any BRIDGIA section is missing or uses placeholder text
- The metadata block is incomplete
- The concept does not have at least one tested analogy
- The minimal code example does not run in the current environment

---

## How to Add a New Analogy Skin

An analogy skin is an alternative framing for an existing concept. The core explanation and code remain unchanged. Only the analogy and its associated restaurant/domain context change.

This matters because the default analogies are restaurant-based. Learners from different backgrounds may connect better with culinary, musical, architectural, or sports-based framings.

**Steps:**

1. Locate the existing concept card in `concepts/<module>/<concept>.md`.
2. In the `skins/` section of that file (at the bottom), add a new subsection with the skin name.
3. Include: the alternative analogy, the analogy limit specific to this framing, and the domain context (e.g., "music production," "logistics," "architecture").
4. Do not remove or modify existing skins when adding new ones.
5. Open a pull request with the label `new-skin`.

Skins are reviewed for accuracy and clarity. A skin that introduces a new misconception will not be merged.

---

## How to Add a New Confusion Entry

The confusion library is at `docs/confusion-library.md`. Entries document the most common conceptual confusions beginners face.

**Format for each entry:**

```
## [Number]. [Concept A] vs [Concept B]

**Confusion:** [exact labels being confused]

**Why it's confusing:** [explanation of why the confusion arises]

**The key difference:** [clear, precise distinction]

**Restaurant analogy:** [analogy that clarifies the difference]

**Trap to avoid:** [specific mistake and how to prevent it]
```

**Steps:**

1. Check that your proposed confusion is not already covered.
2. Add your entry at the bottom of `docs/confusion-library.md` with the next sequential number.
3. Keep the analogy grounded in a concrete, familiar scenario. Abstract explanations belong in the "key difference" field, not the analogy.
4. Open a pull request with the label `confusion-entry`.

Entries that duplicate existing content, introduce inaccurate distinctions, or use an analogy that is itself confusing will be returned for revision.

---

## How to Submit a Notebook

Notebooks live in the `notebooks/` directory, organized by module. Each notebook corresponds to one module and contains all concepts in that module in sequence.

**Requirements for a new notebook:**

- Must correspond to an existing module with at least three validated concept cards
- Must use the standard cell structure: explanation, code comprehension, application, contribution (in that order, per concept)
- Must include the validation cell at the top of each locked section (see `notebooks/template.ipynb` for reference)
- Must run from top to bottom without errors in a clean environment using only the packages in `requirements.txt`
- Must include a module-level introduction cell and a summary cell at the end

**Steps:**

1. Copy `notebooks/template.ipynb` to `notebooks/<module-name>.ipynb`.
2. Build out the notebook following the structure in the template.
3. Run the full notebook in a clean virtual environment before submitting.
4. Open a pull request with the label `notebook`.

Notebooks that fail to run, deviate from the standard cell structure, or reference packages outside `requirements.txt` will not be merged until those issues are resolved.

---

## Pull Request Process

1. Fork the repository and create a branch named for your contribution type and concept, e.g. `concept/gradient-descent` or `confusion/loss-vs-cost`.
2. Make your changes on that branch.
3. Open a pull request against `main` with a clear title and description.
4. Apply the appropriate label from: `new-concept`, `new-skin`, `confusion-entry`, `notebook`, `bugfix`, `docs`.
5. A maintainer will review within 14 days. You may be asked to revise before merge.
6. Do not open multiple pull requests for the same contribution. Revise in the existing PR.

For questions before opening a PR, use GitHub Discussions.
