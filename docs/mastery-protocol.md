# Mastery Validation Protocol

This document defines the formal mastery validation protocol used in the FILS Framework. It governs how understanding is measured, how advancement is gated, and how notebooks enforce the progression.

---

## The 6-Point Scoring System

Every concept in the FILS Framework is evaluated on a uniform 6-point scale. The points are earned independently and represent six distinct dimensions of understanding.

| Point | Dimension           | What It Measures                                                  |
|-------|---------------------|-------------------------------------------------------------------|
| 1     | Definition Recall   | Can restate the concept simply in their own words                |
| 2     | Code Comprehension  | Understands the minimal working code example                      |
| 3     | Analogy             | Grasps the main analogy and can apply or extend it               |
| 4     | Analogy Limit       | Knows where the analogy breaks down (Mirror Mode)                |
| 5     | Error Avoidance     | Identifies the most common beginner mistake                      |
| 6     | Mini-Application    | Applies the concept to a new small problem                        |

No partial credit is awarded per point. Each point is either earned or not. The Evaluator agent issues one sentence of justification per point in its output.

The six dimensions were chosen because understanding a concept means more than recalling it. A learner who can define tokenization but cannot spot a tokenization bug, or who relies entirely on an analogy without knowing its limits, has a fragile understanding that will break under real conditions.

---

## The 50% Gate

A score of 3 out of 6 is the minimum required to advance.

This threshold was chosen deliberately. A learner who has earned three points has demonstrated:
- They know what the concept is (Point 1)
- They have some practical or analogical grasp (Points 2 or 3)
- They have at least one additional dimension of understanding beyond the surface

A learner with only 2 points has a definition and possibly a surface-level analogy, but no demonstrated ability to apply, extend, or critically evaluate the concept. Advancing them creates compounding confusion — each new concept will rest on a foundation that has not solidified.

### Conditional Advancement at Score 3

A score of 3 does not automatically permit advancement. The Evaluator applies one additional rule:

If the next concept in the sequence depends directly on the current concept, the learner must reach at least 4 before advancing. The Evaluator checks the concept dependency graph (defined in each module's index) before issuing a decision at score 3.

A learner at score 3 on an isolated concept — one with no direct dependents in the current sequence — may advance with a provisional pass flag attached to the session record.

---

## Gesture-Based Scoring

Gesture-based scoring is an active mechanism for evaluating Points 5 and 6. It cannot be earned passively.

### The Mechanism

The Evaluator invokes the Struggling Student persona. The Struggling Student presents plausible but flawed logic — a miswritten line of code, an incorrect chain of reasoning, or a misapplied concept — as if a real peer had written it. The learner is asked to review it.

The learner is not told the content is flawed. They must make that determination themselves.

### Scoring Rules

- Correctly identifying the error and explaining it: earns Point 5
- Correctly identifying the error, explaining it, and proposing a valid correction: earns Points 5 and 6
- Missing the error entirely: neither point is awarded; the specific misconception is logged for the Teacher
- Incorrectly flagging correct content as flawed: no points lost, but no points gained

### Design Rationale

Passive assessment — asking "what is the common mistake?" — rewards learners who memorized the mistake from the lesson without truly understanding it. Gesture-based scoring requires the learner to detect an error in an unfamiliar framing, which tests genuine comprehension rather than recall.

---

## Three Mastery Tiers

### Tier 1 — Continue (score 3/6)

The learner meets the minimum threshold. They are cleared to proceed, with conditional rules applying as described in the 50% Gate section. This tier does not unlock optional exercises or contribution privileges.

### Tier 2 — Validated (score 4/6)

The learner has demonstrated beginner-level mastery. This is the standard completion threshold. The concept is marked validated in the session record. The learner can proceed without conditions, and the next concept in the module is unlocked.

### Tier 3 — Contribute (score 5/6 or 6/6)

The learner has demonstrated strong or excellent mastery. In addition to all Validated privileges, they are eligible to:
- Write an alternative analogy skin for the concept and submit it as a pull request
- Add a confusion entry to the confusion library
- Review another learner's gesture-based response if the platform supports peer review

A score of 6/6 additionally generates a mastery badge that can be displayed in the learner's session summary.

---

## Notebook Integration

Each Jupyter notebook in the FILS Framework is divided into sections. Each section corresponds to one concept. Within a section, cells are organized as follows:

1. Explanation cells (always visible, always runnable)
2. Code comprehension cells (unlocked after Point 2 is earned)
3. Application cells (locked until the concept reaches Validated tier)
4. Contribution cells (locked until the concept reaches Contribute tier)

Cell locking is implemented via notebook metadata tags and a lightweight validation cell at the top of each locked section. The validation cell reads the session record and raises an informative message if the prerequisite score has not been reached.

When a learner advances to a new tier mid-session, they can run the validation cell again to unlock the next section without restarting the notebook.

---

## Example Walkthrough: Tokenization

This walkthrough illustrates the full protocol applied to a single concept.

**Concept:** Tokenization
**Module:** Language Models — Fundamentals
**Dependency:** None (entry-level concept)

---

**Step 1 — Teacher presents the concept**

The Teacher agent explains tokenization using the Kitchen Ticket analogy: the tokenizer is the ticket system that breaks an order (a sentence) into line items (tokens) that the kitchen (the model) processes one by one. The minimal code example uses `transformers.AutoTokenizer`.

---

**Step 2 — Learner responds**

The learner says: "So a token is basically a word, split up if it is long. Like the word tokenization becomes two tokens. And the model never sees the actual word, it only sees the token IDs."

---

**Step 3 — Evaluator scores**

- Point 1 (Definition): Earned. The learner restated the concept correctly.
- Point 2 (Code): Not yet earned. No code engagement demonstrated.
- Point 3 (Analogy): Partially. The learner described the output of the analogy without demonstrating they understood the mapping. Not earned.
- Point 4 (Analogy Limit): Not earned.
- Point 5 (Error Avoidance): Not earned.
- Point 6 (Mini-Application): Not earned.

Score so far: 1/6. The Evaluator does not issue a decision yet — it prompts the Teacher to run a code comprehension check and the gesture-based exercise.

---

**Step 4 — Code check**

The Teacher shows a cell encoding the word "unhappiness" and asks the learner to predict the token count. The learner says "probably 3 — un, happi, ness." The tokenizer returns ["un", "happiness"] = 2 tokens. The learner explains the discrepancy correctly.

- Point 2 (Code): Earned.

---

**Step 5 — Gesture-based exercise**

The Evaluator invokes the Struggling Student: "A classmate wrote this and says it counts the words in a sentence correctly."

```python
text = "The model processes language."
tokens = text.split(" ")
print(f"Token count: {len(tokens)}")
```

The learner identifies that `split(" ")` counts words by spaces, not tokens, and that punctuation and subword splits are not handled. They note that "language." would be one split item but would likely be two tokens in a real tokenizer.

- Point 5 (Error Avoidance): Earned.
- Point 6 (Mini-Application): Earned (proposed a correction using the actual tokenizer).

---

**Step 6 — Analogy limit check**

The Teacher asks where the kitchen ticket analogy breaks down. The learner notes that kitchen tickets are readable by humans, while token IDs are just integers — the analogy does not capture that the model never sees the original text in any recognizable form.

- Point 3 (Analogy): Earned.
- Point 4 (Analogy Limit): Earned.

---

**Step 7 — Final score**

Total: 6/6.
Decision: Excellent Mastery.
Tier: Contribute. The learner is invited to submit an alternative analogy skin.
The next concept — Attention Mechanism — is unlocked in the notebook.
