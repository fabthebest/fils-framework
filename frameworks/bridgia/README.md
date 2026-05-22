# BRIDGIA Framework

**A structured card system for teaching AI/ML concepts**
Part of [The FILS Framework](../../README.md)

---

## What is BRIDGIA?

BRIDGIA is a 7-case card system designed to deliver every AI/ML concept through a consistent, pedagogically grounded structure. The name is an acronym derived from the seven cases:

| Letter | French Name | English Name |
|--------|-------------|--------------|
| B | **B**ase / Accroche | Hook |
| R | Image mentale (**R**epresentation) | Mental Image |
| I | Décryptage (**I**nner truth) | Decryption |
| D | Co**d**e minimal | Minimal Code |
| G | Analyse & Intuition (**G**rasp) | Analysis & Intuition |
| I | P**i**èges & Limites | Traps & Limits |
| A | **A**pplication & Création | Application & Creation |

Each case serves a specific cognitive function. They are not decorative sections — they correspond to distinct phases of learning: attention capture, model building, precise understanding, procedural grounding, generalization, error awareness, and mastery verification.

BRIDGIA cards are the primary content unit of The FILS Framework. Every concept — whether a loss function, a regularization technique, or a neural architecture — gets exactly one BRIDGIA card.

---

## Why Each Case Exists

### Case 1 — Hook (Accroche)

Learning begins with a question the learner cannot immediately answer. The Hook is a single provocative question or a surprising fact that creates a state of productive tension. It should feel slightly uncomfortable — the learner should sense that they do not fully understand something they thought they did, or encounter a phenomenon that defies their current model of the world.

A good Hook is not a riddle and not a trivia question. It is an honest invitation into a problem that the concept genuinely resolves.

**Purpose:** Activate prior knowledge. Create a felt need for the concept.

---

### Case 2 — Mental Image (Image mentale)

Before any technical definition is introduced, the learner receives an analogy. This analogy is built using one of six pre-approved "skins" — constrained metaphorical domains that recur throughout the curriculum. Reusing skins across cards builds a cumulative mental vocabulary: once a learner knows how the Restaurant skin works, every new card that uses it gives them a head start.

The Mental Image is not an approximation of the concept. It is a structural isomorphism — the relationships between elements in the analogy must mirror the relationships between elements in the actual concept. Superficial resemblance is not enough.

**Purpose:** Give the learner a working model before the formal definition. Leverage narrative memory.

---

### Case 3 — Decryption (Décryptage)

After the analogy, the technical truth is stated plainly. This is the "nucleus" of the card — the definition stripped of metaphor, written in the language of the field. It should be precise enough to use in a technical conversation and dense enough that a reader who skips Cases 1 and 2 would feel the weight of it.

The Decryption never refers back to the analogy. It stands alone. It should be what a textbook would say, minus the textbook's habit of assuming motivation has already been established.

**Purpose:** Establish ground truth. Provide the learner with the terminology and formalism they need.

---

### Case 4 — Minimal Code (Code minimal)

A working Python example that is as short as possible while still being honest. "Honest" means the code actually demonstrates the concept — it does not hide complexity by importing a library that does all the work without explanation. It does not fabricate results.

The target is approximately 15 to 30 lines. The code must run. There should be at least one comment line that connects back to the concept, not just to Python syntax.

**Purpose:** Ground abstract understanding in executable reality. Force precision.

---

### Case 5 — Analysis & Intuition (Analyse & Intuition)

This case builds the learner's capacity for judgment. It answers the questions that do not appear in textbooks: When should I use this? What happens when I change this parameter? How does this interact with the things I already know?

Analysis & Intuition is structured as a series of short observations or conditionals rather than a narrative paragraph. The learner should finish this case able to make defensible choices — not just able to describe the concept.

**Purpose:** Develop engineering intuition. Bridge knowing to doing.

---

### Case 6 — Traps & Limits (Pièges & Limites)

Two things live in this case:

**Traps** are the most common mistakes practitioners make when first applying the concept. They are stated directly: "If you do X, the result is Y and it is wrong because Z."

**Mirror Mode** is a deliberate analysis of where the analogy from Case 2 breaks down. Every analogy has a failure boundary — a dimension along which the metaphor stops being a useful guide and starts being a misleading one. Mirror Mode names that boundary explicitly. This prevents the learner from over-extending the analogy into territory where it will hurt them.

Mirror Mode is one of the defining features of the BRIDGIA framework. Most pedagogy introduces analogies and never dismantles them. BRIDGIA treats the dismantling as mandatory.

**Purpose:** Prevent the most common errors. Teach the learner to distrust their own models at the right moments.

---

### Case 7 — Application & Creation (Application & Création)

A hands-on exercise that requires the learner to produce something. Not a comprehension question. Not a multiple-choice test. A task where failure is informative.

The exercise should be scoped to approximately 20 to 45 minutes of work for a learner who has read the full card. It should involve at least one decision point where the learner must apply judgment from Case 5, and at least one pitfall that Case 6 warned about.

**Purpose:** Verify mastery. Create a concrete artifact the learner can inspect and feel ownership of.

---

## How to Read a BRIDGIA Card

BRIDGIA cards are designed to be read in sequence from Case 1 to Case 7. The sequence is intentional:

- Cases 1 and 2 build motivation and a working model
- Cases 3 and 4 establish the technical foundation
- Cases 5 and 6 develop judgment and calibrated skepticism
- Case 7 activates what has been built

Readers who skip directly to Case 3 will find it accurate but cold. Readers who stop at Case 2 will have an analogy without grounding. The full value of the card is only accessible when all seven cases are engaged in order.

When using cards for review rather than first learning, Cases 3 and 4 can serve as a quick reference. Mirror Mode in Case 6 is particularly useful when a concept is behaving unexpectedly in practice — it often identifies the source of confusion.

---

## The Six Analogy Skins

A skin is a constrained metaphorical domain used consistently across multiple BRIDGIA cards. Using a small number of skins rather than an unlimited set of arbitrary analogies serves a specific purpose: cumulative familiarity. After a learner has encountered three or four cards that use the Restaurant skin, the skin itself becomes a cognitive tool they carry into every new card that uses it.

### Restaurant

The central actors are: a restaurant, a chef, customers, a menu, ingredients, orders, and the bill.

Useful for: supervised learning (customers order from a menu = the model receives labeled inputs), loss functions (the bill is wrong = the prediction is wrong), optimization (improving the recipe based on complaints), regularization (limiting the number of ingredients to prevent overcomplication).

### Construction

The central actors are: an architect, a blueprint, workers, materials, a foundation, floors, and inspections.

Useful for: model architecture (the blueprint is the network structure), training (workers build the structure according to the plan), layers in neural networks (floors), the distinction between design time and runtime.

### Orchestra

The central actors are: a conductor, musicians, instruments, a score, rehearsal, and a performance.

Useful for: ensemble methods (multiple musicians contributing to a single output), hyperparameter tuning (rehearsal), the difference between a single model and a coordinated system, attention mechanisms (the conductor directing focus).

### City

The central actors are: a city grid, roads, neighborhoods, traffic, infrastructure, residents, and planning authorities.

Useful for: graph-based concepts, routing algorithms, distributed systems, urban planning as an analogy for data infrastructure, traffic flow as an analogy for gradient flow.

### Video Game

The central actors are: a player, levels, rewards, penalties, a game state, rules, and a save file.

Useful for: reinforcement learning (the player navigating levels with rewards and penalties), exploration versus exploitation, state representation, policy learning, the Markov property.

### Human Body

The central actors are: organs, the nervous system, blood flow, cells, immune response, and metabolism.

Useful for: neural networks (nervous system), data pipelines (blood flow carrying nutrients), regularization (immune response), activation functions (cellular response to signals), the distinction between local and global regulation.

---

### Guidelines for Skin Selection

When creating a new card, select the skin whose structural relationships most closely mirror the structural relationships of the concept. Ask: which elements of this skin map cleanly onto the technical components of the concept? If more than one skin fits equally well, prefer the skin that has already been used for related concepts in the same module — this reinforces the cumulative vocabulary effect.

Avoid mixing skins within a single card. A concept explained with two different analogies simultaneously is usually explained with zero effective analogies.

If none of the six skins fits the concept adequately, document the mismatch in a comment in the card file and escalate to the curriculum maintainers for possible skin addition. Do not force a fit.

---

## How Mirror Mode Works

Mirror Mode is the section within Case 6 where the analogy from Case 2 is deliberately broken.

The structure of a Mirror Mode entry is:

1. State the dimension along which the analogy is valid
2. State the dimension along which the analogy fails
3. Explain what the reality is on that dimension, in technical terms

Mirror Mode is not a criticism of the analogy. It is a completion of it. A good analogy that includes its own failure conditions is more useful than a perfect analogy that does not exist.

Example of Mirror Mode reasoning (not a full entry):

> The Restaurant skin helps model the idea that a chef learns from complaints (gradient descent on loss). But it breaks down on scale: a real chef can only serve a few tables at once, while a model may train on millions of examples simultaneously. The analogy implies sequential feedback, but the technical reality is often batched, parallel computation. Do not carry the sequential assumption into any reasoning about training efficiency.

Contributors should write Mirror Mode entries with the same precision as technical documentation. Vague statements like "the analogy is not perfect" are not acceptable Mirror Mode entries.

---

## Complete Example Card: Linear Regression

**Concept:** Linear Regression
**Skin:** Restaurant
**Module:** Supervised Learning — Fundamentals

---

### Case 1 — Hook

You walk into a restaurant for the second time. The chef has never seen you before, but they know roughly what you ordered last time based on the neighborhood you live in, the day of the week, and whether you came alone or with company.

Here is the question: if the chef sees 500 customers and records what they order alongside four facts about each of them, can the chef build a system that predicts what the next customer will order — before the customer opens their mouth?

Linear regression is the answer to a version of this question. But it only works if you are willing to make a specific assumption about what "order" means. What is that assumption?

---

### Case 2 — Mental Image

Imagine the restaurant tracks one thing about each customer: how long they stayed. And it records one number about each visit: the total bill.

Over hundreds of visits, the chef notices a pattern: the longer a customer stays, the higher the bill. Not perfectly — sometimes a customer who stayed two hours only ordered a salad and wine. But in general, the trend is there.

The chef draws a line on a graph: time on the horizontal axis, bill on the vertical axis. The line does not pass through every point. It passes through the middle of the cloud of points — the position where, if you pick any value of time and look up at the line, you get a reasonable guess at what the bill would be.

That line is the model. Training the model means finding the line that is as close as possible to all the points at once. Prediction means reading off the line for a new value of time.

The slope of the line is a parameter. The position where it crosses the vertical axis is another parameter. Linear regression finds the values of these parameters that minimize the total error across all the points.

---

### Case 3 — Decryption

Linear regression is a supervised learning method that models the relationship between a continuous target variable and one or more input features by fitting a linear function.

Given a dataset of n observations, each with p features, linear regression estimates a weight vector w and a bias term b such that the predicted output for input x is:

```
y_hat = w_1 * x_1 + w_2 * x_2 + ... + w_p * x_p + b
```

The parameters are estimated by minimizing the Mean Squared Error (MSE) between predicted values and observed targets:

```
MSE = (1/n) * sum((y_i - y_hat_i)^2)
```

The ordinary least squares (OLS) solution is available in closed form when the number of features is small relative to the number of observations. For larger problems or when regularization is needed, gradient-based optimization is used.

Linear regression assumes:
- A linear relationship between features and target
- Independence of observations
- Homoscedasticity (constant variance of residuals)
- No severe multicollinearity among features

Violating these assumptions does not prevent the algorithm from running. It degrades the reliability of the results.

---

### Case 4 — Minimal Code

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Simulate the restaurant dataset: time_spent (minutes) vs total_bill (dollars)
np.random.seed(42)
time_spent = np.random.uniform(20, 180, size=200).reshape(-1, 1)
total_bill = 5 + 0.3 * time_spent.flatten() + np.random.normal(0, 8, size=200)

# Split into train and test sets
split = int(0.8 * len(time_spent))
X_train, X_test = time_spent[:split], time_spent[split:]
y_train, y_test = total_bill[:split], total_bill[split:]

# Fit the model — this finds the line that minimizes MSE on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Inspect the learned parameters
print(f"Slope (weight): {model.coef_[0]:.4f}")
print(f"Intercept (bias): {model.intercept_:.4f}")

# Evaluate on held-out data
y_pred = model.predict(X_test)
print(f"Test MSE: {mean_squared_error(y_test, y_pred):.2f}")
```

The slope should be close to 0.3 — the value used to generate the data. Noise prevents an exact recovery. This gap between the true parameter and the estimated parameter is the irreducible consequence of working with real data.

---

### Case 5 — Analysis & Intuition

**When to use linear regression:**
- The relationship between features and target is approximately linear, or can be made approximately linear through feature transformation
- Interpretability matters: the weight assigned to each feature is directly readable
- The target is continuous (not categorical, not count-based)
- A fast baseline is needed before trying more complex models

**What changes when parameters shift:**
- More training data generally reduces the variance of the estimated weights, not the bias
- Adding irrelevant features does not improve performance and may degrade it if the sample size is small relative to the number of features
- Scaling features does not affect predictions but affects the magnitude of the weights — unscaled features make weight comparison meaningless
- Outliers in the target variable have an outsized effect on the MSE objective and therefore on the fitted line

**How this interacts with other concepts:**
- Adding an L2 penalty to the MSE objective produces Ridge Regression, which shrinks weights toward zero and reduces overfitting
- Adding an L1 penalty produces Lasso Regression, which can drive weights to exactly zero, performing implicit feature selection
- Polynomial regression is linear regression applied to polynomial features — the model is still linear in the parameters

**A check that often catches errors:**
- Plot the residuals (y - y_hat) against the predicted values. If the residuals show a pattern — a curve, a funnel shape — the linear assumption is likely violated and the model's predictions cannot be trusted equally across the range of the target.

---

### Case 6 — Traps & Limits

**Trap 1 — Assuming linearity without checking**
Linear regression will always produce an output, even when the underlying relationship is nonlinear. The model will not raise an error. The test MSE will appear acceptable. Only a residual plot reveals the problem. Always plot residuals before trusting a linear regression result.

**Trap 2 — Confusing correlation with causation from the weights**
A large positive weight on a feature means that feature is correlated with the target in the training data. It does not mean the feature causes the target to increase. Weights from linear regression are descriptive in observational data, not causal.

**Trap 3 — Extrapolation**
The line extends infinitely in both directions, but the training data does not. A model trained on bills between 15 and 120 dollars should not be asked to predict a bill of 400 dollars. The linear relationship may not hold outside the observed range.

**Trap 4 — Multicollinearity**
If two features are highly correlated with each other, their individual weights become unstable — small changes in the data produce large swings in both weights. The predictions may still be accurate, but the individual weight values become uninterpretable. Variance Inflation Factor (VIF) can detect this.

---

**Mirror Mode: Where the Restaurant Skin Breaks Down**

The Restaurant analogy works well for the idea of "finding the line that fits the cloud of points" — the chef summarizing patterns across many customers. It is useful for understanding what the model is doing at a high level.

It breaks down on at least two dimensions:

**Dimension 1 — The chef can adapt the analogy.**
In the restaurant story, the chef can look at the graph and decide "this customer seems unusual, I will adjust my guess." The linear regression model cannot do this. The model applies the same learned function to every new input, regardless of how anomalous it appears. The analogy implies adaptive judgment; the model provides none.

**Dimension 2 — The analogy implies one feature.**
The restaurant story used a single input (time spent) to predict a single output (bill). Real linear regression typically uses many features simultaneously. The geometric intuition of a line in two dimensions becomes a hyperplane in high dimensions. The Restaurant analogy does not support reasoning about high-dimensional spaces — it will actively mislead a learner who tries to extend it there. When thinking about multicollinearity, feature importance, or regularization, leave the analogy behind and work directly with the mathematics.

---

### Case 7 — Application & Creation

**Exercise: Predicting apartment rental prices**

You are given a dataset of apartment listings. Each listing includes: square footage, number of bedrooms, distance from the city center in kilometers, age of the building in years, and monthly rental price.

Your task:

1. Load the dataset and perform an initial exploration. Identify which features appear most correlated with rental price using a correlation matrix or scatter plots.

2. Fit a linear regression model using all features. Report the MSE on a held-out test set (20% of the data).

3. Inspect the weights. Do the signs and magnitudes make intuitive sense? Would you expect distance from the city center to have a positive or negative weight? What does the actual weight tell you?

4. Plot the residuals against the predicted values. Does the pattern suggest the linear assumption is appropriate? If you see a funnel shape, what does that suggest about the data?

5. Try fitting the model with only the two features you identified as most correlated in step 1. Compare the test MSE to the full model. What does this tell you about the other features?

6. Introduce one intentionally bad feature: add a column that is pure random noise. Retrain and compare MSE. Did the model's performance change meaningfully? Why or why not?

**Deliverable:** A Jupyter notebook with code, plots, and a written paragraph for each of the six steps above. The written paragraphs should connect your observations back to the concepts in Cases 5 and 6 of this card.

**Success condition:** You can explain in plain language why the model's weights for the noise feature are not zero, and what regularization would do about it.

---

## Creating a New BRIDGIA Card

### Before You Start

Confirm the concept does not already have a card. Check the card index at `frameworks/bridgia/cards/index.md`.

Confirm the concept is scoped correctly. A BRIDGIA card should cover one learnable unit — something a student can internalize in a focused session. If the concept requires splitting across multiple cards, plan the split before writing any single card.

### File Structure

Create a new file at:

```
frameworks/bridgia/cards/<module-slug>/<concept-slug>.md
```

For example:
```
frameworks/bridgia/cards/supervised-learning/linear-regression.md
```

The file should begin with a metadata block:

```
---
concept: Linear Regression
module: Supervised Learning
skin: Restaurant
difficulty: Foundational
prerequisites: [basic-statistics, numpy-fundamentals]
version: 1.0
---
```

### Writing Each Case

Follow the specifications in the "Why Each Case Exists" section above. Additional guidelines:

- The Hook should be one to four sentences. It should not answer itself.
- The Mental Image should not use the word "like" as a shortcut for structural explanation. Show the isomorphism, do not just assert it.
- The Decryption should include the mathematical form of the concept where one exists. Do not defer all formalism to external references.
- The Minimal Code must be tested and verified to run before submission. Include the Python version and library versions in a comment if the code depends on specific API behavior.
- Analysis & Intuition should include at least one "when not to use this" observation.
- Mirror Mode must identify at least two dimensions where the analogy fails. One is usually sufficient to find; the second requires careful thought and is often more valuable.
- The Application exercise must have a stated deliverable and a stated success condition.

### Review Checklist

Before submitting a new card for review:

- [ ] All seven cases are present and non-empty
- [ ] The analogy skin is one of the six approved skins
- [ ] The Mental Image and the Decryption are consistent with each other
- [ ] The Minimal Code runs without modification
- [ ] Mirror Mode has at least two entries
- [ ] The Application exercise has both a deliverable and a success condition
- [ ] The card has been read aloud (or reviewed by a second person) for clarity

---

## Versioning and Maintenance

BRIDGIA cards are versioned. When a card is updated, the version field in the metadata block is incremented and the change is noted in `frameworks/bridgia/CHANGELOG.md`.

Cards should be reviewed when:
- The underlying library API changes in a way that affects the Minimal Code
- A better analogy skin is identified
- Learner testing reveals systematic misunderstanding traceable to a specific case
- A new concept is added that creates a dependency requiring the existing card to be revised

Cards are never deleted. Deprecated cards are marked with a `deprecated: true` flag in the metadata and moved to `frameworks/bridgia/cards/deprecated/`. This preserves the history of pedagogical decisions.

---

## Relationship to Other FILS Framework Components

BRIDGIA cards are the atomic unit of the curriculum. They are designed to be embedded into larger learning sequences but are also self-contained. A reader who encounters a single BRIDGIA card without context should be able to extract full value from it.

The broader FILS Framework includes sequencing guidelines (which cards to read in which order for specific learning goals), assessment rubrics (how to evaluate the Application exercises), and instructor notes (additional context for facilitators who are guiding learners through the material). These components are documented separately but reference BRIDGIA cards by their concept slug.

---

*BRIDGIA Framework — Part of The FILS Framework*
*Open source under the terms specified in the root LICENSE file*
