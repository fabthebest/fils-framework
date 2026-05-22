# Module 03 — Machine Learning

**The FILS Framework**
Part of the open-source AI/ML curriculum for complete beginners.

---

## Module Overview

This module introduces the core ideas of machine learning: how a system learns patterns from data, makes predictions, and fails in recognizable ways. Every concept is delivered as a BRIDGIA card — seven cases per concept, from Hook to Application.

**Concepts in this module:**

| Card | Concept | Difficulty | Relevance |
|------|---------|------------|-----------|
| 14 | Train/Test Split | 2/5 | 5/5 |
| 15 | Linear Regression | 3/5 | 5/5 |
| 16 | Logistic Regression | 3/5 | 5/5 |
| 17 | Decision Trees | 2/5 | 5/5 |
| 18 | Random Forest | 3/5 | 5/5 |
| 19 | k-Nearest Neighbors | 2/5 | 4/5 |
| 20 | Support Vector Machines | 4/5 | 3/5 |
| 21 | Overfitting and Regularization | 3/5 | 5/5 |
| 22 | Cross-Validation | 3/5 | 5/5 |
| 23 | Feature Engineering | 3/5 | 5/5 |

---

---

## Card 14 — Train/Test Split

**Difficulty:** 2/5 | **Relevance:** 5/5 | **Skin:** Restaurant

---

### Case 1 — Hook

You spend a full afternoon seasoning a pot of soup. You taste it as you go — adjusting salt, adding herbs, tasting again. By the end, you are confident: it is perfect.

You serve it to guests. Three of them say it is too salty.

What went wrong? You used the same batch of soup to both season and judge. The question this card answers: why is the practice of judging a model on the data it trained on the exact same mistake, and what is the minimal change that fixes it?

---

### Case 2 — Mental Image

**Restaurant skin:** Before a restaurant opens to the public, the chef prepares a full batch of soup. Half of it goes to kitchen staff for seasoning and refinement — this is the training set. The other half is reserved and covered; no one touches it. Once the recipe is finalized, the covered pot is brought out and tasted by a fresh tester who had no role in the seasoning. The tester's verdict is the only score that counts.

If the chef tasted from the covered pot during seasoning — even once — the evaluation is contaminated. The tester is no longer measuring a recipe's general quality. They are measuring how well it was tuned to that exact pot.

**Construction skin:** A contractor builds a section of wall, then measures how straight it is using the same plank they used to align it during construction. Of course it looks straight — the plank defined the standard. To get an honest evaluation, you bring in an independent inspector with their own measuring tools who was not present during the build.

---

### Case 3 — Decryption

A train/test split is a data partitioning procedure that divides a labeled dataset into two non-overlapping subsets: a training set and a test set.

The training set is used to fit the model — to estimate its parameters. The test set is held out and used only once, after training is complete, to evaluate the model's performance on unseen data. The test set simulates the distribution of new, real-world inputs the model will encounter in deployment.

The purpose is to obtain an unbiased estimate of generalization error — the expected error on new data. If the same data is used for both training and evaluation, the performance metric measures memorization, not generalization.

Standard practice splits the data into 80% training and 20% test, though 70/30 and 90/10 splits are also used depending on dataset size. The split must be random (with a fixed seed for reproducibility) to avoid distributional mismatch between the two subsets.

In temporal data (time series, event logs), the split must respect time ordering: training data must precede test data chronologically. Random splitting on temporal data constitutes data leakage.

---

### Case 4 — Minimal Code

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset: 150 samples, 4 features, 3 classes
X, y = load_iris(return_X_y=True)

# Split: 80% train, 20% test — random_state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a simple model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Evaluate on training data (this score is optimistic)
train_acc = accuracy_score(y_train, model.predict(X_train))

# Evaluate on held-out test data (this score is honest)
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"Training accuracy:  {train_acc:.4f}")
print(f"Test accuracy:      {test_acc:.4f}")
print(f"Training set size:  {len(X_train)}")
print(f"Test set size:      {len(X_test)}")
```

**Expected output:**
```
Training accuracy:  0.9583
Test accuracy:      1.0000
Training set size:  120
Test set size:      30
```

Note: with k=3 on the Iris dataset, the test accuracy happens to be 1.0 on this seed. On a noisier dataset, training accuracy exceeds test accuracy — the gap reveals overfitting.

---

### Case 6 — Analysis and Intuition

- If your test accuracy is significantly lower than your training accuracy, the model has memorized the training data rather than learned general patterns.
- If your test accuracy is higher than your training accuracy (as in the example above), the training set happened to contain the harder cases. This is a size artifact — with 30 test samples, the score is noisy.
- Small datasets produce unstable test scores. A 5% difference in test accuracy on 50 samples may reflect randomness in the split rather than a real difference between models. This is why cross-validation exists (Card 22).
- Never use the test set to select hyperparameters. If you try multiple models and pick the one with the best test score, the test set is no longer independent. You need a third partition: validation set.
- The random state matters for comparing models. Two models compared with different random seeds are evaluated on different test sets. Always fix the seed.

---

### Case 6 — Traps and Limits

**Trap 1 — Evaluating on training data and calling it "accuracy"**
The most common beginner mistake. `model.predict(X_train)` does not tell you how the model performs on new data. The number it returns is always optimistic and sometimes dramatically so.

**Trap 2 — Splitting before preprocessing**
If you normalize features or fill missing values using statistics from the full dataset before splitting, the test set has leaked information into the training process. Always split first, then fit your preprocessing pipeline on training data only, then apply it to the test set.

**Trap 3 — Splitting temporal data randomly**
A model trained on data from 2023 and 2024 that is evaluated on randomly selected 2022 data will appear to generalize. It is predicting the past from the future. For time series, the test set must be the most recent observations.

**Mirror Mode: Where the Restaurant Skin Breaks Down**

The covered-pot analogy maps well onto the concept of data isolation: the held-out portion is not touched during development, and the final evaluation is done by a fresh judge.

It breaks down on two dimensions:

**Dimension 1 — The chef can always make more soup.**
In the restaurant story, the chef can prepare a new batch to test a new variation. In machine learning, the dataset is fixed. You cannot generate more labeled data on demand. This means the choice of how much data to allocate to training versus testing involves a real tradeoff — more test data means more reliable evaluation, but less training data means a weaker model. The analogy implies unlimited raw material; the technical reality is a constrained budget.

**Dimension 2 — The taster's verdict is one number.**
A single test on a single pot gives one judgment: good or bad. In machine learning, the test set score is an aggregate over many individual predictions. Two models can have identical test accuracy while failing on completely different subsets of the data. The analogy conceals the distributional complexity of evaluation. A single accuracy number can hide a model that performs well on the majority class and catastrophically on a minority class.

---

### Case 7 — Application Exercise

**Exercise: Isolating the evaluation**

Take the Iris dataset (or any dataset you have used before). Train a K-Nearest Neighbors classifier with k=1.

1. Evaluate it on the training data. Record the accuracy.
2. Evaluate it on a held-out 20% test set. Record the accuracy.
3. Explain the gap in one paragraph. Why does k=1 produce near-perfect training accuracy and lower test accuracy?
4. Repeat with k=15. How does the gap change? What does this tell you about the relationship between model complexity and the train/test gap?

**Deliverable:** A short notebook with both evaluations, the two accuracy numbers for each value of k, and a written explanation connecting your observations to the concept of generalization.

**Success condition:** You can articulate why a model that scores 100% on training data is not necessarily a good model, and explain what specific property of k=1 KNN causes this behavior.

---

---

## Card 15 — Linear Regression

**Difficulty:** 3/5 | **Relevance:** 5/5 | **Skin:** Restaurant / Construction

---

### Case 1 — Hook

A restaurant owner notices that on hot days, more customers come in for cold drinks. On cold days, fewer. She starts wondering: if she knew tomorrow's temperature at 8am, could she predict how many customers to expect by noon — precisely enough to decide how many staff to schedule?

She has two years of daily records: temperature and customer count. Is that enough to build a prediction system? And if so, what assumptions must be true for that system to be reliable?

---

### Case 2 — Mental Image

**Restaurant skin:** Plot every recorded day as a dot on a chart — temperature on the horizontal axis, customer count on the vertical axis. You see a loose cloud of dots slanting upward to the right: warmer days tend to bring more customers. Not perfectly — some cold days had inexplicable crowds, some hot days were slow. But the trend is there.

Linear regression draws the single straight line that minimizes the total vertical distance from every dot to the line. It is the line of best fit through the cloud. Once you have that line, predicting for a new temperature means reading up from the temperature axis to the line and reading across to the customer axis.

**Construction skin:** An architect estimating material costs tracks the floor area of past projects and the total amount of concrete used. Larger floors require more concrete — not perfectly, because some floors have unusual shapes — but the trend is linear enough. The architect draws a line through the data points and uses it to estimate concrete for the next project based on its planned area. The line is the model. Its slope is the rate at which concrete needs increase per square meter.

---

### Case 3 — Decryption

Linear regression models the relationship between a continuous target variable y and one or more input features X by fitting a linear function:

```
y_hat = w_1 * x_1 + w_2 * x_2 + ... + w_p * x_p + b
```

where w is the weight vector and b is the bias (intercept). Parameters are estimated by minimizing the Mean Squared Error over the training set:

```
MSE = (1/n) * sum((y_i - y_hat_i)^2)
```

For a single feature, the ordinary least squares (OLS) solution has a closed form. For multiple features, the solution is w = (X^T X)^{-1} X^T y, computed directly or via gradient descent.

Linear regression assumes a linear relationship between features and target, independence of observations, homoscedasticity (equal variance of residuals across the range of predictions), and no severe multicollinearity among features. Violating these assumptions does not prevent the algorithm from running — it silently degrades prediction reliability.

---

### Case 4 — Minimal Code

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Simulate daily restaurant data: temperature (Celsius) vs customer count
np.random.seed(0)
temperature = np.random.uniform(5, 35, size=300).reshape(-1, 1)
customers = 20 + 3.5 * temperature.flatten() + np.random.normal(0, 10, size=300)

X_train, X_test, y_train, y_test = train_test_split(
    temperature, customers, test_size=0.2, random_state=0
)

# Fit: finds the slope and intercept that minimize MSE on training data
model = LinearRegression()
model.fit(X_train, y_train)

print(f"Slope (customers per degree C): {model.coef_[0]:.4f}")
print(f"Intercept:                      {model.intercept_:.4f}")
print(f"Test MSE:                       {mean_squared_error(y_test, model.predict(X_test)):.2f}")

# Predict for a new day: 28 degrees Celsius
prediction = model.predict([[28]])
print(f"Predicted customers at 28C:     {prediction[0]:.1f}")
```

**Expected output:**
```
Slope (customers per degree C): 3.5341
Intercept:                      19.4872
Test MSE:                       96.74
Predicted customers at 28C:     118.4
```

The slope should be close to 3.5 — the true value used to generate the data. The residual noise prevents exact recovery.

---

### Case 6 — Analysis and Intuition

- A large slope means a strong linear relationship between the feature and the target. A slope near zero means the feature carries little predictive information.
- Adding irrelevant features increases the risk of fitting noise, especially when the dataset is small relative to the number of features.
- Scaling features does not change predictions but does change the magnitude of weights — comparing weights across features with different units is meaningless without scaling first.
- Plot residuals against predicted values before trusting any linear regression result. A curved pattern in the residuals means the linear assumption is violated.
- Linear regression provides a fast, interpretable baseline. If a more complex model does not clearly outperform it, prefer the simpler model.

---

### Case 6 — Traps and Limits

**Trap 1 — Assuming linearity without checking**
The model always produces output. A curved relationship will be fit with a line, and the result will look acceptable until you plot the residuals. Always check.

**Trap 2 — Extrapolating outside the training range**
A line extends infinitely; training data does not. Predictions for inputs far outside the observed range have no empirical support. The relationship may not remain linear there.

**Trap 3 — Confusing weight magnitude with importance**
A large weight does not mean an important feature if that feature is measured in small units (centimeters versus kilometers). Always scale before interpreting weights.

**Mirror Mode: Where the Analogy Breaks Down**

Both skins — the restaurant cloud of dots and the architect's material estimate — work well for the single-feature case. They break down when the feature space grows:

**Dimension 1 — The line becomes a hyperplane.**
When there are 20 input features, the model is fitting a 20-dimensional hyperplane, not a line. The restaurant chart (two axes, one line) provides no geometric intuition for this. Learners who carry the two-dimensional image into high-dimensional work will struggle to reason about multicollinearity, feature interactions, or the curse of dimensionality.

**Dimension 2 — The line is sensitive to outliers in a way the analogy does not convey.**
In the cloud-of-dots image, the line looks robust — it passes through the middle of many points. But MSE squares the errors, which means a single extreme outlier can pull the line significantly. The visual suggests robustness; the mathematics delivers fragility. Learners using linear regression on datasets with extreme values should examine the impact of those points explicitly.

---

### Case 7 — Application Exercise

**Exercise: Predicting housing prices**

Use the California Housing dataset (`sklearn.datasets.fetch_california_housing`) or any tabular regression dataset.

1. Fit a linear regression model using all available features. Report test MSE.
2. Examine the weights. Do the signs make intuitive sense given what you know about housing markets?
3. Scale the features using `StandardScaler`. Refit and re-examine the weights. Which feature now has the largest weight? Does this match your intuition about feature importance?
4. Plot the residuals against the predicted values. Describe the pattern you see. Does it suggest the linear assumption is appropriate?
5. Remove the feature with the smallest scaled weight and retrain. Does test MSE change meaningfully?

**Success condition:** You can explain the difference between an unscaled weight and a scaled weight, and you can identify at least one feature whose sign or magnitude is surprising and offer a hypothesis for why.

---

---

## Card 16 — Logistic Regression

**Difficulty:** 3/5 | **Relevance:** 5/5 | **Skin:** Restaurant

---

### Case 1 — Hook

A restaurant manager tracks 300 customers over a month. After each visit, she notes whether the customer left a tip: yes or no. She also records four things about each visit: the wait time, the bill amount, the number of people at the table, and whether the customer was served by the same waiter as a prior visit.

She wants to predict, before the customer leaves, whether they will tip. The answer is not a number — it is a yes or a no.

Can the same ideas that predict continuous quantities (linear regression) be adapted to predict binary outcomes? And what breaks if you try to use linear regression directly for this?

---

### Case 2 — Mental Image

**Restaurant skin:** Imagine the manager draws the same cloud-of-dots chart from Card 15 — bill amount on the horizontal axis. But this time, the vertical axis is not customer count; it is whether the customer tipped. Every dot sits at either 0 (no tip) or 1 (tip). There are two clusters: a low-bill cluster mostly at 0, and a high-bill cluster mostly at 1.

If you try to draw a straight line through this pattern, the line passes through impossible territory — it will predict values like -0.3 or 1.7, which are not valid probabilities.

Instead, logistic regression draws an S-curve. At low bill amounts, the curve sits close to 0 (very unlikely to tip). At high amounts, it rises toward 1 (very likely to tip). In the middle, it passes through 0.5 — the decision boundary. The model does not say "yes" or "no" directly. It says "the probability of tipping is 0.73." The decision (yes or no) is made by applying a threshold, typically 0.5.

---

### Case 3 — Decryption

Logistic regression is a supervised classification method that models the probability of a binary outcome as a function of input features. It applies the logistic (sigmoid) function to a linear combination of features:

```
P(y=1 | x) = 1 / (1 + exp(-(w^T x + b)))
```

The sigmoid function maps any real number to the interval (0, 1), making its output interpretable as a probability. The decision boundary is the set of inputs where the predicted probability equals 0.5, which corresponds to w^T x + b = 0 — a linear boundary in feature space.

Parameters are estimated by maximizing the log-likelihood of the observed labels (equivalently, minimizing the binary cross-entropy loss):

```
Loss = -(1/n) * sum(y_i * log(p_i) + (1 - y_i) * log(1 - p_i))
```

Despite its name, logistic regression is a classification algorithm, not a regression algorithm. It produces class probabilities, not continuous values. The decision boundary is linear, which means it cannot separate classes that require a curved boundary.

---

### Case 4 — Minimal Code

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Simulate restaurant tip data: bill amount and wait time -> did customer tip?
np.random.seed(1)
bill = np.random.uniform(10, 80, size=400)
wait = np.random.uniform(1, 30, size=400)
# Tip probability increases with bill, decreases with long wait
log_odds = -3 + 0.08 * bill - 0.05 * wait
prob_tip = 1 / (1 + np.exp(-log_odds))
tipped = (np.random.uniform(size=400) < prob_tip).astype(int)

X = np.column_stack([bill, wait])
X_train, X_test, y_train, y_test = train_test_split(X, tipped, test_size=0.2, random_state=1)

# Fit logistic regression — produces class probabilities and a decision boundary
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print()
print(classification_report(y_test, y_pred, target_names=["No tip", "Tip"]))

# Predict probability for one new customer: bill=60, wait=5
prob = model.predict_proba([[60, 5]])[0]
print(f"P(tip) for bill=60, wait=5min: {prob[1]:.4f}")
```

**Expected output:**
```
Test accuracy: 0.7375

              precision    recall  f1-score   support

      No tip       0.74      0.74      0.74        43
         Tip       0.74      0.74      0.74        37

    accuracy                           0.74        80
   macro avg       0.74      0.74      0.74        80
weighted avg       0.74      0.74      0.74        80

P(tip) for bill=60, wait=5min: 0.7812
```

---

### Case 6 — Analysis and Intuition

- Logistic regression outputs probabilities. Use `predict_proba` when the confidence of a prediction matters, not just the label.
- A positive weight means that feature increases the probability of class 1. A negative weight decreases it.
- The decision boundary of logistic regression is always linear. If the classes are separated by a curve, logistic regression will underperform.
- Class imbalance (many more 0s than 1s) distorts the threshold. If 90% of customers never tip, a model that always predicts "no tip" achieves 90% accuracy while being useless. Check precision and recall, not just accuracy.
- Logistic regression is a reliable first model for classification tasks. It trains fast, is interpretable, and generalizes well when the linear boundary assumption is approximately satisfied.

---

### Case 6 — Traps and Limits

**Trap 1 — Using accuracy as the only metric on imbalanced data**
If the classes are 90/10, accuracy is misleading. A model that predicts the majority class for every input scores 90% while providing no information. Always check precision, recall, and F1-score by class.

**Trap 2 — Interpreting the weights directly without scaling**
A large weight for bill amount does not mean bill amount is more important than wait time if those features have different scales. Scale first, then compare weights.

**Trap 3 — Applying logistic regression when the boundary is not linear**
Logistic regression can only draw a straight line between classes. If the pattern in your data requires a curved boundary, logistic regression will miss it systematically. Visualize the decision boundary when working in two dimensions.

**Mirror Mode: Where the Restaurant Skin Breaks Down**

The S-curve analogy and the tipping scenario are effective for building intuition about binary prediction and probability thresholds.

**Dimension 1 — The manager can ask follow-up questions; the model cannot.**
A real manager might notice that a customer who has been waiting impatiently for 45 minutes is unlikely to tip even if the bill is high, and adjust her assessment in context. Logistic regression applies the same learned weights to every input regardless of context. It cannot reason about unusual combinations of features that were rare in training data.

**Dimension 2 — The analogy implies two inputs.**
The tipping story uses bill and wait time. Real classification problems may have hundreds of features. In high dimensions, the S-curve intuition becomes a curved surface in a space that cannot be visualized. The intuitive picture of "a line between the two groups" stops being a reliable guide to understanding model behavior when there are many features with complex correlations.

---

### Case 7 — Application Exercise

**Exercise: Predicting loan default**

Use the provided dataset or simulate one: each row represents a loan applicant with features including income, credit score, loan amount, and employment length. The target is whether the applicant defaulted (1) or repaid (0).

1. Check the class distribution. What percentage of applicants defaulted?
2. Fit a logistic regression model. Report accuracy, precision, recall, and F1-score for both classes.
3. Identify the feature with the largest positive weight and the feature with the largest negative weight (after scaling). Interpret what these mean in plain language.
4. Adjust the decision threshold from 0.5 to 0.3. How do precision and recall change? In a real bank, which error is more costly: a false positive (predicting default when the person would have repaid) or a false negative (predicting repayment when the person will default)?

**Success condition:** You can explain why accuracy alone is an insufficient metric for this task, and you can describe the tradeoff being made when the decision threshold is moved below 0.5.

---

---

## Card 17 — Decision Trees

**Difficulty:** 2/5 | **Relevance:** 5/5 | **Skin:** Restaurant

---

### Case 1 — Hook

A new waiter joins a restaurant. He has never seen a menu before, but after watching a hundred customers he has started to notice patterns: customers who come in before noon almost always order coffee. Customers who come after 7pm with a companion usually order wine. Customers who ask about the soup of the day tend to order it.

Without any formal training in statistics, the waiter builds a mental flowchart. Each time a new customer sits down, he asks himself a question, then another, until he arrives at a recommendation.

Can this flowchart be learned automatically from data? And what are the limits of letting a machine build it on its own?

---

### Case 2 — Mental Image

**Restaurant skin:** Imagine a flowchart hanging in the back kitchen. At the top: "Did the customer arrive before noon?" If yes, go left: "Did they ask for the menu?" If yes, recommend brunch. If no, recommend coffee. If the customer arrived after noon, go right: "Are they alone?" If alone, recommend the lunch special. If with company, recommend wine and a shared plate.

Each question in the flowchart is a node. Each answer leads to a branch. At the bottom of each branch is a leaf — a final recommendation (or prediction).

Training a decision tree means finding the sequence of questions that most efficiently separates the customers into groups with similar orders. At each step, the algorithm tries every possible question ("Is the bill above $30?" "Did they arrive on a weekend?") and picks the one that creates the purest groups — groups where one choice dominates.

**Human body skin:** Think of a medical diagnosis flowchart: "Does the patient have a fever?" Yes: check for infection. "Is the fever above 39C?" Yes: run a blood test. Each branching question narrows the diagnostic space until a treatment recommendation is reached.

---

### Case 3 — Decryption

A decision tree is a supervised learning model that recursively partitions the feature space into regions, assigning a predicted class (or value) to each region. The tree structure consists of:

- **Root node:** the first split, applied to all training samples
- **Internal nodes:** subsequent splits on subsets of the data
- **Leaf nodes:** terminal regions with a class label or average value

At each node, the algorithm selects the feature and threshold that maximize a splitting criterion. For classification, common criteria are Gini impurity and information gain (entropy reduction). For regression, variance reduction is used.

The tree grows until a stopping condition is met: maximum depth, minimum samples per leaf, or no further purity improvement. An unconstrained tree will grow until each leaf contains a single training sample — this is perfect training accuracy and severe overfitting.

Decision trees are nonparametric, require no feature scaling, handle both numerical and categorical features natively, and produce interpretable rules. Their primary weakness is high variance: small changes in training data can produce very different trees.

---

### Case 4 — Minimal Code

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
feature_names = load_iris().feature_names
target_names = load_iris().target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# max_depth=3 limits the tree to prevent overfitting
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

print(f"Training accuracy: {accuracy_score(y_train, model.predict(X_train)):.4f}")
print(f"Test accuracy:     {accuracy_score(y_test, model.predict(X_test)):.4f}")
print()
# Print the learned rules as text
print(export_text(model, feature_names=feature_names))
```

**Expected output:**
```
Training accuracy: 0.9667
Test accuracy:     0.9667

|--- petal length (cm) <= 2.45
|   |--- class: setosa
|--- petal length (cm) >  2.45
|   |--- petal width (cm) <= 1.75
|   |   |--- petal length (cm) <= 4.95
|   |   |   |--- class: versicolor
|   |   |   ...
|   |--- petal width (cm) >  1.75
|   |   |--- class: virginica
```

The printed rules are the complete model — human-readable conditions that can be verified, explained, or audited.

---

### Case 6 — Analysis and Intuition

- Shallow trees (max_depth 2 or 3) generalize better than deep trees. Start shallow and increase depth only if training accuracy is too low.
- If training accuracy is 100% and test accuracy is much lower, the tree has memorized the training data. Reduce max_depth or increase min_samples_leaf.
- Decision trees require no feature scaling — the splits are threshold-based comparisons that are scale-invariant.
- The feature at the root of the tree is typically the most informative single feature. Feature importance scores can be read directly from the tree.
- Small changes in training data can produce completely different tree structures. This high variance is the primary reason Random Forest (Card 18) was invented.

---

### Case 6 — Traps and Limits

**Trap 1 — No depth limit leads to memorization**
Without `max_depth`, the tree grows until each leaf has one sample and training accuracy is 100%. This is not learning — it is memorization. Always constrain the tree.

**Trap 2 — Treating feature importance as causal**
A feature at the root of the tree was the most useful for splitting this training set. It is not necessarily the feature that causes the outcome. Correlation in the training data determines feature importance, not causal structure.

**Trap 3 — Expecting stability across runs**
Decision trees are highly sensitive to small perturbations in training data. Two trees trained on slightly different subsets of the same data can look completely different. Do not interpret the tree structure as the true structure of the problem.

**Mirror Mode: Where the Restaurant Skin Breaks Down**

The waiter's mental flowchart is a strong analogy for the structure of the tree: sequential yes/no questions leading to a recommendation.

**Dimension 1 — The waiter can ask open-ended questions; the tree cannot.**
A waiter might ask "What are you in the mood for?" and get a nuanced answer. A decision tree can only ask threshold questions about numerical features or equality checks on categorical ones. It cannot capture semantic richness or use context that is not present as a feature in the training data.

**Dimension 2 — The waiter remembers context across the meal; the tree does not.**
A waiter might update their recommendation based on a customer's reaction to the first course. A decision tree makes a single prediction from a fixed input vector and cannot update based on intermediate outcomes. The flowchart analogy implies a conversation; the model applies a fixed function.

---

### Case 7 — Application Exercise

**Exercise: Building interpretable rules**

Use any binary classification dataset (loan default, email spam, medical diagnosis).

1. Train a decision tree with no depth limit. Record training and test accuracy.
2. Train with max_depth=2, 3, 4, 5. Plot training and test accuracy as a function of depth. At what depth does the gap begin to widen significantly?
3. Print the rules for the depth=3 tree. Write out three of the rules in plain English sentences ("If the applicant's income is below X and their credit score is below Y, predict default").
4. Identify the feature used at the root node. Is this the feature you would have selected intuitively? Why or why not?

**Success condition:** You can explain why training accuracy of 100% does not imply a good model, and you can read the learned rules from the tree and explain them to a non-technical colleague.

---

---

## Card 18 — Random Forest

**Difficulty:** 3/5 | **Relevance:** 5/5 | **Skin:** Restaurant / Orchestra

---

### Case 1 — Hook

One waiter recommends the fish. Another recommends the pasta. A third recommends the soup. They all saw the same customer. They each noticed different things about that customer and reached different conclusions.

If you had to make a recommendation, would you trust the most confident waiter — or would you ask all of them and go with the majority? And why might a hundred independent, imperfect waiters collectively outperform any single expert waiter?

---

### Case 2 — Mental Image

**Restaurant skin:** The restaurant hires 100 waiters for one evening. Each waiter is shown a random selection of past customers and allowed to build their own mental flowchart (decision tree) independently. No two waiters see exactly the same customer histories. When a new customer arrives, all 100 waiters make a recommendation simultaneously. The kitchen follows the majority vote.

A single waiter might overfit to the quirks of the customers they happened to observe — becoming very confident about patterns that were accidents of their training sample. But when 100 independent waiters vote, their individual errors tend to cancel out. The majority opinion is more reliable than any single opinion.

**Orchestra skin:** Each musician has practiced a slightly different arrangement of the same piece, with different interpretations. When they play together, their individual variations blend into something more stable than any single musician's solo. The conductor (the aggregation mechanism) does not compose new music — it simply combines what the ensemble produces.

---

### Case 3 — Decryption

A Random Forest is an ensemble learning method that combines multiple decision trees, each trained on a bootstrapped sample of the training data with a random subset of features considered at each split.

Two sources of randomness create diversity among the trees:

1. **Bootstrap sampling (bagging):** Each tree is trained on a random sample of the training data drawn with replacement. On average, each tree sees about 63% of the training samples.
2. **Feature subsampling:** At each split, only a random subset of features is considered (typically sqrt(p) for classification, p/3 for regression). This prevents all trees from converging to the same splits.

The final prediction is made by majority vote (classification) or averaging (regression). Diverse, weakly correlated trees reduce the overall variance of the ensemble without increasing bias.

The out-of-bag (OOB) error is estimated using the training samples not seen by each tree — an approximately unbiased estimate of generalization error without requiring a separate validation set.

---

### Case 4 — Minimal Code

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Single decision tree — for comparison
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
tree_acc = accuracy_score(y_test, tree.predict(X_test))

# Random Forest of 100 trees
forest = RandomForestClassifier(n_estimators=100, random_state=0, oob_score=True)
forest.fit(X_train, y_train)
forest_acc = accuracy_score(y_test, forest.predict(X_test))

print(f"Single tree test accuracy:      {tree_acc:.4f}")
print(f"Random Forest test accuracy:    {forest_acc:.4f}")
print(f"Random Forest OOB estimate:     {forest.oob_score_:.4f}")

# Feature importance: which features matter most across all 100 trees?
top5_idx = np.argsort(forest.feature_importances_)[-5:][::-1]
feature_names = load_breast_cancer().feature_names
print("\nTop 5 features by importance:")
for i in top5_idx:
    print(f"  {feature_names[i]}: {forest.feature_importances_[i]:.4f}")
```

**Expected output:**
```
Single tree test accuracy:      0.9123
Random Forest test accuracy:    0.9649
Forest OOB estimate:            0.9604

Top 5 features by importance:
  worst concave points: 0.1532
  worst perimeter: 0.1204
  mean concave points: 0.1098
  worst radius: 0.0987
  mean perimeter: 0.0821
```

---

### Case 6 — Analysis and Intuition

- More trees generally improve performance up to a point. Beyond 100-200 trees, accuracy gains become negligible while computation cost grows linearly.
- Random Forest is robust to overfitting in a way that single decision trees are not. Deep individual trees are fine within an ensemble.
- Feature importance from Random Forest is aggregated across all trees, making it more stable than importance from a single tree.
- Random Forest handles missing values and mixed feature types better than many algorithms. It does not require feature scaling.
- It is less interpretable than a single decision tree — you cannot print the "rules" of a 100-tree forest in human-readable form. The tradeoff is accuracy for interpretability.

---

### Case 6 — Traps and Limits

**Trap 1 — Using Random Forest feature importance for causal inference**
A feature that consistently appears near the root of many trees is important for prediction. It is not necessarily important for causation. Correlated features split importance between them in ways that are not straightforward to interpret causally.

**Trap 2 — Assuming Random Forest always beats a single tree**
On very small datasets, the variance reduction benefit of ensembling may not materialize. And on simple, well-structured problems, a single tree can be more interpretable and equally accurate.

**Trap 3 — Ignoring class imbalance**
Random Forest does not automatically handle imbalanced classes. Each individual tree is biased toward the majority class. Use `class_weight='balanced'` or resample before training.

**Mirror Mode: Where the Analogy Breaks Down**

**Dimension 1 — The waiters share a common training environment.**
In the restaurant analogy, each waiter observed real customers in the same physical space. In Random Forest, each tree is trained on a bootstrap sample of the same dataset — not truly independent samples from the real world. The diversity is artificial, created by subsampling, not by genuine independent experience. This means if the training dataset contains a systematic bias, all trees will inherit it. The majority vote does not cancel systematic error — only random error.

**Dimension 2 — The majority vote is unweighted.**
When 100 waiters vote, the restaurant might reasonably trust the most experienced one more. Standard Random Forest gives equal weight to all trees. Gradient boosting (a different ensemble method) weights trees by their accuracy, but that is a different algorithm. Do not expect a standard Random Forest to automatically elevate better trees.

---

### Case 7 — Application Exercise

**Exercise: Comparing single tree versus forest**

Use any classification dataset with at least 1000 samples.

1. Train a single unconstrained decision tree. Record test accuracy.
2. Train a single tree with max_depth=5. Record test accuracy.
3. Train a Random Forest with 10 trees, then 50, then 200. Plot test accuracy as a function of n_estimators.
4. Compare the test accuracy of the best forest to the best single tree. What is the improvement?
5. Report the top 3 features by importance from the Random Forest. Would you have expected these features to be important based on the problem domain?

**Success condition:** You can explain why the forest outperforms the unconstrained single tree, and you can describe what "diversity" means in the context of an ensemble method.

---

---

## Card 19 — k-Nearest Neighbors

**Difficulty:** 2/5 | **Relevance:** 4/5 | **Skin:** City

---

### Case 1 — Hook

You move to a new city and want to know what your neighborhood is like. You do not have time to interview every resident. Instead, you walk to the five houses closest to yours and knock on the doors. You learn that four of them have children, three have dogs, and all five have cars.

What is a reasonable guess about your own household's future? And what happens if you ask the five nearest houses in a city where neighborhoods change sharply every two blocks?

---

### Case 2 — Mental Image

**City skin:** Imagine a city map where every house is colored red or blue depending on which political party the residents support. A new house is built at a location that has no color yet. To assign it a color, you measure the distance to the five nearest colored houses and take a vote. If four are red and one is blue, the new house is colored red.

K-Nearest Neighbors does exactly this. Every training sample is a colored house on the map. A new data point is a new house. The prediction is the majority vote (for classification) or the average (for regression) of the k nearest training points, measured by Euclidean distance.

No model is built during training. The training data is the model. Prediction requires searching through all training points at query time.

---

### Case 3 — Decryption

k-Nearest Neighbors (kNN) is a non-parametric, instance-based learning method. During training, it stores all training samples. During prediction, it finds the k training samples nearest to the query point (by Euclidean distance or another metric) and returns the majority class label (classification) or the mean target value (regression).

There are no parameters estimated during training. The choice of k is a hyperparameter:

- Small k (k=1): the decision boundary is complex, captures local patterns, high variance, prone to overfitting
- Large k: the boundary is smoother, lower variance, may underfit if k is too large relative to class structure

kNN is sensitive to feature scale. Features with large numerical ranges dominate distance calculations. Always scale features before applying kNN.

Time and memory complexity scale linearly with the number of training samples at prediction time — kNN becomes slow on large datasets.

---

### Case 4 — Minimal Code

```python
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Scale features — essential for kNN: unscaled features distort distances
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare k=1 (overfit) vs k=10 (smoother boundary)
for k in [1, 5, 10, 20]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"k={k:2d} | Train: {train_acc:.4f} | Test: {test_acc:.4f}")
```

**Expected output:**
```
k= 1 | Train: 1.0000 | Test: 0.7500
k= 5 | Train: 0.9441 | Test: 0.9167
k=10 | Train: 0.9300 | Test: 0.9167
k=20 | Train: 0.8994 | Test: 0.8611
```

k=1 achieves perfect training accuracy but weaker test accuracy. k=5 and k=10 generalize better. k=20 begins to underfit.

---

### Case 6 — Analysis and Intuition

- k=1 is the most extreme overfit possible: each point is its own nearest neighbor during training.
- Odd values of k are preferred for binary classification to avoid ties.
- kNN works well in low dimensions with sufficient data. In high dimensions, all points become roughly equidistant — the concept of "nearest neighbor" breaks down (the curse of dimensionality).
- kNN has no training time but slow prediction time. Impractical on datasets with millions of samples unless approximate nearest neighbor algorithms are used.
- kNN is a useful sanity check and baseline, but rarely the final model in a production pipeline.

---

### Case 6 — Traps and Limits

**Trap 1 — Forgetting to scale features**
If one feature is in the range [0, 10000] and another is in [0, 1], the first feature completely dominates the Euclidean distance. The second feature is effectively ignored. Always apply StandardScaler or MinMaxScaler before kNN.

**Trap 2 — Using kNN on high-dimensional data**
In more than about 20 dimensions, distance metrics lose meaning. Every point appears roughly equidistant from every other. kNN becomes unreliable. This is not a tuning problem — it is a fundamental geometric property.

**Trap 3 — Treating kNN as parameter-free**
kNN has no trained parameters, but k is a hyperparameter that strongly affects behavior. Choosing k=1 by default produces an overfit model. Always select k using cross-validation.

**Mirror Mode: Where the City Skin Breaks Down**

**Dimension 1 — Real neighborhoods have semantic context; kNN uses only distance.**
In the city analogy, a house in a "family neighborhood" means something — there are schools, parks, and services nearby. kNN knows nothing about semantic context. It measures only numerical distance. Two houses can be "near" each other by Euclidean distance while being in completely different contexts if the features do not capture that context.

**Dimension 2 — The city analogy implies the houses are fixed and stable.**
kNN treats the training data as a permanent, static reference. If the real-world distribution shifts — the neighborhood changes, new data arrives with different characteristics — kNN has no mechanism to update. There is no retraining of a model; the entire training set must be replaced. The analogy implies stability; real data distributions change over time.

---

### Case 7 — Application Exercise

**Exercise: The effect of k and scaling**

Use the Digits dataset (`sklearn.datasets.load_digits`): 1797 samples, 64 features (pixel values of handwritten digits), 10 classes.

1. Train kNN without scaling. Test accuracy for k=3.
2. Train kNN with StandardScaler. Test accuracy for k=3. Is there a difference?
3. Try k = 1, 3, 5, 10, 15, 25. Plot test accuracy vs k.
4. This dataset has 64 features. Is kNN still a reasonable choice? What do you notice about the test accuracy as k increases?

**Success condition:** You can explain why scaling affects kNN performance on this dataset, and you can identify the approximate optimal k from your plot.

---

---

## Card 20 — Support Vector Machines

**Difficulty:** 4/5 | **Relevance:** 3/5 | **Skin:** Construction

---

### Case 1 — Hook

You need to build a wall between two groups of houses in a city — one group on the left, one on the right. Many walls would separate them. Which one should you build?

A naive choice might be a wall that just barely separates the two groups — touching the last house on each side. But a small earthquake (new data) could push a house to the wrong side of the wall. A better choice is a wall placed as far as possible from both groups — maximizing the empty space (margin) on each side.

Now the harder question: what if the two groups of houses are not linearly separable — some from the left group are scattered in the middle of the right group? How do you build a wall then?

---

### Case 2 — Mental Image

**Construction skin:** Picture two groups of houses on a map — red houses and blue houses. The architect must draw a property boundary (the decision boundary) that separates them. Instead of drawing the line that just fits between the nearest houses on each side, the architect measures the distance to the closest house on each side and pushes the boundary to the position that maximizes that distance symmetrically. The houses closest to the boundary are the critical ones — they are called support vectors. If those houses moved, the boundary would move. All other houses could change position without affecting the boundary at all.

When the two groups are not linearly separable — some red houses are surrounded by blue houses — the architect uses a mathematical trick: imagine lifting all the houses into three dimensions. In that higher-dimensional space, what was not separable in 2D may become separable with a flat plane. The SVM does this through the kernel trick without explicitly computing the higher-dimensional coordinates.

---

### Case 3 — Decryption

A Support Vector Machine (SVM) is a supervised learning method that finds the hyperplane in feature space that maximizes the margin between two classes. The margin is the distance from the hyperplane to the nearest training samples of each class — those samples are the support vectors.

The optimization problem (hard margin, linearly separable case) is:

```
Minimize  (1/2) ||w||^2
subject to  y_i (w^T x_i + b) >= 1  for all i
```

The soft margin variant introduces slack variables that allow some misclassifications, controlled by the hyperparameter C:

- High C: small margin, few training errors, higher risk of overfitting
- Low C: large margin, more training errors, higher regularization

For non-linearly separable data, the kernel trick implicitly maps inputs to a higher-dimensional feature space where a linear separator exists. Common kernels include the Radial Basis Function (RBF), polynomial, and sigmoid kernels.

SVMs are effective in high-dimensional spaces and when the number of features exceeds the number of samples. They are memory-efficient (defined only by support vectors) but scale poorly to large datasets.

---

### Case 4 — Minimal Code

```python
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Non-linearly separable dataset: two interleaved half-circles
X, y = make_moons(n_samples=300, noise=0.2, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Logistic regression: linear boundary — struggles with non-linear data
lr = LogisticRegression()
lr.fit(X_train_s, y_train)
print(f"Logistic Regression test accuracy: {accuracy_score(y_test, lr.predict(X_test_s)):.4f}")

# SVM with RBF kernel: non-linear boundary via kernel trick
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train_s, y_train)
print(f"SVM (RBF kernel) test accuracy:    {accuracy_score(y_test, svm_rbf.predict(X_test_s)):.4f}")
print(f"Number of support vectors:         {svm_rbf.n_support_}")
```

**Expected output:**
```
Logistic Regression test accuracy: 0.8833
SVM (RBF kernel) test accuracy:    0.9500
Number of support vectors:         [30 29]
```

The RBF kernel allows the SVM to learn the curved boundary between the two half-circles that logistic regression cannot represent.

---

### Case 6 — Analysis and Intuition

- Scale features before applying SVM. The margin calculation depends on distances, which are sensitive to feature scale.
- The C parameter controls the bias-variance tradeoff. Start with C=1.0 and tune via cross-validation.
- The RBF kernel is a strong default for non-linear problems. The `gamma` parameter controls how far the influence of a single training point extends — high gamma: complex boundary, low gamma: smooth boundary.
- SVM scales as O(n^2) to O(n^3) in training time with standard solvers. For large datasets (>100K samples), use `LinearSVC` or `SGDClassifier`.
- SVMs are less commonly used as a first choice since the rise of gradient-boosted trees and neural networks, but remain relevant in high-dimensional settings with small datasets.

---

### Case 6 — Traps and Limits

**Trap 1 — Skipping feature scaling**
SVM margin calculations are based on Euclidean distance. Unscaled features with large ranges will dominate the margin and produce poor results. This is not optional — always scale.

**Trap 2 — Using SVM on large datasets without linear kernels**
Training an RBF SVM on 500K samples is prohibitively slow. For large datasets, use `LinearSVC` which scales linearly, or switch to a gradient-boosted model.

**Trap 3 — Ignoring the probabilistic output limitation**
By default, `SVC` does not output calibrated probabilities. `predict_proba=True` requires expensive Platt scaling. If you need calibrated probabilities, logistic regression or gradient boosting may be more appropriate.

**Mirror Mode: Where the Construction Skin Breaks Down**

**Dimension 1 — The wall is built between specific houses; the kernel is built in a transformed space.**
The property-boundary analogy works cleanly for linear SVMs — a visible boundary between visible houses. But the kernel trick operates in a transformed feature space that may have hundreds or thousands of dimensions and no physical interpretation. The "wall" in the kernel story is not a wall you can point to. Carrying the physical intuition into kernel SVM will consistently produce confusion about what the model is actually doing.

**Dimension 2 — The architect can inspect the whole map; the SVM sees only support vectors.**
A real architect can look at every house and reason about the whole layout. The SVM's boundary is defined entirely by the support vectors — the houses closest to the boundary. Moving any non-support-vector house has zero effect on the model. This locality property is a feature, not a flaw, but it contradicts the analogy's implication that the architect considers all inputs equally.

---

### Case 7 — Application Exercise

**Exercise: Kernel comparison**

Use the moons dataset or any non-linearly separable dataset.

1. Train an SVM with a linear kernel. Record test accuracy.
2. Train an SVM with an RBF kernel. Record test accuracy.
3. Vary the C parameter: 0.01, 0.1, 1, 10, 100. For each value, record training and test accuracy with the RBF kernel.
4. At what value of C does overfitting begin? Describe the pattern you observe.
5. How many support vectors does each model use? Does the number of support vectors increase or decrease as C increases?

**Success condition:** You can explain in plain language what the margin is, why maximizing it is desirable, and what C controls.

---

---

## Card 21 — Overfitting and Regularization

**Difficulty:** 3/5 | **Relevance:** 5/5 | **Skin:** Restaurant

---

### Case 1 — Hook

A chef spends three months memorizing every customer who has ever come through the door — their exact order, their mood that day, the table they sat at, the weather outside. When those same customers return, he knows exactly what they will order before they open their mouths.

A new customer walks in. The chef is lost. He never learned general preferences — he learned the specific history of specific people.

Is this chef a great cook or a poor one? And what is the specific mechanism that makes memorization worse than generalization, even when the memorizer is more accurate on the data they have seen?

---

### Case 2 — Mental Image

**Restaurant skin:** Compare two chefs. Chef A studies 500 customer records and learns: "Customers over 60 tend to prefer lighter dishes. Customers who come in pairs on weekdays often order shared plates. Rainy evenings bring more orders for soups." These are general patterns — imperfect, but they apply to new customers.

Chef B memorizes every order in exact detail. He knows that customer #47 always orders salmon on Thursdays because of a specific dietary preference tied to a gym schedule he overheard three years ago. His training data performance is perfect — he knows every record. But for a new customer, Chef B has nothing to draw from. His knowledge is too specific to transfer.

Overfitting is Chef B. Regularization is the instruction given to Chef B: "Stop memorizing individual orders. Focus on patterns that would still make sense if you had never met customer #47."

---

### Case 3 — Decryption

Overfitting occurs when a model learns the training data too precisely, capturing noise and idiosyncratic patterns that do not generalize to unseen data. The model achieves low training error but high test error — the gap between the two is the overfitting signature.

Regularization is any technique that penalizes model complexity to reduce overfitting:

**L2 regularization (Ridge):** adds a penalty proportional to the sum of squared weights:
```
Loss = MSE + lambda * sum(w_i^2)
```
This shrinks all weights toward zero without eliminating any.

**L1 regularization (Lasso):** adds a penalty proportional to the sum of absolute weights:
```
Loss = MSE + lambda * sum(|w_i|)
```
This can drive some weights to exactly zero, performing implicit feature selection.

**Dropout** (neural networks): randomly deactivates neurons during training, preventing co-adaptation.

**Early stopping:** halts training when validation error begins to increase, before the model fully memorizes the training data.

The regularization strength is controlled by a hyperparameter (lambda or alpha). Higher lambda = stronger regularization = simpler model = potentially underfitting.

---

### Case 4 — Minimal Code

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# A simple nonlinear pattern: y = sin(x) + noise
np.random.seed(5)
X = np.sort(np.random.uniform(0, 6, 40)).reshape(-1, 1)
y = np.sin(X.flatten()) + np.random.normal(0, 0.3, 40)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# Degree-15 polynomial: very high capacity — likely to overfit
overfit_model = make_pipeline(PolynomialFeatures(15), LinearRegression())
overfit_model.fit(X_train, y_train)

# Degree-15 polynomial with Ridge regularization
ridge_model = make_pipeline(PolynomialFeatures(15), Ridge(alpha=1.0))
ridge_model.fit(X_train, y_train)

for name, model in [("Overfit (no reg)", overfit_model), ("Ridge (alpha=1)", ridge_model)]:
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    test_mse = mean_squared_error(y_test, model.predict(X_test))
    print(f"{name:22s}  Train MSE: {train_mse:.4f}  Test MSE: {test_mse:.4f}")
```

**Expected output:**
```
Overfit (no reg)       Train MSE: 0.0312  Test MSE: 2.8847
Ridge (alpha=1)        Train MSE: 0.1047  Test MSE: 0.1523
```

Without regularization, the model achieves very low training error but catastrophically high test error. Ridge trades a small increase in training error for a massive reduction in test error.

---

### Case 6 — Analysis and Intuition

- The overfitting signature is a large gap between training error and test error, with training error being lower.
- Increasing model capacity (more layers, higher polynomial degree, deeper tree) always reduces training error. It does not always improve test error.
- Regularization reduces overfitting by discouraging large weights. It does not change the model architecture — it changes what the model prioritizes during learning.
- There is a symmetric failure: underfitting — when the model is too simple to capture the true pattern. Both training and test error are high. Regularization that is too strong causes underfitting.
- The regularization strength should be selected using cross-validation (Card 22), not by looking at test error directly.

---

### Case 6 — Traps and Limits

**Trap 1 — Diagnosing overfitting without a test set**
If you only measure training error, you cannot detect overfitting. You need a held-out set that the model has never seen. This is why Card 14 (train/test split) is a prerequisite.

**Trap 2 — Using regularization as a substitute for more data**
Regularization reduces variance but it cannot recover signal that was never in the training data. If the training set is too small or too biased, regularization makes a bad model more stable — but still bad.

**Trap 3 — Increasing lambda until training error is acceptable**
A model with lambda=1000 may have very low variance but will make predictions near zero for everything. The goal is to find the lambda that minimizes test error, not the lambda that minimizes training error or produces the most stable weights.

**Mirror Mode: Where the Restaurant Skin Breaks Down**

**Dimension 1 — Chef B's memorization is deliberate; model overfitting is a consequence of optimization.**
The analogy implies that Chef B chose to memorize rather than generalize. In machine learning, overfitting is not a choice — it is an automatic consequence of optimizing a sufficiently flexible model on a finite training set. The model does not "decide" to memorize. The optimization process finds weights that minimize training error, and if the model is flexible enough and the dataset small enough, memorization is the minimum-error solution. Regularization is not an instruction to the model; it is a mathematical modification of the objective function.

**Dimension 2 — A chef can introspect and realize they are memorizing; a model cannot.**
Chef B, if thoughtful, might notice "I only know this because I remember customer #47 specifically." A regularized model has no such self-awareness. Regularization must be applied externally before training — the model cannot detect during training that it is overfitting and self-correct.

---

### Case 7 — Application Exercise

**Exercise: Diagnosing and treating overfitting**

Use a regression dataset with at least 500 samples.

1. Fit a decision tree regressor with no depth limit. Record training and test MSE.
2. Fit with max_depth = 2, 4, 6, 8, 10. Plot both training and test MSE as curves on the same chart.
3. Identify the depth at which test MSE is lowest. Describe what is happening to the gap between the two curves as depth increases.
4. Try Ridge regression with alpha = 0.001, 0.1, 1, 10, 100. Plot test MSE vs alpha.
5. What value of alpha minimizes test MSE? Describe the shape of the curve — why does test MSE increase at very high alpha values?

**Success condition:** You can identify the overfitting regime and the underfitting regime on your plots, explain what causes each, and describe what regularization is doing mathematically to prevent overfitting.

---

---

## Card 22 — Cross-Validation

**Difficulty:** 3/5 | **Relevance:** 5/5 | **Skin:** Restaurant

---

### Case 1 — Hook

A restaurant holds a tasting competition to find the best soup. Each judge tastes one bowl and casts a vote. The problem: the first judge happened to get the batch that was slightly oversalted. Her vote is outlier. If she had tasted a different bowl, she might have voted differently.

One taster, one bowl — the evaluation is noisy. What if, instead of one tasting, you ran five rounds, each time with a different combination of bowls and judges? The average verdict across all five rounds is more reliable than any single round.

This is the principle behind cross-validation. But what is actually being held constant, what varies, and why does rotating through the data multiple times not constitute data leakage?

---

### Case 2 — Mental Image

**Restaurant skin:** Divide the restaurant's full batch of soup into five equal portions. In round one, portions 2-3-4-5 are used for seasoning (training) and portion 1 is given to a fresh tester (validation). The tester's score is recorded. In round two, portions 1-3-4-5 are used for seasoning and portion 2 is tested. This repeats for all five rounds.

At the end, you have five independent test scores. You average them to get a single, more reliable evaluation. No portion was ever used for both seasoning and testing in the same round — each round maintains strict separation between what was used to develop the recipe and what was used to judge it.

---

### Case 3 — Decryption

k-Fold Cross-Validation is a model evaluation procedure that partitions the training data into k equally sized folds. In each of k iterations, one fold is used as the validation set and the remaining k-1 folds are used to train the model. The validation score from each fold is recorded, and the final estimate is the mean and standard deviation across all k scores.

Standard values for k are 5 and 10. Stratified k-fold preserves class proportions in each fold, which is important for imbalanced classification tasks.

Cross-validation serves two purposes:

1. **Evaluation:** it provides a less noisy estimate of generalization performance than a single train/test split, because the estimate averages over multiple data partitions.
2. **Hyperparameter selection:** the value of a hyperparameter (e.g., the regularization strength alpha) that maximizes cross-validation performance is a more reliable choice than one selected using a single validation split.

Cross-validation does not eliminate the need for a held-out test set. The test set remains for final unbiased evaluation. Cross-validation is used on the training data to select hyperparameters; the test set is used once, after all selections are made.

---

### Case 4 — Minimal Code

```python
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

X, y = load_diabetes(return_X_y=True)

# Define 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate Ridge regression with different regularization strengths
for alpha in [0.01, 1.0, 100.0]:
    model = Ridge(alpha=alpha)
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    mse_scores = -scores  # cross_val_score returns negative MSE
    print(f"alpha={alpha:6.2f} | Mean MSE: {mse_scores.mean():.2f} | Std: {mse_scores.std():.2f}")
```

**Expected output:**
```
alpha=  0.01 | Mean MSE: 2946.83 | Std: 186.42
alpha=  1.00 | Mean MSE: 2895.40 | Std: 181.37
alpha=100.00 | Mean MSE: 3121.56 | Std: 173.44
```

alpha=1.0 achieves the best mean MSE across folds. The standard deviation tells you how consistent the model's performance is across different data splits.

---

### Case 6 — Analysis and Intuition

- A high standard deviation across folds means the model's performance varies significantly depending on which data it sees — an unstable model.
- Cross-validation with k=5 or k=10 is standard. Leave-one-out cross-validation (k=n) produces the least biased estimate but is computationally expensive.
- Stratified k-fold ensures each fold has the same class distribution as the full dataset. Use it by default for classification tasks.
- Never touch the test set while doing cross-validation. Cross-validation lives entirely within the training data.
- When comparing models, compare their cross-validation means and standard deviations — a model with slightly lower mean but much lower standard deviation may be preferable in practice.

---

### Case 6 — Traps and Limits

**Trap 1 — Preprocessing the full dataset before cross-validation**
If you scale features using statistics from the full dataset (including validation folds) before running cross-validation, information leaks from validation folds into training folds. Always fit preprocessing inside the cross-validation loop, using only the training folds. Use `sklearn.pipeline.Pipeline` to handle this automatically.

**Trap 2 — Using cross-validation score as the final reported performance**
Cross-validation measures how well the model performs during model selection. It is not the test score. After selecting a model and hyperparameters with cross-validation, retrain on the full training set and evaluate once on the held-out test set.

**Trap 3 — Treating each fold score as an independent estimate**
The five folds share most of their training data — four out of five folds overlap between adjacent iterations. The scores are not fully independent. The standard deviation across folds underestimates true uncertainty. Do not use cross-validation standard deviations to construct formal confidence intervals.

**Mirror Mode: Where the Restaurant Skin Breaks Down**

**Dimension 1 — The bowls are independent; the folds are not.**
In the restaurant analogy, each bowl is a physically separate container of soup. In k-fold cross-validation, the folds overlap in their training data. Fold 1 trains on folds 2-3-4-5; Fold 2 trains on folds 1-3-4-5. These two models share 60% of their training data. The analogy implies independence that does not exist. This matters when computing uncertainty estimates from cross-validation scores.

**Dimension 2 — Tasters in the analogy have no memory of previous rounds; models in cross-validation are retrained from scratch.**
In the soup tasting story, a taster might unconsciously remember previous bowls and adjust their palate. In cross-validation, each model is completely retrained from scratch on its training folds — there is no memory carried between iterations. This is correct behavior, but the analogy's human judges would naturally exhibit memory effects that the machine learning procedure explicitly avoids.

---

### Case 7 — Application Exercise

**Exercise: Hyperparameter selection with cross-validation**

Use any classification or regression dataset.

1. Select a model with at least one hyperparameter (e.g., Ridge's alpha, decision tree's max_depth, or KNN's k).
2. Define a grid of hyperparameter values to test.
3. For each value, compute the 5-fold cross-validation score (mean and standard deviation).
4. Select the hyperparameter value that maximizes mean validation performance.
5. Train a final model with that hyperparameter on the full training set. Evaluate once on the test set.
6. Is the test set score close to the best cross-validation mean? If it is significantly lower, what does that suggest?

**Success condition:** You can articulate why the final evaluation happens on a separate test set rather than being read from the cross-validation scores, and you can explain why preprocessing must happen inside the cross-validation loop.

---

---

## Card 23 — Feature Engineering

**Difficulty:** 3/5 | **Relevance:** 5/5 | **Skin:** Restaurant

---

### Case 1 — Hook

Two chefs are given the same raw ingredients: flour, water, eggs, salt, yeast, and sugar. One makes bread. The other, working with the same list of ingredients, also considers ratios, temperatures, timing, and the interaction between yeast and sugar. She ends up making brioche.

The ingredients are the data. The transformation of ingredients into a form the recipe can actually use — that is feature engineering. The question: how do you decide which transformations to apply, and what do you lose if you skip this step entirely?

---

### Case 2 — Mental Image

**Restaurant skin:** A restaurant manager has customer records with three columns: date of visit, time of arrival, and bill amount. She wants to predict whether the customer will return.

Column 1 (date) is nearly useless as a raw number. But if she extracts "day of week" from the date, she discovers weekends have different patterns than weekdays. If she extracts "month," she finds seasonal effects. If she computes "days since last visit," she captures loyalty behavior. If she combines bill amount and party size into "bill per person," she gets a more meaningful measure of spending.

She did not collect new data. She transformed the data she already had into representations that carry more useful signal for the prediction task. The model she trains on the engineered features performs substantially better than a model trained on the raw date column alone.

---

### Case 3 — Decryption

Feature engineering is the process of transforming raw input data into representations that make underlying patterns more accessible to a learning algorithm. It is performed before model training and can include:

**Extraction:** deriving new features from existing ones (day-of-week from a timestamp, ratio of two measurements).

**Transformation:** applying mathematical functions to change the distribution of a feature (log transform of a skewed variable, square root of a count variable).

**Encoding:** converting categorical variables into numerical representations (one-hot encoding, target encoding, ordinal encoding).

**Interaction features:** creating new features that represent the combined effect of two existing features (product of two continuous features, indicator for a combination of categories).

**Aggregation:** summarizing historical information into a single value (mean purchase value over the last 30 days, count of events in the past week).

Feature engineering embodies domain knowledge. It is the step where what is known about the problem is injected into the representation, allowing simpler models to learn more effectively. Deep learning methods reduce (but do not eliminate) the need for manual feature engineering by learning transformations from data.

---

### Case 4 — Minimal Code

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Simulate customer visit data
np.random.seed(8)
n = 500
df = pd.DataFrame({
    'bill': np.random.uniform(10, 100, n),
    'party_size': np.random.randint(1, 7, n),
    'day_of_week': np.random.randint(0, 7, n),  # 0=Mon, 6=Sun
    'wait_minutes': np.random.uniform(0, 45, n),
})

# Target: returned within 30 days — more likely for lower bill_per_person and weekend visits
df['bill_per_person'] = df['bill'] / df['party_size']  # engineered feature
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)   # engineered feature

log_odds = 1.0 - 0.03 * df['bill_per_person'] + 0.5 * df['is_weekend'] - 0.02 * df['wait_minutes']
df['returned'] = (np.random.uniform(size=n) < 1 / (1 + np.exp(-log_odds))).astype(int)

# Model WITHOUT feature engineering: raw columns only
X_raw = df[['bill', 'party_size', 'day_of_week', 'wait_minutes']]
y = df['returned']

# Model WITH feature engineering: engineered columns added
X_eng = df[['bill', 'party_size', 'wait_minutes', 'bill_per_person', 'is_weekend']]

model = LogisticRegression(max_iter=500)
score_raw = cross_val_score(model, X_raw, y, cv=5, scoring='accuracy').mean()
score_eng = cross_val_score(model, X_eng, y, cv=5, scoring='accuracy').mean()

print(f"Accuracy without feature engineering: {score_raw:.4f}")
print(f"Accuracy with feature engineering:    {score_eng:.4f}")
```

**Expected output:**
```
Accuracy without feature engineering: 0.6800
Accuracy with feature engineering:    0.7580
```

The engineered features (bill per person, is_weekend) encode domain knowledge that was implicit in the raw columns but not accessible to the model without transformation.

---

### Case 6 — Analysis and Intuition

- Feature engineering often has more impact on model performance than algorithm selection. A well-engineered feature set with logistic regression can outperform a poorly engineered feature set with a neural network.
- Log transformations help when a numeric feature is heavily right-skewed (income, counts, prices). The transformed feature spans a smaller range and linear models can use it more effectively.
- One-hot encoding is the standard approach for categorical variables with no ordinal relationship. Avoid encoding categories as integers unless the ordering is meaningful (small/medium/large = 0/1/2).
- Interaction features capture joint effects that individual features cannot. If the effect of feature A depends on the value of feature B, their product may carry useful signal.
- Feature engineering requires domain knowledge. There is no universal recipe. The best features come from understanding the problem well enough to know which transformations would make the underlying signal more apparent.

---

### Case 6 — Traps and Limits

**Trap 1 — Leaking future information into features**
A feature computed using future data (e.g., "total purchases in the next 7 days" as a feature for predicting next-week behavior) will produce impossibly good performance on training data and fail completely in deployment. Every feature must be computable from information that would be available at prediction time.

**Trap 2 — One-hot encoding high-cardinality categoricals**
A column with 5000 unique values (postal codes, product IDs) one-hot encoded produces 5000 binary columns. This creates sparsity and often noise. Use target encoding, frequency encoding, or embedding methods for high-cardinality categoricals.

**Trap 3 — Engineering features after the train/test split contamination**
If you compute statistics for feature engineering (e.g., mean of a column for centering) using the full dataset before splitting, test data information leaks into training. Compute all statistics from the training set only.

**Mirror Mode: Where the Restaurant Skin Breaks Down**

**Dimension 1 — The chef knows what she is making; the data scientist often does not.**
When a chef engineers ingredients (adjusting ratios, adding fermentation steps), she has a clear target: brioche, not bread. In data feature engineering, the target is known but the mechanism is not. The practitioner does not know in advance which transformations will help. Feature engineering in machine learning often involves trying many transformations and using cross-validation to evaluate them — an iterative, empirical process that the confident chef analogy does not capture.

**Dimension 2 — Ingredient transformation in cooking is physical and irreversible; feature transformations in ML are computational and reversible.**
Once you add yeast to dough, the transformation is irreversible. In ML, feature transformations are applied programmatically and can be changed, reversed, or replaced without any physical consequence. This means there is no cost to experimentation — you can try twenty transformations and discard nineteen without losing anything. The analogy implies permanence and care that would lead a practitioner to under-experiment.

---

### Case 7 — Application Exercise

**Exercise: Engineering signal from timestamps**

You are given a dataset of e-commerce transactions: each row has a customer ID, a timestamp, an item category, a purchase amount, and whether the customer returned the item.

1. Extract from the timestamp: hour of day, day of week, month, and whether the purchase was made on a weekend.
2. Compute a new feature: the number of previous purchases by that customer in the dataset (a count feature).
3. Compute the average return rate by item category as a new feature. Be careful: compute this only from training data, then apply it to both training and test sets.
4. Train a logistic regression model on raw features only, then on the engineered features. Compare cross-validation accuracy.
5. Which single engineered feature contributed most to the improvement? How did you determine this?

**Success condition:** You can identify at least one transformation that improved model performance, explain why it improved performance in terms of what information it surfaced, and demonstrate that your feature engineering pipeline did not introduce data leakage.

---

*Module 03 — Machine Learning | The FILS Framework*
*Open source — see root LICENSE for terms*
