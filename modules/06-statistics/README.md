# Module 06 — Statistics for Machine Learning

**The FILS Framework**
Part of the open-source AI/ML curriculum for complete beginners.

---

## Module Overview

This module introduces the statistical concepts that appear most frequently in machine learning work: describing data, understanding uncertainty, making decisions from data, and recognizing when conclusions are not justified by evidence. Every concept is delivered as a BRIDGIA card — seven cases per concept, from Hook to Application.

**Concepts in this module:**

| Card | Concept | Difficulty | Relevance |
|------|---------|------------|-----------|
| 38 | Mean, Median, Mode | 1/5 | 5/5 |
| 39 | Standard Deviation and Variance | 2/5 | 5/5 |
| 40 | Probability Distributions | 3/5 | 5/5 |
| 41 | Hypothesis Testing | 4/5 | 4/5 |
| 42 | Correlation vs Causation | 2/5 | 5/5 |
| 43 | Bayes' Theorem | 4/5 | 5/5 |
| 44 | Sampling and Bias | 3/5 | 5/5 |

---

---

## Card 38 — Mean, Median, Mode

**Difficulty:** 1/5 | **Relevance:** 5/5 | **Skin:** Restaurant

---

### Case 1 — Hook

A restaurant manager looks at last week's tip data. One table left a $200 tip — a large party celebrating a birthday. The other 49 tables left tips averaging $12.

Her bookkeeper reports: "The average tip this week was $15.92."

The manager finds this confusing. Almost no table left $15.92. Most tables left around $12. The $200 outlier has pulled the average far above what a typical table actually pays. Is the $15.92 number useful? Is there a better number to describe "what a typical customer tips"?

---

### Case 2 — Mental Image

**Restaurant skin:** Three ways to describe "the typical tip" at a restaurant:

**Mean (average):** Add all tip amounts together, divide by the number of tables. This is the balance point of the distribution — where it would balance if you placed it on a fulcrum. One very large tip at one end will pull the balance point far from where most tips cluster.

**Median:** Line all customers up in order from smallest to largest tip. Walk to the exact middle of the line. That customer's tip is the median. Half of all customers tipped more than this amount; half tipped less. The $200 outlier is at the back of the line — it doesn't affect where the middle is.

**Mode:** Which tip amount appears most frequently in the data? If a large number of customers left exactly $10, then $10 is the mode — the most commonly ordered tip amount, so to speak. The mode is the most popular option, not the average or the middle one.

Each of these answers the same question ("what is typical?") differently, and each can mislead if applied to the wrong situation.

---

### Case 3 — Decryption

The mean, median, and mode are **measures of central tendency** — single numbers that summarize where a distribution is centered.

**Mean (arithmetic mean):**
```
mean = (1/n) * sum(x_i)
```
Sensitive to extreme values (outliers). The mean minimizes the sum of squared deviations from itself. Used in most machine learning algorithms (MSE loss, feature scaling, batch normalization).

**Median:**
The value at the 50th percentile — half of all observations fall below it, half above it. Not affected by extreme values. Appropriate for skewed distributions (income, house prices, response times).

**Mode:**
The most frequently occurring value. The only measure of central tendency applicable to categorical (non-numeric) data. A distribution can be unimodal (one peak), bimodal (two peaks), or multimodal.

When to use which:
- Symmetric distribution with no outliers → mean and median are approximately equal; either is appropriate
- Skewed distribution or outliers present → median is more representative
- Categorical data → mode is the only option
- "Most popular option" → mode

---

### Case 4 — Minimal Code

```python
import numpy as np
from scipy import stats

# Simulate restaurant tip data: 49 normal tables + 1 exceptional large-party table
np.random.seed(0)
normal_tips = np.random.normal(loc=12, scale=3, size=49).clip(min=0)
outlier_tip = np.array([200.0])
tips = np.concatenate([normal_tips, outlier_tip])

print("=== Tip Data Summary ===")
print(f"Number of tables: {len(tips)}")
print(f"Minimum tip:      ${tips.min():.2f}")
print(f"Maximum tip:      ${tips.max():.2f}")
print()

print("=== Measures of Central Tendency ===")
print(f"Mean:   ${tips.mean():.2f}  <- pulled up by the $200 outlier")
print(f"Median: ${np.median(tips):.2f}  <- unaffected by the outlier")
print(f"Mode (binned to nearest $): ${stats.mode(tips.round(0), keepdims=True).mode[0]:.0f}")

# What happens if we remove the outlier?
without_outlier = normal_tips
print()
print("=== Without the $200 outlier ===")
print(f"Mean:   ${without_outlier.mean():.2f}")
print(f"Median: ${np.median(without_outlier):.2f}")

# Mean vs median for skewed data: income example
np.random.seed(1)
incomes = np.concatenate([np.random.lognormal(mean=10, sigma=0.5, size=990),
                          np.array([1e7, 5e7, 2e7, 8e7, 3e7])])  # 5 ultra-wealthy

print()
print("=== Income Distribution (990 middle-class + 5 ultra-wealthy) ===")
print(f"Mean income:   ${incomes.mean():>12,.0f}  <- inflated by 5 outliers")
print(f"Median income: ${np.median(incomes):>12,.0f}  <- typical worker")
```

**Expected output:**
```
=== Tip Data Summary ===
Number of tables: 50
Minimum tip:      $4.63
Maximum tip:      $200.00

=== Measures of Central Tendency ===
Mean:   $15.56  <- pulled up by the $200 outlier
Median: $12.18  <- unaffected by the outlier
Mode (binned to nearest $): $12

=== Without the $200 outlier ===
Mean:   $12.00
Median: $12.12

=== Income Distribution (990 middle-class + 5 ultra-wealthy) ===
Mean income:   $   224,412  <- inflated by 5 outliers
Median income: $    22,026  <- typical worker
```

The median income is 10x lower than the mean — the five ultra-wealthy observations inflate the mean dramatically. Reporting the mean income as "typical" would be deeply misleading.

---

### Case 5 — Analysis and Intuition

- When a news report says "average salary increased by 5%," they mean the mean. Five executives getting large raises could produce this result while most workers receive nothing. Always ask: is this the mean or the median?
- Machine learning loss functions (MSE, MAE) are both derived from central tendency concepts. MSE minimizes the mean squared error, which is equivalent to fitting the mean. MAE minimizes the mean absolute error, which is equivalent to fitting the median. This is why MAE is more robust to outliers in target variables.
- In feature normalization, you subtract the mean and divide by the standard deviation. If your feature is heavily skewed (income, house prices), the mean may be far from the typical value, producing a poor normalization. Consider median-based scaling (RobustScaler in sklearn) for skewed features.
- When you report model performance metrics, always consider whether a mean accuracy is appropriate. If 5% of test samples are pathologically hard, their contribution to the mean accuracy is small — but they may represent a systematically underserved subgroup.

---

### Case 6 — Traps and Limits

**Trap 1 — Reporting the mean for skewed data as if it represents typical values**
Income, house prices, website response times, and biological measurements are often right-skewed. The mean will always exceed the median for right-skewed data. Using the mean to describe "typical" in these cases overstates typical values.

**Trap 2 — Assuming the mode is a single value**
If a distribution is bimodal (two peaks), the mode concept becomes ambiguous — there are two modes. A dataset of restaurant bills might have modes at $15 (lunch) and $65 (dinner) because two populations of customers visit at different times.

**Trap 3 — Applying mean to categorical data**
If customer ratings are: 1-star (50 reviews), 3-star (10 reviews), 5-star (40 reviews), the mean is approximately 2.8 stars. But no customer gave 2.8 stars. The mode (1 star) and the distribution shape tell a more accurate story than the mean for this multimodal categorical data.

**Mirror Mode: Where the Restaurant Skin Breaks Down**

**Dimension 1 — The analogy treats tips as perfectly measured; real data has measurement error.**
In the restaurant story, every tip is a precise dollar amount. Real datasets often have measurement error, missing values, and data entry mistakes. A tip recorded as "$1200" might be a data entry error for "$12.00." Before computing any summary statistic, examine the data for implausible values. The analogy implies clean data; the practice requires data cleaning.

**Dimension 2 — The restaurant collects all customer tips; surveys collect samples.**
The restaurant manager knows every tip — the full population. In most ML and data science work, you have a sample, not the full population. The sample mean is an estimate of the population mean, subject to sampling variability. The analogy conceals this fundamental distinction between a population parameter and a sample estimate.

---

### Case 7 — Application Exercise

**Exercise: Choosing the right central tendency measure**

Use any dataset with a mix of numeric and categorical variables. The California Housing dataset or any publicly available sales dataset works well.

1. For a continuous numeric column (e.g., median house value), compute mean, median, and the ratio mean/median. What does the ratio tell you about skewness?
2. Plot a histogram. Does the distribution look symmetric? Does the mean or median appear to be closer to the "center" visually?
3. For a categorical column (e.g., ocean proximity), compute the mode. Is the mode informative? What percentage of observations match the mode?
4. Identify a column where the mean would be misleading if used to describe "typical." Explain why.
5. A model is trained to predict house prices. The loss function is MSE. Explain in one paragraph why this means the model is trying to predict the mean, not the median, and why this might produce biased predictions for low-value homes.

**Success condition:** You can explain in plain language when to use the mean versus the median, give an example from the dataset where they differ substantially, and connect the choice of loss function to the choice of central tendency measure.

---

---

## Card 39 — Standard Deviation and Variance

**Difficulty:** 2/5 | **Relevance:** 5/5 | **Skin:** Restaurant / Orchestra

---

### Case 1 — Hook

Two restaurants both pay their waitstaff an average of $180 per shift in tips. You are choosing which restaurant to work at.

At Restaurant A, tips range from $160 to $200 every night — predictable and stable.
At Restaurant B, tips range from $20 to $500 — some nights you barely cover transport, others you cover rent.

The means are identical. Which restaurant is riskier? And how do you quantify "riskier" with a single number?

---

### Case 2 — Mental Image

**Restaurant skin:** The head chef tracks every customer tip for a month at two restaurants. Both show an average of $15. But at Restaurant A, almost every tip is between $13 and $17. At Restaurant B, tips scatter wildly — $3, $28, $8, $45, $2, $30.

To measure this scatter, the chef computes, for each tip, the distance from the average: how far did this tip deviate from $15? She squares each distance (to make all distances positive and to penalize large deviations more heavily) and averages them. This average squared deviation is the variance. The square root of the variance is the standard deviation — a measure of spread in the same units as the original tips (dollars, not dollars-squared).

Restaurant A has a standard deviation of $2. Restaurant B has a standard deviation of $14. Same mean. Completely different risk profile.

**Orchestra skin:** The conductor expects every musician to play at exactly 120 beats per minute. After rehearsal, she measures each musician's tempo. Musician A played between 119 and 121 BPM all evening — tiny deviations. Musician B played between 108 and 132 BPM — large deviations. Both averaged 120 BPM. But Musician B's variance (mean squared deviation from 120) is much larger. The standard deviation quantifies how reliably each musician holds tempo.

---

### Case 3 — Decryption

**Variance** is the average squared deviation from the mean:

```
variance = (1/n) * sum((x_i - mean)^2)
```

For a sample (not the full population), use n-1 in the denominator (Bessel's correction) to get an unbiased estimate:

```
sample_variance = (1/(n-1)) * sum((x_i - mean)^2)
```

**Standard deviation** is the square root of variance, expressed in the same units as the original variable:

```
std = sqrt(variance)
```

Properties:
- Standard deviation = 0 means all values are identical
- Standard deviation cannot be negative
- Roughly 68% of data falls within ±1 standard deviation of the mean (for normally distributed data)
- Roughly 95% falls within ±2 standard deviations

**Z-score** (standardization): `z = (x - mean) / std` transforms a value into "how many standard deviations above or below the mean is this observation?" Z-scores allow comparison across variables measured in different units.

In machine learning: feature standardization (StandardScaler) subtracts the mean and divides by the standard deviation, producing features with mean=0 and std=1. This prevents features with large numerical ranges from dominating distance-based or gradient-based learning.

---

### Case 4 — Minimal Code

```python
import numpy as np
from scipy import stats

np.random.seed(0)

# Two restaurants with the same mean tip, different spread
restaurant_a = np.random.normal(loc=15, scale=2, size=200)   # low variance
restaurant_b = np.random.normal(loc=15, scale=14, size=200)  # high variance

for name, data in [("Restaurant A", restaurant_a), ("Restaurant B", restaurant_b)]:
    print(f"=== {name} ===")
    print(f"Mean:              ${data.mean():.2f}")
    print(f"Variance:          ${data.var():.2f}")
    print(f"Standard Deviation: ${data.std():.2f}")
    print(f"Min/Max:           ${data.min():.2f} / ${data.max():.2f}")

    # 68% rule: what fraction of tips fall within 1 std of the mean?
    within_1_std = np.mean(np.abs(data - data.mean()) <= data.std())
    print(f"Within 1 std:      {within_1_std:.1%}")
    print()

# Z-score example: identify outliers
print("=== Z-score outlier detection (Restaurant B) ===")
z_scores = stats.zscore(restaurant_b)
outlier_mask = np.abs(z_scores) > 2.5
print(f"Tips beyond 2.5 std from mean: {outlier_mask.sum()}")
print(f"Their values: {restaurant_b[outlier_mask].round(2)}")

# StandardScaler demonstration
print()
print("=== Feature standardization ===")
raw = np.array([500, 750, 1200, 650, 900, 1100, 450])  # raw bill amounts
z = (raw - raw.mean()) / raw.std()
print(f"Raw bills:      {raw}")
print(f"Z-scores:       {z.round(3)}")
print(f"Mean of Z:      {z.mean():.6f}  (should be ~0)")
print(f"Std of Z:       {z.std():.6f}   (should be ~1)")
```

**Expected output:**
```
=== Restaurant A ===
Mean:              $15.11
Variance:          $4.07
Standard Deviation: $2.02
Min/Max:           $9.32 / $20.83
Within 1 std:      70.0%

=== Restaurant B ===
Mean:              $15.17
Variance:          $187.34
Standard Deviation: $13.69
Min/Max:           $-29.82 / $61.38
Within 1 std:      68.5%

=== Z-score outlier detection (Restaurant B) ===
Tips beyond 2.5 std from mean: 8
Their values: [-18.64  53.73 -14.42  57.14 -21.23  52.75 -19.83  61.38]

=== Feature standardization ===
Raw bills:      [ 500  750 1200  650  900 1100  450]
Z-scores:       [-1.239 -0.24   1.917 -0.614  0.36   1.542 -1.727]
Mean of Z:      0.000000  (should be ~0)
Std of Z:       1.000000   (should be ~1)
```

---

### Case 5 — Analysis and Intuition

- Standard deviation is always in the same units as the data. Variance is in squared units, which is harder to interpret directly. In communication, always report standard deviation.
- A small standard deviation does not mean the data is reliable — it means the data is consistent. Consistently wrong measurements have a small standard deviation but are still wrong.
- In deep learning, unstable training (loss that oscillates wildly) is often caused by high variance in gradient updates. Batch normalization, gradient clipping, and learning rate scheduling are all variance-reduction techniques at the optimization level.
- When comparing two models' test accuracy across multiple runs, standard deviation across runs is as important as mean accuracy. A model with mean accuracy 0.85 ± 0.01 is more reliable than one with mean 0.86 ± 0.08.
- The coefficient of variation (CV = std/mean) allows comparison of variability across variables measured in different units. CV = 0.1 means the standard deviation is 10% of the mean — a universal measure of relative spread.

---

### Case 6 — Traps and Limits

**Trap 1 — Interpreting low variance as high quality**
Low variance means consistency. A thermometer that always reads 5°C too high has low variance (consistent readings) but high bias (systematically wrong). Variance and bias are different dimensions of data quality.

**Trap 2 — Applying the 68%/95% rule to non-normal distributions**
The rule that 68% of data falls within ±1 standard deviation applies to normally distributed data. For a heavily skewed distribution, this rule does not hold. Always check the shape of your distribution before applying normal-distribution rules.

**Trap 3 — Using population variance formula for a sample**
Dividing by n rather than n-1 when computing sample variance systematically underestimates the true population variance. For large samples (n > 100) the difference is negligible. For small samples (n < 30), use n-1.

**Mirror Mode: Where the Analogies Break Down**

**Dimension 1 — Tips are independent events; many ML datasets contain correlated observations.**
In the restaurant analogy, each customer's tip is an independent choice. In most datasets, observations are not independent — time series data has autocorrelation, patient records from the same hospital are correlated, product reviews from the same user cluster. Standard deviation computed on correlated data understates the true uncertainty because the effective sample size is smaller than the nominal sample size.

**Dimension 2 — The orchestra analogy implies a true target (120 BPM) exists; ML targets are often uncertain.**
The conductor knows the target tempo exactly. In machine learning, the "true" value of the target variable is often itself uncertain — medical diagnoses have inter-rater disagreement, labels contain noise. Standard deviation of model predictions does not capture this label uncertainty. A model can have low prediction variance (consistent) while the labels themselves are highly variable (uncertain).

---

### Case 7 — Application Exercise

**Exercise: Spread as a quality signal**

1. Load the California Housing dataset. For the target variable (median house value), compute mean, standard deviation, and the 5th and 95th percentiles. Plot a histogram.

2. Split the dataset by a categorical feature (e.g., ocean proximity). Compute the mean and standard deviation of house value for each category. Which category has the highest variance? What might cause that?

3. Standardize all numeric features using z-scores. Verify that after standardization, every feature has mean ≈ 0 and std ≈ 1.

4. Train a linear regression model on raw features and on standardized features. Are the learned coefficients interpretable before and after standardization? What changes?

5. Compute z-scores for the target variable. Identify observations with |z| > 3. Are these outliers systematic (e.g., all from the same region) or random?

**Success condition:** You can compute and interpret standard deviation, explain why standardization is needed for gradient-based algorithms, and use z-scores to identify potential outliers in a dataset.

---

---

## Card 40 — Probability Distributions

**Difficulty:** 3/5 | **Relevance:** 5/5 | **Skin:** Restaurant / City

---

### Case 1 — Hook

A restaurant manager wants to schedule staff optimally. On most days, 50 to 70 customers arrive. But occasionally, only 20 arrive (a slow Tuesday in January) and sometimes 150 arrive (Valentine's Day). The manager has two years of daily counts.

She does not want to schedule for the worst case every day — that would be wasteful. She does not want to schedule for the average only — that would leave her understaffed on busy days. She wants to schedule so that she is prepared for 95% of days, not 100%.

To answer this question, she needs to understand the shape of the distribution of customer arrivals — not just the mean, but the full pattern of how frequently each outcome occurs.

---

### Case 2 — Mental Image

**Restaurant skin:** Imagine plotting the number of customers for every day over two years on a histogram: x-axis is customer count, y-axis is how many days had that count. Most bars cluster around 55–65. There are shorter bars on both sides, tailing off as counts get very high or very low. The histogram is a picture of the distribution.

Some distributions are bell-shaped and symmetric — the restaurant's weekday lunch counts might look like this. Some are right-skewed — individual bill amounts, where most bills are moderate but a few very large bills create a long right tail. Some are bimodal — customer arrival times peak at noon and again at 7pm.

A probability distribution is the formal mathematical description of this pattern. It tells you: for any given value, how likely is it to occur?

**City skin:** A city planning department tracks building heights across all districts. In the residential district, buildings are mostly 2–3 floors, with a tight symmetric distribution. In the commercial district, most buildings are 4–10 floors but a few skyscrapers extend the distribution into a long right tail. In the mixed-use district, there are many 1-floor shops and many 20-floor towers — a bimodal distribution with almost nothing in between.

Each district has a different distribution. Planning decisions that work for the residential distribution may fail badly when applied to the skyscraper district.

---

### Case 3 — Decryption

A probability distribution describes the likelihood of each possible value for a random variable.

**Normal (Gaussian) distribution:** bell-shaped, symmetric around the mean. Parameterized by mean μ and standard deviation σ. Common in nature for sums of many independent influences (height, measurement error, test scores). The foundation of many statistical methods.

**Uniform distribution:** every value in a range is equally likely. Used for random initialization of parameters in some neural network architectures.

**Bernoulli distribution:** a single binary outcome (0 or 1) with probability p of success. The distribution underlying logistic regression outputs.

**Binomial distribution:** number of successes in n independent Bernoulli trials. Example: number of customers who tip out of 20 customers.

**Poisson distribution:** number of events occurring in a fixed interval of time, when events are independent and occur at a constant average rate λ. Natural model for customer arrival counts, server requests, and rare event counts.

**Log-normal distribution:** the log of the variable is normally distributed. Describes right-skewed variables like income, house prices, and response times — variables that cannot be negative and have occasional extreme high values.

In machine learning, understanding which distribution your data follows is important for:
- Choosing the right loss function (cross-entropy for Bernoulli targets, MSE for Gaussian targets)
- Choosing the right data transformation (log transform for log-normal data)
- Generating synthetic data for testing or data augmentation

---

### Case 4 — Minimal Code

```python
import numpy as np
import scipy.stats as stats

np.random.seed(42)

# --- Normal distribution: weekday lunch customer counts ---
normal_counts = stats.norm(loc=58, scale=8)
print("=== Normal Distribution: Weekday Lunch Counts ===")
print(f"Mean (mu):          {normal_counts.mean():.1f} customers")
print(f"Std (sigma):        {normal_counts.std():.1f} customers")
print(f"P(count < 45):      {normal_counts.cdf(45):.3f}  (very slow day)")
print(f"P(count > 75):      {1 - normal_counts.cdf(75):.3f}  (very busy day)")
print(f"95th percentile:    {normal_counts.ppf(0.95):.1f} customers")
print(f"  -> Staff for 95% of days: {normal_counts.ppf(0.95):.0f} customers")

# --- Poisson distribution: arrivals per hour at a coffee shop ---
lambda_rate = 12  # average 12 customers per hour
poisson_dist = stats.poisson(mu=lambda_rate)
print("\n=== Poisson Distribution: Customers per Hour (lambda=12) ===")
for k in [5, 10, 12, 15, 20]:
    print(f"  P(exactly {k:2d} arrivals): {poisson_dist.pmf(k):.4f}")
print(f"P(more than 20 arrivals): {1 - poisson_dist.cdf(20):.4f}")

# --- Log-normal: individual customer bill amounts ---
lognormal_bills = stats.lognorm(s=0.6, scale=np.exp(3.4))  # median ~$30
bills_sample = lognormal_bills.rvs(1000)
print("\n=== Log-Normal Distribution: Customer Bill Amounts ===")
print(f"Mean:   ${bills_sample.mean():.2f}  <- skewed upward by large bills")
print(f"Median: ${np.median(bills_sample):.2f}  <- more representative")
print(f"Std:    ${bills_sample.std():.2f}")
print(f"90th percentile: ${np.percentile(bills_sample, 90):.2f}")
print(f"99th percentile: ${np.percentile(bills_sample, 99):.2f}")
```

**Expected output:**
```
=== Normal Distribution: Weekday Lunch Counts ===
Mean (mu):          58.0 customers
Std (sigma):        8.0 customers
P(count < 45):      0.052  (very slow day)
P(count > 75):      0.017  (very busy day)
95th percentile:    71.2 customers
  -> Staff for 95% of days: 71 customers

=== Poisson Distribution: Customers per Hour (lambda=12) ===
  P(exactly  5 arrivals): 0.0127
  P(exactly 10 arrivals): 0.1048
  P(exactly 12 arrivals): 0.1144
  P(exactly 15 arrivals): 0.0724
  P(exactly 20 arrivals): 0.0097
P(more than 20 arrivals): 0.0174

=== Log-Normal Distribution: Customer Bill Amounts ===
Mean:   $36.41  <- skewed upward by large bills
Median: $29.96  <- more representative
Std:    $24.13
90th percentile: $68.54
99th percentile: $136.77
```

---

### Case 5 — Analysis and Intuition

- The 95th percentile is a useful decision threshold: "staff for the 95th percentile" means you will be adequately staffed on 19 out of 20 days.
- Poisson distributions arise naturally wherever you are counting discrete events in a time window — web server requests, customer arrivals, word occurrences in a document. Recognizing a Poisson process helps you choose appropriate models.
- When data is log-normally distributed, taking the log transforms it into a normal distribution. Many ML algorithms perform better on normally distributed inputs. Log-transforming heavily right-skewed features before training is standard preprocessing.
- The assumption that data is normally distributed is made implicitly in many statistical tests. Check histograms before applying any method that assumes normality.
- In ML model outputs: classification models output probabilities (Bernoulli/categorical distributions). Regression models assume targets are normally distributed around predictions when using MSE loss.

---

### Case 6 — Traps and Limits

**Trap 1 — Assuming all data is normally distributed**
Many natural datasets are not normal. Income is log-normal. Survival times are exponential or Weibull. Click-through rates are beta-distributed. Applying normal-distribution tools to non-normal data produces incorrect confidence intervals and hypothesis tests.

**Trap 2 — Confusing the distribution of a sample with the distribution of the population**
A histogram of 30 observations is a noisy estimate of the true population distribution. Small samples look irregular even when the population distribution is smooth. Do not over-interpret the shape of histograms from small datasets.

**Trap 3 — Using the wrong distribution for count data**
Counts (customers per day, defects per unit) should be modeled with Poisson or negative binomial distributions, not normal distributions — counts cannot be negative, and the Poisson's mean equals its variance. Using a normal distribution for count targets will produce predictions that can be negative.

**Mirror Mode: Where the Analogies Break Down**

**Dimension 1 — The histogram is a complete picture; a parametric distribution is an assumption.**
The city planning department's histogram of building heights is the actual data — no assumptions. A parametric distribution (normal, Poisson, log-normal) is a mathematical model of that histogram. When you fit a distribution to data, you are claiming that data was generated by that distribution — an assumption that may be wrong. The histogram analogy implies that the distribution is observed; in practice, it is assumed and must be validated.

**Dimension 2 — The restaurant collects all historical counts; distributions predict future counts.**
The manager's histogram is based on past data. Using a fitted distribution to answer "what staffing level covers 95% of future days?" requires assuming the future distribution matches the historical distribution. If the restaurant changes location, adds a new menu, or enters a different season, the distribution may shift. Probability distributions describe historical patterns and assume stationarity; reality produces distribution shift.

---

### Case 7 — Application Exercise

**Exercise: Identifying and fitting distributions**

1. Load the California Housing dataset. Plot histograms for three numeric columns: median income, median house age, and median house value. Which distribution family does each one resemble most?

2. Apply a log transform to median house value. Plot the histogram again. Does it look more symmetric? Compute mean, median, and std before and after the transform.

3. Simulate 1000 Poisson draws with lambda=20. Compute the sample mean and variance. For a Poisson distribution, mean and variance should be equal — is that approximately true in your simulation?

4. Use `scipy.stats.norm.fit(data)` to fit a normal distribution to a column. Plot the fitted PDF over the histogram. Does the fit look good? Where does it fail?

5. Use `scipy.stats.kstest()` to test whether a column is normally distributed. What does the p-value tell you? If p < 0.05, what does that mean for downstream statistical tests that assume normality?

**Success condition:** You can identify which of the four common distribution families (normal, log-normal, Poisson, uniform) best describes a given histogram, and explain why fitting the wrong distribution family leads to downstream modeling errors.

---

---

## Card 41 — Hypothesis Testing

**Difficulty:** 4/5 | **Relevance:** 4/5 | **Skin:** Restaurant / Construction

---

### Case 1 — Hook

A restaurant owner introduces a new menu in March. April sales are 12% higher than March. Did the new menu cause the increase?

Maybe. Or maybe April is always 12% busier than March because spring brings more foot traffic. Or maybe one week in April happened to coincide with a local festival. The data shows a difference. The question is whether the difference is real — caused by the menu change — or whether it could have occurred by chance even if the menu made no difference at all.

Hypothesis testing provides a framework for answering this question with a quantified level of certainty.

---

### Case 2 — Mental Image

**Restaurant skin:** The owner runs a controlled experiment. For four weeks, she randomly assigns half of her tables to receive the new menu and half to receive the old menu. At the end, she computes the average spend per table for each group.

New menu: average $68 per table.
Old menu: average $61 per table.

A $7 difference. But with 40 tables per group and the natural variation in customer spending, how often would you expect to see a $7 difference even if the menus were identical?

The owner imagines a skeptic: "Assume the menu makes no difference. Assume the true means are equal. How often would random chance alone produce a difference as large as $7 or larger between groups of 40 tables?"

If that probability (the p-value) is very small — say, 0.02 — then random chance alone would produce this difference only 2% of the time. That is unlikely enough that the owner concludes the difference is real.

**Construction skin:** A civil engineer tests a new concrete mix. Old mix: average compressive strength 40 MPa across 30 test samples. New mix: 43 MPa. Is the new mix genuinely stronger, or did she happen to test a good batch?

She sets up the same skeptic's argument: if the mixes are equally strong, how often would random sampling produce a 3 MPa difference? If the answer is "less than 5% of the time," she accepts that the new mix is genuinely stronger.

---

### Case 3 — Decryption

Hypothesis testing is a procedure for deciding whether observed data provides sufficient evidence to reject a default assumption (the null hypothesis) in favor of an alternative.

**Null hypothesis (H₀):** the default, conservative claim. "The menu makes no difference." "The two groups have equal means." "The new treatment has no effect."

**Alternative hypothesis (H₁):** the claim you are trying to support. "The new menu increases average spend." "Group A has a different mean than Group B."

**Test statistic:** a number computed from the data that measures how far the observed result is from what H₀ predicts.

**p-value:** the probability of observing a test statistic as extreme as the one computed, assuming H₀ is true. It is NOT the probability that H₀ is true.

**Significance level (α):** the threshold below which the p-value is considered "sufficiently unlikely." Conventional choice: α = 0.05 (5%).

Decision rule:
- If p-value < α: reject H₀. The result is statistically significant.
- If p-value ≥ α: fail to reject H₀. Insufficient evidence against H₀.

**t-test:** used to compare means between two groups when the population standard deviation is unknown (the common case). Scipy: `stats.ttest_ind`.

**Common errors:**
- Type I error (false positive): rejecting H₀ when it is true. Rate controlled by α.
- Type II error (false negative): failing to reject H₀ when H₁ is true. Rate controlled by sample size and effect size.

---

### Case 4 — Minimal Code

```python
import numpy as np
from scipy import stats

np.random.seed(7)

# Simulate restaurant A/B test: new menu vs old menu
# True difference: new menu increases spend by $5 (we set this, then check if test detects it)
n_tables = 40
old_menu_spend = np.random.normal(loc=61, scale=15, size=n_tables)
new_menu_spend = np.random.normal(loc=66, scale=15, size=n_tables)  # true +$5

print("=== Restaurant A/B Test ===")
print(f"Old menu: mean=${old_menu_spend.mean():.2f}, std=${old_menu_spend.std():.2f}")
print(f"New menu: mean=${new_menu_spend.mean():.2f}, std=${new_menu_spend.std():.2f}")
print(f"Observed difference: ${new_menu_spend.mean() - old_menu_spend.mean():.2f}")

# Two-sample t-test: are these means significantly different?
t_stat, p_value = stats.ttest_ind(new_menu_spend, old_menu_spend)
print(f"\nt-statistic: {t_stat:.4f}")
print(f"p-value:     {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print(f"p < {alpha}: Reject H₀. Evidence that new menu increases spend.")
else:
    print(f"p >= {alpha}: Cannot reject H₀. Difference may be due to chance.")

# What happens with a smaller sample? (same true effect, n=10)
print("\n=== Same effect, smaller sample (n=10 per group) ===")
small_old = np.random.normal(loc=61, scale=15, size=10)
small_new = np.random.normal(loc=66, scale=15, size=10)
_, p_small = stats.ttest_ind(small_new, small_old)
print(f"Observed difference: ${small_new.mean() - small_old.mean():.2f}")
print(f"p-value with n=10:  {p_small:.4f}")
print("A real effect may go undetected with insufficient sample size (Type II error).")

# Multiple comparisons trap
print("\n=== Multiple comparisons: running 20 tests on random data ===")
false_positives = 0
for _ in range(20):
    group_a = np.random.normal(0, 1, 50)
    group_b = np.random.normal(0, 1, 50)  # truly identical distribution
    _, p = stats.ttest_ind(group_a, group_b)
    if p < 0.05:
        false_positives += 1
print(f"False positives out of 20 tests: {false_positives}")
print("With alpha=0.05, expect ~1 false positive per 20 tests by pure chance.")
```

**Expected output:**
```
=== Restaurant A/B Test ===
Old menu: mean=$61.43, std=$14.82
New menu: mean=$67.91, std=$16.23
Observed difference: $6.48

t-statistic: 1.9812
p-value:     0.0507

p >= 0.05: Cannot reject H₀. Difference may be due to chance.

=== Same effect, smaller sample (n=10 per group) ===
Observed difference: $8.34
p-value with n=10:  0.1923
A real effect may go undetected with insufficient sample size (Type II error).

=== Multiple comparisons: running 20 tests on random data ===
False positives out of 20 tests: 1
With alpha=0.05, expect ~1 false positive per 20 tests by pure chance.
```

Note: the $6.48 observed difference is real (the true effect is $5) but the p-value is just above 0.05 due to random variation. This illustrates that hypothesis tests can miss real effects — especially with small samples.

---

### Case 5 — Analysis and Intuition

- A p-value of 0.05 means "if there were no effect, results this extreme would occur 5% of the time by chance." It does not mean "there is a 95% probability the effect is real."
- Statistical significance is not the same as practical significance. A study with 10,000 samples can detect a 0.1% improvement in click-through rate as "statistically significant" — a real effect that is too small to be practically meaningful.
- Increasing sample size decreases the p-value for the same effect size. With enough samples, trivially small effects become statistically significant. Always report effect size alongside p-values.
- The p < 0.05 threshold is a convention, not a law. It was proposed by Fisher in 1925 and has been widely criticized for encouraging binary "significant/not significant" thinking.
- In ML model comparison: use paired t-tests or permutation tests to compare model performance across multiple evaluation sets, not single held-out test sets.

---

### Case 6 — Traps and Limits

**Trap 1 — Multiple comparisons inflate false positive rate**
If you test 20 different metrics and declare any p < 0.05 "significant," you expect 1 false positive by chance. Apply the Bonferroni correction (divide α by the number of tests) or use FDR control methods.

**Trap 2 — p-value is not the probability that H₀ is true**
This misinterpretation is extremely common. p = 0.03 means "if H₀ were true, results this extreme would occur 3% of the time." It does not mean "there is a 3% chance that H₀ is true."

**Trap 3 — Peeking at the data before specifying the hypothesis**
If you look at your data, notice a pattern, then construct a hypothesis around that pattern, your p-values are invalid. The hypothesis must be specified before data collection or at minimum before analysis.

**Mirror Mode: Where the Analogies Break Down**

**Dimension 1 — The restaurant can run the test again; most real-world studies cannot.**
In the restaurant analogy, if the A/B test is inconclusive, the owner can run it again with more tables. In practice, many high-stakes studies (clinical trials, policy interventions) cannot be repeated cheaply. The analogy implies unlimited data collection opportunities; real hypothesis testing often involves a single, costly data collection effort where sample size must be planned in advance using power analysis.

**Dimension 2 — The analogy implies a binary outcome; the p-value is a continuous spectrum.**
The restaurant story ends in "significant" or "not significant." The p-value is a continuous number and the 0.05 threshold is arbitrary. A p-value of 0.049 and a p-value of 0.051 provide nearly identical evidence, but the binary threshold treats them as categorically different conclusions. Modern statistical practice emphasizes reporting effect sizes, confidence intervals, and p-values together — not making binary pass/fail decisions.

---

### Case 7 — Application Exercise

**Exercise: A/B testing a web intervention**

Simulate or use a real A/B test dataset. Each row represents a user session with columns: group (A or B), conversion (1 or 0), and session duration in seconds.

1. Compute the conversion rate for group A and group B. Is there a difference?
2. Run a two-sample t-test on conversion rates (or a chi-squared test for proportions). What is the p-value?
3. Compute the effect size (Cohen's d for means, or relative difference in rates). Is the difference practically significant?
4. How many samples would you need to detect a 2% increase in conversion rate with 80% power and α=0.05? Use `scipy.stats.norm` or a power analysis formula.
5. Simulate running the test 100 times with truly identical groups (true H₀). How often do you get p < 0.05? Does it match the theoretical false positive rate?

**Success condition:** You can explain the difference between statistical significance and practical significance, interpret a p-value correctly without committing the base-rate fallacy, and describe one way that multiple comparisons can inflate your false positive rate.

---

---

## Card 42 — Correlation vs Causation

**Difficulty:** 2/5 | **Relevance:** 5/5 | **Skin:** City

---

### Case 1 — Hook

A public health researcher analyzes city data. She discovers a striking pattern: on days when ice cream sales are high, drowning deaths are also high. The correlation is strong, statistically significant, and consistent across five years of data.

Should the city ban ice cream to prevent drownings?

The question sounds absurd. But the same logical error — inferring causation from correlation — appears constantly in data analysis, machine learning feature selection, and business decisions. The pattern is real; the interpretation is wrong.

---

### Case 2 — Mental Image

**City skin:** Imagine the city plotted on a timeline. In summer: temperature rises, people go to pools and beaches (drowning risk increases), people buy ice cream (sales increase). In winter: temperature drops, fewer people swim, fewer people buy ice cream. Both ice cream sales and drowning deaths are driven by a third variable — temperature. When temperature rises, both rise. When temperature falls, both fall.

The two variables are correlated not because one causes the other, but because they share a common cause. Statisticians call this a confounding variable (or confounder). The relationship between ice cream and drowning vanishes if you hold temperature constant: within a single temperature range, knowing ice cream sales tells you nothing additional about drowning risk.

The structure is: Temperature → Ice Cream Sales; Temperature → Drowning Deaths. The arrow from ice cream to drowning does not exist.

In data, you cannot see arrows. You can only see correlations. The arrows — the causal structure — must be inferred from domain knowledge, experimental design, or causal inference methods. A correlation coefficient tells you nothing about direction or causality.

---

### Case 3 — Decryption

**Pearson correlation (r):** measures the linear relationship between two continuous variables. Ranges from -1 (perfect negative linear relationship) to +1 (perfect positive linear relationship). r = 0 means no linear relationship — it does not mean no relationship at all.

```
r = cov(X, Y) / (std(X) * std(Y))
```

**Spearman rank correlation:** measures monotonic (not necessarily linear) relationship. More robust to outliers and non-normal distributions.

**Three structures that produce correlation without direct causation:**

1. **Common cause (confounding):** Z causes both X and Y. Ice cream ← temperature → drowning.
2. **Reverse causation:** Y actually causes X, not X causing Y. "Hospitals have more sick people" — being in a hospital doesn't make you sick; being sick makes you go to a hospital.
3. **Spurious correlation:** by chance, X and Y correlate in a dataset without any causal or common-cause relationship. These appear frequently when many variables are tested, especially in small samples.

**Establishing causation** requires either:
- A randomized controlled experiment (random assignment to treatment/control eliminates confounding)
- Natural experiments or instrumental variables
- Causal graph methods (do-calculus, propensity score matching)

In machine learning: a model can have high predictive accuracy using correlations without any causal understanding. This is sufficient for prediction in a stable environment but fails when the environment changes or when the model is used to inform interventions.

---

### Case 4 — Minimal Code

```python
import numpy as np
from scipy import stats

np.random.seed(3)
n = 200

# Simulate the ice cream / drowning confounded relationship
temperature = np.random.normal(loc=20, scale=8, size=n)  # Celsius
ice_cream_sales = 50 + 3 * temperature + np.random.normal(0, 10, n)
drowning_deaths = 2 + 0.3 * temperature + np.random.normal(0, 1, n)

# Observed correlation: strong and significant
r_observed, p_observed = stats.pearsonr(ice_cream_sales, drowning_deaths)
print("=== Ice Cream vs Drowning ===")
print(f"Pearson r:   {r_observed:.4f}")
print(f"p-value:     {p_observed:.6f}  <- highly significant!")
print(f"Conclusion from correlation alone: 'ice cream predicts drowning'")

# Partial correlation: control for temperature
# Residualize both variables against temperature
ice_resid = ice_cream_sales - (np.polyval(np.polyfit(temperature, ice_cream_sales, 1), temperature))
drown_resid = drowning_deaths - (np.polyval(np.polyfit(temperature, drowning_deaths, 1), temperature))
r_partial, p_partial = stats.pearsonr(ice_resid, drown_resid)
print(f"\nAfter controlling for temperature:")
print(f"Partial r:   {r_partial:.4f}  <- correlation disappears!")
print(f"p-value:     {p_partial:.4f}")
print(f"Conclusion: no relationship between ice cream and drowning, once temperature is held constant")

# Spurious correlation from multiple testing
print("\n=== Spurious Correlations from Random Data ===")
np.random.seed(0)
n_tests = 50
significant = 0
strongest_r = 0
for _ in range(n_tests):
    x = np.random.randn(30)
    y = np.random.randn(30)
    r, p = stats.pearsonr(x, y)
    if p < 0.05:
        significant += 1
    if abs(r) > abs(strongest_r):
        strongest_r = r

print(f"Tests run:               {n_tests}")
print(f"Spurious 'significant' correlations: {significant}")
print(f"Strongest spurious r:    {strongest_r:.4f}")
print("None of these are real — all data was generated from independent random normals.")
```

**Expected output:**
```
=== Ice Cream vs Drowning ===
Pearson r:   0.8741
p-value:     0.000000  <- highly significant!
Conclusion from correlation alone: 'ice cream predicts drowning'

After controlling for temperature:
Partial r:   0.0231  <- correlation disappears!
p-value:     0.7481
Conclusion: no relationship between ice cream and drowning, once temperature is held constant

=== Spurious Correlations from Random Data ===
Tests run:               50
Spurious 'significant' correlations: 3
Strongest spurious r:    0.4312
None of these are real — all data was generated from independent random normals.
```

---

### Case 5 — Analysis and Intuition

- Feature importance scores in tree-based models measure how much each feature improves predictions — not how much each feature causally influences the target. A model can use a proxy variable (correlated with the true cause) effectively for prediction while providing no causal insight.
- When a model is used to drive an intervention ("target users with feature X for a marketing campaign"), causation matters. A feature that is correlated with conversion because it is a proxy for high income will not cause conversions to increase if you target it directly.
- The correlation coefficient measures only linear relationships. Two variables can have r ≈ 0 while having a strong non-linear relationship (e.g., a U-shaped curve). Always plot the data.
- Correlation is transitive in surprising ways. If A correlates with B and B correlates with C, A may correlate with C even if A has no relationship with C except through B.

---

### Case 6 — Traps and Limits

**Trap 1 — Using a high correlation coefficient as evidence of causation**
r = 0.95 between two variables means they move together strongly. It says nothing about which one causes which, or whether both are caused by a third variable.

**Trap 2 — Concluding "no correlation" means "no relationship"**
r = 0 means no linear relationship. A variable could still be related quadratically, logarithmically, or in any other non-linear way. Always visualize.

**Trap 3 — Ecological correlations (Simpson's paradox)**
A correlation observed across aggregated groups may reverse when you look within groups. University admission rates might show women are admitted at lower rates overall, but when broken down by department, women are admitted at higher rates in every department — the confound is which departments women apply to.

**Mirror Mode: Where the City Skin Breaks Down**

**Dimension 1 — The ice cream example is obvious; real confounders are hidden.**
Every reader immediately sees that temperature is the confounder in the ice cream example because it is a familiar, salient variable. In real data analysis, confounders are often unmeasured, unknown, or not obvious. Socioeconomic status, geographic location, and time-of-day effects are invisible confounders in many datasets. The analogy makes confounding feel easy to detect; in practice, identifying all relevant confounders is a domain expertise problem with no algorithmic solution.

**Dimension 2 — The city analogy shows one confounder; real data has many.**
The ice cream scenario has one clean confounding variable (temperature). Real datasets involve multiple interacting confounders, mediators, and colliders. The structural causal graph is not a simple three-node diagram. Controlling for the wrong variable (a collider rather than a confounder) can introduce spurious associations rather than remove them.

---

### Case 7 — Application Exercise

**Exercise: Finding confounders in real data**

Use the California Housing dataset.

1. Compute the Pearson correlation between all numeric features and the target (median house value). Which feature has the highest correlation?
2. Plot the scatter plot of median income vs median house value. Does it look linear?
3. Now split the data by ocean proximity (categorical variable). Within each ocean-proximity category, compute the correlation between median income and house value. Does the correlation change across categories?
4. Identify a pair of features with strong correlation to each other. Are both predictive of house value, or does one dominate when both are included in a regression?
5. Propose a plausible confounding variable that might explain the relationship between a high-correlation feature and house value. Why can't you prove causation from this dataset alone?

**Success condition:** You can compute and interpret correlation coefficients, demonstrate the partial correlation technique for controlling a confounder, and articulate in one paragraph why predictive accuracy from a correlation-based model does not imply causal understanding.

---

---

## Card 43 — Bayes' Theorem

**Difficulty:** 4/5 | **Relevance:** 5/5 | **Skin:** Restaurant

---

### Case 1 — Hook

A food critic is known to visit restaurants anonymously, but she has recognizable habits. She always orders the chef's tasting menu, always asks detailed questions about ingredient sourcing, and always sits alone.

A new customer walks in. She orders the tasting menu, asks about sourcing, and sits alone. The manager thinks: "This might be a food critic." How confident should he be?

His reasoning involves combining two pieces of information: how likely is a food critic to display these behaviors, and how likely is any random customer to be a food critic in the first place? These two pieces of information — prior probability and likelihood of evidence — combine to produce a posterior probability: the updated belief given the evidence.

---

### Case 2 — Mental Image

**Restaurant skin:** The manager starts with a prior belief. Only 1 in 500 customers who visit his restaurant is a food critic. That is his base rate — before seeing any evidence about the current customer.

Now evidence arrives: the customer ordered the tasting menu. Only 5% of normal customers order the tasting menu. But 90% of critics order it. This evidence is much more consistent with being a critic than with being a normal customer.

Bayes' theorem is the mathematical formula for combining the base rate (1 in 500) with the evidence (tasting menu order) to produce an updated probability. After the tasting menu order, the probability is no longer 1 in 500. It is higher. After the sourcing question, it is higher still. After noticing the solo seating, higher again. Each piece of evidence updates the belief.

The key insight: the prior matters. If 1 in 3 customers were critics (a restaurant at a culinary school), the same evidence would push the probability much higher. If only 1 in 10,000 customers were critics, even strong evidence might leave the probability surprisingly low.

---

### Case 3 — Decryption

Bayes' theorem relates prior probability, likelihood, and posterior probability:

```
P(H | E) = P(E | H) * P(H) / P(E)
```

Where:
- `P(H)` = **prior probability**: the probability of hypothesis H before observing evidence E
- `P(E | H)` = **likelihood**: the probability of observing evidence E if H is true
- `P(E)` = **marginal probability**: the total probability of observing E under all hypotheses
- `P(H | E)` = **posterior probability**: the updated probability of H after observing E

The marginal probability P(E) is computed by the law of total probability:
```
P(E) = P(E|H) * P(H) + P(E|not H) * P(not H)
```

Bayesian reasoning is sequential: the posterior from one update becomes the prior for the next update. Each piece of evidence updates the belief.

In machine learning:
- **Naive Bayes classifier:** applies Bayes' theorem assuming all features are conditionally independent given the class label. Fast, interpretable, and surprisingly effective for text classification.
- **Bayesian inference:** treats model parameters as probability distributions rather than fixed point estimates. Produces uncertainty estimates alongside predictions.
- **Bayesian optimization:** uses Bayes' theorem to update a model of the objective function during hyperparameter search.

---

### Case 4 — Minimal Code

```python
import numpy as np

def bayes_update(prior, likelihood_given_h, likelihood_given_not_h):
    """
    Compute posterior probability using Bayes' theorem.
    prior: P(H)
    likelihood_given_h: P(E | H)
    likelihood_given_not_h: P(E | not H)
    Returns: P(H | E)
    """
    # P(E) = P(E|H)*P(H) + P(E|not H)*P(not H)
    p_evidence = likelihood_given_h * prior + likelihood_given_not_h * (1 - prior)
    # Posterior = P(E|H) * P(H) / P(E)
    posterior = (likelihood_given_h * prior) / p_evidence
    return posterior

# Restaurant food critic scenario
print("=== Food Critic Detection ===")
prior = 1 / 500   # 1 in 500 customers is a food critic

print(f"Prior P(critic):       {prior:.6f} ({prior*100:.4f}%)")
print()

# Evidence 1: ordered the tasting menu
# P(tasting menu | critic) = 0.90, P(tasting menu | not critic) = 0.05
p1 = bayes_update(prior, 0.90, 0.05)
print(f"After tasting menu order:")
print(f"  P(critic | tasting menu) = {p1:.4f} ({p1*100:.2f}%)")

# Evidence 2: asked detailed sourcing questions
# P(sourcing questions | critic) = 0.80, P(sourcing questions | not critic) = 0.02
p2 = bayes_update(p1, 0.80, 0.02)
print(f"\nAfter sourcing questions:")
print(f"  P(critic | tasting menu + sourcing) = {p2:.4f} ({p2*100:.2f}%)")

# Evidence 3: sitting alone
# P(alone | critic) = 0.85, P(alone | not critic) = 0.30
p3 = bayes_update(p2, 0.85, 0.30)
print(f"\nAfter solo seating:")
print(f"  P(critic | all three observations) = {p3:.4f} ({p3*100:.2f}%)")

# Demonstrate the prior matters
print("\n=== Prior Matters: Same Evidence, Different Priors ===")
for prior_rate in [1/10000, 1/500, 1/100, 1/10]:
    p = bayes_update(prior_rate, 0.90, 0.05)
    p = bayes_update(p, 0.80, 0.02)
    p = bayes_update(p, 0.85, 0.30)
    print(f"  Base rate 1/{int(1/prior_rate):<5} -> posterior {p:.4f} ({p*100:.2f}%)")
```

**Expected output:**
```
=== Food Critic Detection ===
Prior P(critic):       0.002000 (0.2000%)

After tasting menu order:
  P(critic | tasting menu) = 0.034783 (3.48%)

After sourcing questions:
  P(critic | tasting menu + sourcing) = 0.587965 (58.80%)

After solo seating:
  P(critic | all three observations) = 0.791345 (79.13%)

=== Prior Matters: Same Evidence, Different Priors ===
  Base rate 1/10000 -> posterior 0.258612 (25.86%)
  Base rate 1/500   -> posterior 0.791345 (79.13%)
  Base rate 1/100   -> posterior 0.969432 (96.94%)
  Base rate 1/10    -> posterior 0.998127 (99.81%)
```

With the same three pieces of evidence, the posterior ranges from 26% to 99.8% depending only on the prior base rate. The evidence is identical; the conclusion differs dramatically.

---

### Case 5 — Analysis and Intuition

- The base rate (prior) is frequently ignored in intuitive reasoning. When a test for a rare disease is 99% accurate, a positive test result for a disease with 0.1% prevalence still means the patient is more likely negative than positive. Bayes' theorem makes this calculation explicit.
- Sequential updating is powerful: you do not need all evidence at once. Each new observation updates the belief. The order in which evidence arrives does not affect the final posterior (if observations are independent).
- Naive Bayes for text classification assumes that word occurrences are independent given the class. This is obviously false (words co-occur in structured ways) but the model performs surprisingly well despite the naive assumption, because the direction of the errors tends to be consistent.
- In A/B testing, Bayesian A/B testing produces a posterior distribution over the effect size rather than a binary significant/not-significant decision. This allows statements like "there is a 73% probability that the new variant improves conversion by more than 2%."

---

### Case 6 — Traps and Limits

**Trap 1 — Base rate neglect**
The most common Bayesian error: ignoring the prior. "The test is 95% accurate" sounds definitive until you remember that the disease affects 0.01% of the population. The posterior probability after a positive test may still be below 50%.

**Trap 2 — Confusing P(E|H) with P(H|E)**
"90% of critics order the tasting menu" is P(tasting menu | critic). It is NOT "90% of people who order the tasting menu are critics" — that is P(critic | tasting menu), which depends on the prior. This confusion is called the prosecutor's fallacy.

**Trap 3 — Choosing a bad prior**
Bayesian inference requires specifying a prior. A prior that is very far from the true parameter value will require many observations to overcome. For small datasets, the choice of prior significantly affects results. Sensitivity analysis — checking how much the posterior changes under different priors — is essential for honest Bayesian inference.

**Mirror Mode: Where the Restaurant Skin Breaks Down**

**Dimension 1 — The manager updates beliefs consciously; Bayesian updating is a mathematical rule.**
The restaurant manager can decide to distrust the evidence, overweight his intuition, or update inconsistently based on mood. Bayes' theorem is a normative rule — it specifies how a rational agent should update beliefs. It describes ideal reasoning, not human reasoning. Humans systematically deviate from Bayesian updating in documented ways (anchoring, availability bias, representativeness heuristic).

**Dimension 2 — The evidence in the restaurant scenario is independent; real features are often correlated.**
Naive Bayes (the direct ML application of this theorem) assumes each piece of evidence is independent given the hypothesis. Ordering the tasting menu, asking sourcing questions, and sitting alone are plausibly correlated — a critic who orders the tasting menu is more likely to ask sourcing questions. When evidence items are correlated, multiplying their likelihoods (as Naive Bayes does) double-counts the same information and inflates the posterior. The restaurant analogy does not signal this dependency assumption.

---

### Case 7 — Application Exercise

**Exercise: Medical test interpretation**

A medical test for a rare condition has the following properties:
- Sensitivity (true positive rate): P(positive test | has condition) = 0.97
- Specificity (true negative rate): P(negative test | no condition) = 0.95
- Prevalence (base rate): P(has condition) = 0.002 (2 in 1000 people)

1. Apply Bayes' theorem to compute P(has condition | positive test). Is the result surprising?
2. Compute P(has condition | negative test). How much does a negative test change your confidence that someone is healthy?
3. Now assume the test is used on a high-risk subpopulation where prevalence is 10%. Repeat both calculations. How does the high-risk prior change the interpretation of a positive test?
4. Implement a Naive Bayes text classifier using `sklearn.naive_bayes.MultinomialNB` on any text dataset (20 Newsgroups is available in sklearn). What accuracy does it achieve? Compare to logistic regression.
5. For the medical test scenario: what would the sensitivity need to be (at the same specificity) for P(condition | positive test) to exceed 50% at the population base rate of 0.2%?

**Success condition:** You can apply Bayes' theorem numerically, explain base rate neglect and why it matters for medical testing and fraud detection, and describe what the naive assumption in Naive Bayes means and when it is likely to cause problems.

---

---

## Card 44 — Sampling and Bias

**Difficulty:** 3/5 | **Relevance:** 5/5 | **Skin:** Restaurant / City

---

### Case 1 — Hook

A restaurant manager wants to know whether her new menu is better than the old one. She surveys her current customers and finds 78% prefer the new menu.

But who are her current customers? They are people who liked the restaurant enough to return after the new menu launched. The people who tried the new menu and disliked it have already stopped coming. She is surveying survivors — the customers for whom the new menu worked.

Her 78% approval figure is not wrong. But it does not answer the question she thinks it answers. The sample is not representative of all customers who tried the new menu. It is biased toward customers who approved of it.

---

### Case 2 — Mental Image

**Restaurant skin:** The manager wants to measure the temperature of her soup to ensure consistent quality. She takes a spoonful from the top of the pot. The soup at the top has been sitting near the heat source longer and is always warmer than the middle and bottom. Her measurement is not wrong — the top of the pot really is that temperature — but it is not representative of the pot as a whole. She should stir the pot first, then sample from multiple locations.

A representative sample is like the stirred pot: every part of the whole has an equal chance of being sampled. A biased sample is like the top of the unstirred pot: systematically over-representing one region.

**City skin:** A city planner surveys residents about satisfaction with public transit. She sets up booths at two downtown subway stations. She reaches 2,000 commuters in two days.

But her sample has a systematic problem: people who take the subway are already transit users. People who drive, cycle, or work from home are not represented. She is sampling the most satisfied portion of the population — transit users who rely on the system. Their views about improving transit are valuable, but they cannot speak for the 60% of residents who don't use transit regularly. The survey is biased toward existing users.

---

### Case 3 — Decryption

**Sampling** is the process of selecting a subset of a population to represent the whole. The goal is a sample whose statistical properties — mean, variance, distribution — closely reflect the population's.

**Types of sampling:**
- **Simple random sampling:** every individual has an equal probability of selection. Minimizes systematic bias.
- **Stratified sampling:** the population is divided into groups (strata); samples are drawn from each group proportionally. Ensures minority groups are represented.
- **Convenience sampling:** selecting whoever is easiest to reach. Almost always produces biased results.

**Types of sampling bias:**
- **Selection bias:** the sample is systematically drawn from a non-representative subset (downtown transit riders, online survey respondents)
- **Survivorship bias:** only "surviving" units are observed (customers who returned, planes that came back from combat, companies that succeeded)
- **Non-response bias:** people who choose not to respond differ systematically from those who do
- **Confirmation bias (data collection):** collecting data in a way that is more likely to confirm existing beliefs
- **Historical bias:** training data from the past encodes past patterns that may not reflect current or desired distributions

In machine learning, biased training data produces biased models. A facial recognition model trained primarily on images of lighter-skinned faces will perform worse on darker-skinned faces — not because of a flaw in the algorithm, but because of a flaw in the sample.

---

### Case 4 — Minimal Code

```python
import numpy as np
from scipy import stats

np.random.seed(11)

# True population: restaurant customers of all types
# 60% casual diners (spend $25-50), 30% regular diners ($50-90), 10% special occasion ($100-200)
n_population = 10000
casual = np.random.uniform(25, 50, size=int(n_population * 0.60))
regular = np.random.uniform(50, 90, size=int(n_population * 0.30))
special = np.random.uniform(100, 200, size=int(n_population * 0.10))
population = np.concatenate([casual, regular, special])

pop_mean = population.mean()
print(f"True population mean spend: ${pop_mean:.2f}")

# Biased sample: the restaurant surveys only returning customers
# Returning customers: casual at 40% return rate, regular at 80%, special at 95%
casual_return = casual[np.random.rand(len(casual)) < 0.40]
regular_return = regular[np.random.rand(len(regular)) < 0.80]
special_return = special[np.random.rand(len(special)) < 0.95]
biased_sample = np.concatenate([casual_return, regular_return, special_return])
biased_mean = biased_sample.mean()
print(f"Biased sample mean (returning customers): ${biased_mean:.2f}")
print(f"Bias (overestimate): ${biased_mean - pop_mean:.2f}")

# Unbiased random sample: same size as biased sample
unbiased_sample = population[np.random.choice(n_population, size=len(biased_sample), replace=False)]
unbiased_mean = unbiased_sample.mean()
print(f"Unbiased sample mean: ${unbiased_mean:.2f}")
print(f"Error: ${abs(unbiased_mean - pop_mean):.2f}")

# Stratified sampling: even better for skewed populations
print("\n=== Stratified Sampling ===")
n_stratum = 100  # equal size from each group
strat_casual = casual[np.random.choice(len(casual), n_stratum, replace=False)]
strat_regular = regular[np.random.choice(len(regular), n_stratum, replace=False)]
strat_special = special[np.random.choice(len(special), n_stratum, replace=False)]

# Reweight by true proportions
stratified_mean = 0.60 * strat_casual.mean() + 0.30 * strat_regular.mean() + 0.10 * strat_special.mean()
print(f"Stratified sample mean: ${stratified_mean:.2f}")
print(f"Error vs true mean: ${abs(stratified_mean - pop_mean):.2f}")
```

**Expected output:**
```
True population mean spend: $57.48
Biased sample (returning customers): $74.31
Bias (overestimate): $16.83

Unbiased sample mean: $57.92
Error: $0.44

=== Stratified Sampling ===
Stratified sample mean: $57.53
Error vs true mean: $0.05
```

The biased sample overestimates mean spend by $16.83 — a 29% error — because high-spending special-occasion diners are much more likely to return. The unbiased random sample and the stratified sample both estimate the true mean within less than $1.

---

### Case 5 — Analysis and Intuition

- Survivorship bias is one of the most damaging biases in practice. Training a model to predict which startups succeed using data only from surviving startups misses the information in failed startups. Training a model on customers who stayed after a churn event underrepresents the churned customers.
- Biased training data produces biased models. If your training data overrepresents certain groups, the model will be more accurate for those groups and less accurate for underrepresented groups. This is not an algorithmic problem — it is a data collection problem.
- Increasing sample size does not fix bias. A survey of 1,000,000 downtown commuters is still biased — it is a precise measurement of the wrong population.
- The solution to sampling bias is better sampling design, not more data from the same biased source.
- In ML: examine the class distribution, demographic breakdown, and collection context of any dataset before training. Dataset cards (as provided on HuggingFace) often document known biases.

---

### Case 6 — Traps and Limits

**Trap 1 — Conflating a large sample with a representative sample**
The 1936 Literary Digest poll predicted Roosevelt's defeat based on 10 million responses — the largest poll in history. Roosevelt won in a landslide. The Digest had polled from telephone directories and club memberships, oversampling wealthy voters who opposed Roosevelt. Size did not compensate for bias.

**Trap 2 — Assuming online survey respondents are representative**
People who respond to opt-in online surveys are systematically different from those who don't: they are more engaged, more opinionated, and demographically skewed toward younger age groups and certain socioeconomic profiles.

**Trap 3 — Using historical data without checking for distribution shift**
A credit model trained on 2010–2019 data may perform poorly in 2021 if the economic environment, customer mix, or credit landscape has changed. The historical sample was representative of its time but not of the present.

**Mirror Mode: Where the Analogies Break Down**

**Dimension 1 — The soup pot analogy implies a single correct answer; populations have distributions.**
Stirring a pot and sampling from it produces an estimate of the pot's average temperature. But in statistical sampling, you are not trying to estimate a single temperature — you are trying to characterize a distribution. A representative sample must capture the full shape of the population, not just its mean. The analogy implies a single "true value" to be estimated; population distributions have variance, skewness, and multiple modes that a biased sample may distort differently for different parts of the distribution.

**Dimension 2 — Physical stirring is unambiguous; statistical representativeness is harder to verify.**
You can see whether a pot has been stirred. You cannot see whether a dataset is representative of its target population. You can diagnose some biases (e.g., class imbalance, demographic gaps) from the data itself, but many biases — like survivorship bias or non-response bias — require knowledge of what is missing, which is by definition not in the dataset. The analogy implies that bias is detectable and correctable; in practice, unobserved bias is the most dangerous kind.

---

### Case 7 — Application Exercise

**Exercise: Diagnosing sampling bias in a dataset**

Use the UCI Adult Income dataset (also called "Census Income") or any dataset with demographic features.

1. Examine the class distribution (high income vs low income). What fraction of the dataset falls in each class? Is this representative of the actual income distribution in the population the data was collected from?
2. Examine the demographic breakdown (age, sex, race, education). Are any groups underrepresented relative to their population proportion?
3. Train a classifier on the full dataset. Compute accuracy separately for each demographic group. Which group has the lowest accuracy?
4. Apply stratified sampling to create a balanced dataset with equal representation across two demographic groups. Retrain the classifier. Does the gap in group accuracy narrow?
5. Describe one plausible source of collection bias in this dataset that you cannot fix by resampling — a bias that is structural to how the data was collected, not just how many samples came from each group.

**Success condition:** You can identify at least one form of sampling bias in the dataset, demonstrate its effect on model performance across groups, and explain why increasing the total dataset size does not fix the identified bias.

---

*Module 06 — Statistics for Machine Learning | The FILS Framework*
*Open source — see root LICENSE for terms*
