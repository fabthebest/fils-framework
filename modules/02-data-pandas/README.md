# Module 2: Data and Pandas

**The FILS Framework** | Open-Source AI/ML for Complete Beginners

---

## Module Overview

Before you can train a model, you need to handle data. This module teaches you the tools that every data scientist uses daily: DataFrames for structured data, filtering and selection for extracting what you need, groupby operations for summarizing, merges for combining sources, cleaning for fixing messy reality, and visualization for communicating results.

Concepts 8 through 13. Six cards. One library: **pandas**.

---

## Concept 8: DataFrames

**Difficulty:** 2/5 | **Impact:** 5/5

---

### 1. Hook

You have 10,000 rows of customer data. You want to answer the question: "Which city has the most repeat buyers?" Without a DataFrame, that's a week of work. With one, it is three lines of code. Why does one data structure change everything?

---

### 2. Mental Image

**Restaurant skin:** The reservation book. Every row is one reservation. Every column is one piece of information: guest name, date, party size, table number, special requests. You can flip to any row instantly. You can scan the entire "party size" column without reading anything else. The book is organized so any question — "who is sitting at table 7?" or "how many parties of 4 do we have tonight?" — can be answered fast.

**City skin:** The municipal census database. Every citizen is a row. Every column is an attribute: name, address, age, occupation, tax bracket. The city can ask "how many residents are over 65?" and get an answer in seconds because the data is structured consistently. One column, one type of information, applied uniformly across every single row.

---

### 3. Decryption

A **DataFrame** is a two-dimensional, labeled data structure. Think of it as a table: rows represent observations (one reservation, one citizen, one sale), and columns represent features (attributes of that observation). Under the hood, each column is a **Series** — a one-dimensional array with labels. The DataFrame is just a collection of Series that share the same index.

Pandas DataFrames are the standard container for tabular data in Python. They support fast operations, handle missing values, and connect directly to every other tool in the data science stack.

---

### 4. Minimal Code

```python
import pandas as pd

data = {
    "guest":      ["Alice", "Bob", "Carol", "Dave"],
    "party_size": [2, 4, 1, 3],
    "table":      [5, 2, 8, 2],
    "vip":        [True, False, False, True],
}

df = pd.DataFrame(data)
print(df)
print("\nShape:", df.shape)
print("Columns:", df.columns.tolist())
```

---

### 5. Minimal Code Output

```
   guest  party_size  table    vip
0  Alice           2      5   True
1    Bob           4      2  False
2  Carol           1      8  False
3   Dave           3      2   True

Shape: (4, 3)
Columns: ['guest', 'party_size', 'table', 'vip']
```

---

### 6. Analysis and Intuition

The power of a DataFrame comes from the index. Every row has a label (0, 1, 2, 3 by default), and every column has a name. This lets you access any cell by coordinates: row label plus column name. Operations automatically align on these labels — when you add two DataFrames together, pandas matches rows by index, not by position. This prevents a whole class of silent bugs.

The shape tells you instantly what you are dealing with: `(4, 3)` means 4 rows, 3 columns. Before doing anything with a new dataset, always check `.shape`, `.dtypes`, and `.head()`.

---

### 7. Traps and Limits

**Trap 1 — Modifying a copy:** When you slice a DataFrame, you sometimes get a view and sometimes a copy. Modifying the result may or may not change the original. Use `.copy()` when you intend to work on a separate object.

**Trap 2 — Mixed types in a column:** If one column contains both integers and strings, pandas infers `object` dtype. Operations on that column will be slow and error-prone. Always check `.dtypes` after loading data.

**Trap 3 — Large memory use:** A DataFrame loads everything into RAM. A 5 GB CSV file will require roughly 5-15 GB of memory to work with comfortably. For files that large, look at chunked reading or tools like Polars and Dask.

**Mirror Mode:** You know what a DataFrame is. Now ask: what is it *not*? It is not a database — there is no query optimizer, no transactions, no disk persistence by default. It is not a NumPy array — operations are slower per element, but far more expressive. Choosing the right tool requires knowing where each one breaks down.

---

### 8. Application

Load the following dictionary into a DataFrame, then print its shape, column names, and the first 3 rows using `.head(3)`.

```python
employees = {
    "name":       ["Jordan", "Priya", "Luc", "Aiko", "Marco"],
    "department": ["Sales", "Engineering", "Sales", "HR", "Engineering"],
    "salary":     [52000, 87000, 49000, 61000, 91000],
    "years":      [3, 7, 1, 4, 9],
}
```

Expected: shape `(5, 4)`, four column names, and the first three employee rows displayed.

---

## Concept 9: Filtering and Selection

**Difficulty:** 2/5 | **Impact:** 5/5

---

### 1. Hook

A dataset has 500,000 rows. You care about 312 of them. How do you get exactly those rows — and only those rows — without looping through everything one by one?

---

### 2. Mental Image

**Restaurant skin:** Filtering is the host scanning the reservation book for all VIP guests tonight and pulling only those cards. Selection is reading just the "guest name" column across all reservations — you ignore party size, table number, everything else. You can filter (which rows) and select (which columns) independently, then combine them to get exactly the slice of information you need.

**Video game skin:** Filtering is applying the search filter in your inventory — show only weapons, level 10 and above. Selection is choosing to display only the item name and damage stat, hiding weight and durability. Two separate decisions: what rows pass the gate, and what columns show up in the result.

---

### 3. Decryption

**Selection** extracts one or more columns: `df["column"]` returns a Series, `df[["col1", "col2"]]` returns a DataFrame.

**Filtering** uses a boolean mask — an array of True/False values the same length as the DataFrame. Rows where the mask is True are kept, rows where it is False are dropped. You create the mask with a comparison: `df["salary"] > 60000` produces that array. Pass it back into the DataFrame to filter.

`.loc[]` selects by label. `.iloc[]` selects by integer position. Use `.loc[]` almost always.

---

### 4. Minimal Code

```python
import pandas as pd

df = pd.DataFrame({
    "name":   ["Alice", "Bob", "Carol", "Dave"],
    "dept":   ["Sales", "Eng", "Sales", "Eng"],
    "salary": [52000, 87000, 49000, 91000],
})

# Select one column
print(df["name"])

# Filter rows where salary > 60000
high_earners = df[df["salary"] > 60000]
print(high_earners)

# Filter AND select: names of high earners only
print(df.loc[df["salary"] > 60000, "name"])
```

---

### 5. Minimal Code Output

```
0    Alice
1      Bob
2    Carol
3     Dave
Name: name, dtype: object

   name dept  salary
1   Bob  Eng   87000
3  Dave  Eng   91000

1     Bob
3    Dave
Name: name, dtype: object
```

---

### 6. Analysis and Intuition

The boolean mask is the key mental model. When you write `df["salary"] > 60000`, pandas evaluates that condition for every row and produces a Series of True/False. You are not filtering yet — you are *describing* which rows match. Wrapping it in `df[...]` applies that description as a filter.

Combine conditions with `&` (and) and `|` (or), always wrapping each condition in parentheses: `df[(df["salary"] > 60000) & (df["dept"] == "Eng")]`. The parentheses are not optional — operator precedence will produce wrong results without them.

---

### 7. Traps and Limits

**Trap 1 — `==` vs `=`:** Filtering uses `==` (comparison). Using `=` inside a filter expression causes a syntax error. This catches beginners constantly.

**Trap 2 — Chained indexing:** `df[df["salary"] > 60000]["name"] = "Updated"` may silently fail to update the original DataFrame. Always use `.loc[]` for assignments: `df.loc[df["salary"] > 60000, "name"] = "Updated"`.

**Trap 3 — `and`/`or` vs `&`/`|`:** Python's `and` and `or` do not work element-wise on Series. You will get a cryptic error. Use `&` and `|` for pandas boolean operations.

**Mirror Mode:** Filtering keeps rows. What removes columns? `df.drop(columns=["col"])`. What keeps only specific columns? `df[["col1", "col2"]]`. Filtering and column selection are symmetric operations — one slices horizontally, the other vertically.

---

### 8. Application

Using the employees DataFrame from Concept 8, write three separate queries:

1. Select only the "name" and "salary" columns.
2. Filter rows where salary is above 60,000.
3. Get the names of employees in the "Engineering" department with more than 5 years of experience.

---

## Concept 10: GroupBy and Aggregation

**Difficulty:** 3/5 | **Impact:** 5/5

---

### 1. Hook

You have 10,000 sales records. Your manager asks: "What is the average sale amount by region, and how many transactions did each region have?" That is two numbers per region. GroupBy produces the answer in one line. What is actually happening under the hood?

---

### 2. Mental Image

**Restaurant skin:** GroupBy is the manager sorting all the night's order slips by table number — table 2's orders in one pile, table 5's in another, table 8's in a third. Aggregation is then calculating each pile's total: table 2 spent $120, table 5 spent $85, table 8 spent $47. You went from 60 individual order slips to a 3-row summary. That reduction is the entire point.

**Orchestra skin:** GroupBy is the conductor separating the orchestra by section — strings here, woodwinds there, brass in the back. Aggregation is measuring the average age of each section, or the total years of experience per section. The individual musicians become section statistics.

---

### 3. Decryption

GroupBy follows a **split-apply-combine** pattern:

1. **Split** the DataFrame into groups based on the values of one or more columns.
2. **Apply** an aggregation function (sum, mean, count, min, max) to each group independently.
3. **Combine** the results back into a new DataFrame, one row per group.

`df.groupby("column")` returns a GroupBy object — nothing has been computed yet. The aggregation call (`.sum()`, `.mean()`, `.agg({...})`) triggers the actual computation.

---

### 4. Minimal Code

```python
import pandas as pd

df = pd.DataFrame({
    "table":  [2, 5, 2, 8, 5, 2],
    "item":   ["wine", "steak", "salad", "pasta", "wine", "steak"],
    "amount": [35, 55, 15, 28, 30, 48],
})

summary = df.groupby("table")["amount"].agg(["sum", "mean", "count"])
print(summary)
```

---

### 5. Minimal Code Output

```
        sum       mean  count
table
2       98  32.666667      3
5       85  42.500000      2
8       28  28.000000      1
```

---

### 6. Analysis and Intuition

Notice what happened: 6 rows became 3 rows. Each unique value of "table" produced exactly one output row. The column you grouped by becomes the index of the result.

You can group by multiple columns: `df.groupby(["dept", "region"])` creates groups for every unique combination of department and region. You can apply different aggregations to different columns using a dictionary: `df.groupby("dept").agg({"salary": "mean", "years": "max"})`.

GroupBy is the foundation of almost every business metric: revenue per category, average order value per customer segment, error rate per server. Learn this deeply.

---

### 7. Traps and Limits

**Trap 1 — Forgetting to aggregate:** `df.groupby("column")` alone returns a GroupBy object, not a DataFrame. You must follow it with an aggregation. Printing a GroupBy object gives you its memory address, which confuses beginners.

**Trap 2 — The index shift:** After groupby, the grouped column becomes the index. If you want it back as a regular column, use `.reset_index()`.

**Trap 3 — `count` vs `size`:** `.count()` counts non-null values per column. `.size()` counts all rows including nulls. They give different results on messy data. Know which one you need.

**Mirror Mode:** GroupBy reduces rows. What expands rows? `.explode()` — for columns containing lists. What expands columns? `.pivot_table()` — which is essentially a grouped aggregation reshaped into a matrix. GroupBy, pivot, and explode are three sides of the same reshaping problem.

---

### 8. Application

Using the employees DataFrame (name, department, salary, years), compute the following grouped summary by department:

- Average salary
- Maximum years of experience
- Number of employees

Then reset the index so "department" appears as a regular column in the output.

---

## Concept 11: Merge and Join

**Difficulty:** 3/5 | **Impact:** 4/5

---

### 1. Hook

Your orders table has 50,000 rows with a customer ID column. Your customers table has 8,000 rows with customer details. How do you combine them so every order row also shows the customer's name and city? That is a merge — and it is one of the most common operations in data work.

---

### 2. Mental Image

**Restaurant skin:** You have two binders. Binder A is the reservation list: each row has a guest ID, date, and table. Binder B is the loyalty program database: each row has a guest ID, name, email, and lifetime spend. Merging them means lining up every reservation with its matching loyalty record using the guest ID as the link. The result is a single combined row with all the information from both binders.

**Construction skin:** Two sets of blueprints for the same building. One set shows structural elements, one set shows electrical layout. Merging them means overlaying both onto a single master plan, matching by room number. Rooms that exist in both plans appear fully detailed. Rooms that exist in only one plan depend on the join type: do you keep them (outer join) or drop them (inner join)?

---

### 3. Decryption

`pd.merge(left, right, on="key")` combines two DataFrames on a shared column.

Join types:
- **inner** (default): keep only rows where the key exists in both DataFrames.
- **left**: keep all rows from the left DataFrame; fill missing right-side values with NaN.
- **right**: keep all rows from the right DataFrame.
- **outer**: keep all rows from both; fill gaps with NaN.

Use `left_on` and `right_on` when the key columns have different names in each DataFrame.

---

### 4. Minimal Code

```python
import pandas as pd

reservations = pd.DataFrame({
    "guest_id": [101, 102, 103, 104],
    "table":    [5, 2, 8, 2],
})

loyalty = pd.DataFrame({
    "guest_id": [101, 102, 105],
    "name":     ["Alice", "Bob", "Eve"],
    "tier":     ["Gold", "Silver", "Gold"],
})

result = pd.merge(reservations, loyalty, on="guest_id", how="left")
print(result)
```

---

### 5. Minimal Code Output

```
   guest_id  table   name    tier
0       101      5  Alice    Gold
1       102      2    Bob  Silver
2       103      8    NaN     NaN
3       104      2    NaN     NaN
```

---

### 6. Analysis and Intuition

Guest 103 and 104 have reservations but are not in the loyalty database. A left join keeps them in the result with NaN for the loyalty columns. An inner join would have dropped them entirely.

The choice of join type is a business decision, not a technical one. Ask: "Do I want to keep rows that have no match?" If yes, use left or outer. If a non-match means the row is invalid and should be excluded, use inner.

Watch for duplicate keys: if one table has the same guest_id on multiple rows, the merge will create a row for every combination, which can silently multiply your row count. Always check the shape before and after a merge.

---

### 7. Traps and Limits

**Trap 1 — Silent row multiplication:** If the key column has duplicates in both DataFrames, a merge produces a Cartesian product of matching rows. 3 rows in left matching 3 rows in right on the same key produces 9 rows. Always verify with `.shape`.

**Trap 2 — Suffix confusion:** When both DataFrames have a column with the same name (other than the key), pandas appends `_x` and `_y` suffixes. Name your columns clearly before merging, or use the `suffixes` parameter.

**Trap 3 — Wrong join type by default:** The default is inner join. If you expect to keep all rows from one side and silently lose some, you will get a smaller DataFrame than expected with no error message. Be explicit about `how=`.

**Mirror Mode:** Merge combines data horizontally — adding columns. What adds data vertically? `pd.concat([df1, df2])` — stacking rows. These are complementary operations. If your data is split across files with the same columns, concat. If it is split across tables with the same rows, merge.

---

### 8. Application

Create two DataFrames:

- `orders`: columns `order_id`, `customer_id`, `amount`
- `customers`: columns `customer_id`, `city`

Merge them so every order shows its customer's city. Then try both `how="inner"` and `how="left"` with at least one customer_id that appears in orders but not in customers. Observe the difference in the output shape.

---

## Concept 12: Data Cleaning

**Difficulty:** 3/5 | **Impact:** 5/5

---

### 1. Hook

Industry surveys consistently report that data scientists spend 60 to 80 percent of their time cleaning data. Not building models. Not tuning hyperparameters. Cleaning. Why does dirty data ruin a model that looks mathematically correct?

---

### 2. Mental Image

**Restaurant skin:** Kitchen prep. Before a chef cooks anything, the ingredients must be washed, peeled, trimmed, and portioned. A recipe that calls for "2 cups of diced carrot" assumes clean, uniform cubes — not one chunk, one half-chunk, and a piece with the dirt still on it. Dirty data fed into a model is exactly that: unpeeled carrots in a dish that required precise cuts. The algorithm runs but the output is wrong.

**Human body skin:** The immune system encountering pathogens. Raw data entering a pipeline is full of threats — null values that will crash an aggregation, duplicate records that will double-count revenue, typos like "New Yrok" and "new york" that appear as separate cities. Data cleaning is the immune response: identify each threat, neutralize it specifically, and let only clean signal through.

---

### 3. Decryption

The four main cleaning operations:

1. **Null handling:** `df.isnull().sum()` to find them. `.dropna()` to remove rows with nulls. `.fillna(value)` to replace them.
2. **Duplicate removal:** `df.duplicated().sum()` to count them. `df.drop_duplicates()` to remove them.
3. **Type correction:** `df["col"].astype(int)` to convert types. `pd.to_datetime(df["col"])` for dates.
4. **String normalization:** `.str.lower()`, `.str.strip()`, `.str.replace()` to standardize text.

---

### 4. Minimal Code

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "city":   ["Paris", "paris", "PARIS", "Lyon", "Lyon"],
    "sales":  [100, 200, None, 150, 150],
    "code":   ["FR01", "FR01", "FR01", "FR02", "FR02"],
})

print("Before:", df.shape)
print("Nulls:\n", df.isnull().sum())

df["city"] = df["city"].str.lower().str.strip()
df["sales"] = df["sales"].fillna(df["sales"].median())
df = df.drop_duplicates()

print("\nAfter:", df.shape)
print(df)
```

---

### 5. Minimal Code Output

```
Before: (5, 3)
Nulls:
 city     0
sales    1
code     0
dtype: int64

After: (3, 3)
    city  sales  code
0  paris  100.0  FR01
1  paris  200.0  FR01
3   lyon  150.0  FR02
```

---

### 6. Analysis and Intuition

Three separate things happened in that cleaning pass. First, "Paris", "paris", and "PARIS" became the same string — now groupby will correctly treat them as one city. Second, the null sale was filled with the median (150), which is more robust than the mean when outliers are present. Third, true duplicates were removed.

Notice the row count went from 5 to 3. Row 2 (PARIS/None) merged into the Paris group after normalization and null-filling, and then the duplicate Lyon row (index 4) was dropped.

Cleaning decisions have consequences for your analysis. Filling nulls with 0 is very different from filling with the median. Dropping rows with nulls is different from imputing. There is no universal right answer — it depends on why the data is missing.

---

### 7. Traps and Limits

**Trap 1 — Dropping nulls reflexively:** `.dropna()` is tempting because it makes the problem disappear. But if 30 percent of your rows have at least one null, you just threw away a third of your data. Understand *why* values are missing before deciding what to do.

**Trap 2 — Not resetting the index after dropping rows:** After `drop_duplicates()` or `dropna()`, the index has gaps (0, 1, 3 instead of 0, 1, 2). Some operations behave unexpectedly with a non-contiguous index. Use `.reset_index(drop=True)` to clean it up.

**Trap 3 — Cleaning after aggregating:** If you aggregate first and clean second, the aggregation already included the dirty values. Always clean before you analyze.

**Mirror Mode:** You know how to clean a DataFrame. Now think about what cleaning *cannot* fix: if the data was recorded wrong at the source — a sensor that reported Celsius as Fahrenheit for six months, a form that defaulted to 0 instead of leaving blank — no amount of pandas cleaning recovers the true value. Cleaning fixes format and consistency. It cannot fix truth.

---

### 8. Application

Create a DataFrame with deliberate problems: mixed-case city names, a few null values in a numeric column, and at least two fully duplicate rows. Then write a cleaning pipeline that:

1. Normalizes city names to lowercase with no leading/trailing spaces.
2. Fills numeric nulls with the column mean.
3. Drops duplicate rows.
4. Resets the index.

Print the shape before and after each step.

---

## Concept 13: Visualization

**Difficulty:** 2/5 | **Impact:** 4/5

---

### 1. Hook

Two analysts. Both run the same numbers. One shows a table of 200 values. The other shows a single histogram. Who gets the budget approved? Data without visualization is an argument nobody can follow. Why is visual encoding so cognitively powerful?

---

### 2. Mental Image

**Restaurant skin:** Plating. A michelin-starred chef does not dump a finished dish into a bowl and call it done. Presentation is part of the meal — the height, the color contrast, the sauce placement signal quality and intention before the guest takes a bite. Visualization is how you plate data. A number like 87,000 means nothing in isolation. On a chart next to 49,000 and 52,000, the story is immediate.

**City skin:** A city map versus a raw coordinate list. You could describe every building in a city as GPS coordinates — latitude, longitude, height. Or you could draw a map. The map does not add information. It reorganizes the same information into a form the human visual system can process in milliseconds. Visualization is the map for your data.

---

### 3. Decryption

Pandas integrates directly with matplotlib. `df.plot()` produces a line chart. `df.plot(kind="bar")` produces a bar chart. `df["col"].hist()` produces a histogram. `df.plot(kind="scatter", x="col1", y="col2")` produces a scatter plot.

For more control, import matplotlib directly: `import matplotlib.pyplot as plt`. Call `plt.show()` to display, `plt.savefig("name.png")` to save. Seaborn builds on matplotlib and provides cleaner statistical plots with less code.

---

### 4. Minimal Code

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "department": ["Sales", "Engineering", "HR"],
    "avg_salary": [52000, 89000, 61000],
})

df.plot(kind="bar", x="department", y="avg_salary",
        title="Average Salary by Department",
        legend=False, color="steelblue")

plt.ylabel("Salary (USD)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("salary_by_dept.png")
plt.show()
```

---

### 5. Minimal Code Output

```
[A bar chart saved as salary_by_dept.png with three vertical bars:
 Sales at ~52,000, Engineering at ~89,000, HR at ~61,000.
 The Engineering bar is visibly the tallest.
 X-axis labels are horizontal. Y-axis is labeled "Salary (USD)".]
```

---

### 6. Analysis and Intuition

Chart choice is a design decision, not an aesthetic one. Different chart types answer different questions:

- **Bar chart**: compare a value across categories (salary by department).
- **Line chart**: show change over time (monthly revenue).
- **Histogram**: show the distribution of one variable (age of customers).
- **Scatter plot**: show the relationship between two variables (hours studied vs exam score).
- **Box plot**: show distribution and outliers across groups.

Using a line chart for categorical data (connecting unrelated departments with a line) implies a sequence that does not exist. Chart type carries meaning. Choose deliberately.

---

### 7. Traps and Limits

**Trap 1 — Plotting before cleaning:** A histogram of salary data that still contains nulls or string-formatted numbers will error or silently produce a misleading chart. Clean first, plot second.

**Trap 2 — Overloading a single chart:** Plotting 20 categories on a bar chart makes every bar unreadable. If your groupby produced 20 groups, consider showing only the top 10, or using a different chart type. Visualization is about reducing cognitive load, not displaying everything.

**Trap 3 — Default scales mislead:** A y-axis that starts at 50,000 instead of 0 can make a 5 percent difference look like a 500 percent difference. Always check axis ranges. Always label axes. Always include a title.

**Mirror Mode:** Visualization is the last step of exploration and the first step of communication. Used for exploration, the goal is to find something surprising in the data — so try many chart types quickly, do not polish. Used for communication, the goal is to make one point undeniable — so choose carefully, remove everything that does not support that point, and label everything. The same data requires a different visualization depending on which mode you are in.

---

### 8. Application

Using the employees DataFrame, create two charts:

1. A bar chart showing average salary by department.
2. A histogram of the salary column.

For each chart: add a title, label both axes, and save it to a file. Print a one-sentence interpretation of what each chart shows.

---

## Module Summary

| Concept | Core Operation | One-Line Takeaway |
|---------|---------------|-------------------|
| 8. DataFrames | `pd.DataFrame()` | The table is the atom of data work |
| 9. Filtering and Selection | `df[mask]`, `.loc[]` | Boolean masks slice rows; column lists slice columns |
| 10. GroupBy and Aggregation | `.groupby().agg()` | Split-apply-combine reduces rows to summaries |
| 11. Merge and Join | `pd.merge()` | Join type is a business decision, not a technical default |
| 12. Data Cleaning | `.fillna()`, `.drop_duplicates()`, `.str.lower()` | Clean before you analyze, always |
| 13. Visualization | `.plot()`, `plt.show()` | Chart type carries meaning; choose deliberately |

---

## What Comes Next

Module 3 introduces machine learning. The skills you built here — filtering, groupby, merge, and cleaning — are the preprocessing layer that every ML pipeline depends on. A model trained on dirty data learns the dirt. Master Module 2 before proceeding.

---

*The FILS Framework | Module 2 of 6*
