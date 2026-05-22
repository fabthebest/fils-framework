# Module 1 — Python Foundations

**The FILS Framework** | Difficulty: 1-3 / 5 | Prerequisites: None

This module builds the seven conceptual pillars every AI/ML practitioner uses every single day. Each concept follows the BRIDGIA 7-case card format: Hook, Mental Image, Decryption, Minimal Code, Output, Analysis & Intuition, Traps & Limits, Application.

---

## Concept 1 — Variables & Types

**Difficulty:** 1/5 | **Impact:** 5/5

---

### Hook

If you had to carry 50 different ingredients across a kitchen without any containers or labels, how many steps would it take before everything ended up on the floor?

---

### Mental Image

**Restaurant skin:** A variable is a labeled container sitting in the kitchen pantry. The label is the name (`flour`). The container holds the actual ingredient. You can swap the contents — empty the container and refill it — but the label stays the same. The *type* of the container determines what you can put inside: a spice jar cannot hold a gallon of stock.

**Construction skin:** On a job site, every storage bin has a label taped to it: `nails_count`, `beam_length`, `site_address`. The type enforces the shape of what fits. You would not store a decimal measurement in a bin labeled for whole-number counts — not because the bin refuses, but because the downstream carpenter would measure wrong.

---

### Decryption

A variable is a named reference to a location in memory. The value at that location has a *type* — an integer, a float, a string, a boolean — and the type determines which operations are legal. Python infers the type from the assigned value (dynamic typing), but the type still exists and governs behavior. Assigning a new value rebinds the name to a new memory location; it does not mutate the old one.

---

### Minimal Code

```python
flour_grams = 250          # int
oil_liters = 0.05          # float
dish_name = "Griot"        # str
is_available = True        # bool

print(type(flour_grams))
print(flour_grams + 50)
print(dish_name.upper())
```

**Output:**

```
<class 'int'>
300
GRIOT
```

---

### Analysis & Intuition

Types are not bureaucracy — they are performance contracts. An integer operation runs in nanoseconds. Accidentally storing a number as a string (e.g., `"250"`) and then trying to do arithmetic on it will raise a `TypeError`. In ML pipelines, type mismatches are among the most common causes of silent data corruption, where no error is raised but the computed result is meaningless.

---

### Traps & Limits

**Trap 1 — Integer division looks surprising:**
```python
print(7 / 2)   # 3.5, not 3
print(7 // 2)  # 3  (floor division)
```

**Output:**

```
3.5
3
```

**Trap 2 — String plus number raises an error:**
```python
age = "25"
print(age + 1)   # TypeError
```

**Mirror Mode (where the analogy breaks):** In a real pantry, changing the label on a container changes what the container is called but the old contents remain physically in place. In Python, rebinding a name (`flour_grams = 300`) does not modify the original integer object — it creates a new integer `300` and points the name at it. The original `250` is eventually garbage-collected. Variables are pointers, not physical bins.

---

### Application

1. Declare variables for a recipe: name (string), prep time in minutes (int), cost per serving (float), is vegan (bool).
2. Print each variable and its type.
3. Convert the prep time to hours (float division). Print the result.

---

---

## Concept 2 — Lists & Dictionaries

**Difficulty:** 2/5 | **Impact:** 5/5

---

### Hook

A grocery list and a recipe book are both about food — but you navigate them completely differently. Why does that distinction matter in code?

---

### Mental Image

**Restaurant skin — Lists:** The daily specials board lists dishes in order: position 0 is the soup, position 1 is the main, position 2 is the dessert. Order matters; you retrieve by index. If the chef adds a new starter, it shifts every other position.

**Restaurant skin — Dictionaries:** The master recipe book is indexed by dish name. You do not flip through page by page — you open directly to "Griot" and read the instructions. Order is irrelevant; what matters is the key. Adding a new recipe does not change how you look up the existing ones.

**City skin:** A list is a numbered city block — house 0, house 1, house 2. A dictionary is the city directory — you search by resident name, not by address number.

---

### Decryption

A list is an ordered, mutable sequence accessed by integer index (0-based). A dictionary is an unordered (insertion-ordered since Python 3.7) mapping from keys to values. Lists are O(n) for search; dictionaries are O(1) average for lookup because they use a hash table internally. For ML, lists hold sequences of samples; dictionaries hold configuration parameters and feature mappings.

---

### Minimal Code

```python
# List: ordered menu
specials = ["Soup Joumou", "Griot", "Douce Macoss"]
print(specials[1])
specials.append("Pen Patat")
print(len(specials))

# Dictionary: recipe metadata
recipe = {"name": "Griot", "prep_minutes": 45, "servings": 4}
print(recipe["prep_minutes"])
recipe["vegan"] = False
print(recipe)
```

**Output:**

```
Griot
4
45
{'name': 'Griot', 'prep_minutes': 45, 'servings': 4, 'vegan': False}
```

---

### Analysis & Intuition

The choice between a list and a dictionary is a question of how you intend to retrieve data. If you will always process items sequentially, use a list. If you need to look something up by a meaningful label, use a dictionary. In ML, a training dataset is usually a list of samples; a model's hyperparameters are almost always a dictionary. Mixing them up creates code that works but runs orders of magnitude slower than necessary at scale.

---

### Traps & Limits

**Trap 1 — Index out of range:**
```python
items = ["a", "b", "c"]
print(items[3])   # IndexError: list index out of range
```

**Trap 2 — Missing key:**
```python
recipe = {"name": "Griot"}
print(recipe["servings"])   # KeyError: 'servings'
print(recipe.get("servings", 0))   # Safe default
```

**Output:**

```
0
```

**Mirror Mode:** The recipe book analogy implies the book is organized alphabetically. Python dictionaries are not sorted — they preserve insertion order but do not sort by key. If you need sorted keys, call `sorted(recipe.keys())` explicitly. Also, real books allow duplicate section headers; dictionary keys must be unique.

---

### Application

1. Create a list of five ingredients for a dish.
2. Create a dictionary describing that dish: name, cook time, number of ingredients.
3. Add a new key `ingredients` to the dictionary whose value is your list.
4. Print the third ingredient using only the dictionary (no direct list reference).

---

---

## Concept 3 — Loops & Conditions

**Difficulty:** 2/5 | **Impact:** 5/5

---

### Hook

What if every task you automated still required you to watch it and manually restart it after each item? That is exactly the world without loops.

---

### Mental Image

**Restaurant skin — Loops:** Prep cook instructions: "Peel and dice every onion in this crate." The crate is the iterable. The instruction set (peel, dice) is the loop body. The cook does not re-read the instructions for each onion — the instruction is applied repeatedly until the crate is empty.

**Restaurant skin — Conditions:** The head chef inspects each plate before it leaves the kitchen. If the portion is under 200g, send it back. If the presentation fails, plate again. Otherwise, approve. The condition is the quality gate — it routes each item to a different outcome based on its current state.

**Human Body skin:** The heart does not decide whether to beat based on a manual command each time. It loops indefinitely, and a conditional check (is blood oxygen low?) determines whether to beat faster or slower.

---

### Decryption

A `for` loop iterates over any iterable object (list, string, range, file, generator) and executes a block for each element. A `while` loop continues as long as a boolean expression remains true. An `if/elif/else` block evaluates conditions sequentially and executes the first branch whose condition is true. In ML, loops power training epochs, batch processing, and hyperparameter sweeps.

---

### Minimal Code

```python
ingredients = ["onion", "garlic", "thyme", "scotch bonnet"]

for item in ingredients:
    if item == "scotch bonnet":
        print(f"Handle with care: {item}")
    else:
        print(f"Chop: {item}")
```

**Output:**

```
Chop: onion
Chop: garlic
Chop: thyme
Handle with care: scotch bonnet
```

---

### Analysis & Intuition

Loops are where bugs multiply. A loop body that looks correct on one item can silently corrupt results when applied to thousands of items with varying shapes or null values. In data pipelines, the idiom `for row in dataset` applied to millions of rows is often replaced by vectorized operations (NumPy, pandas) for performance — but understanding the loop is the prerequisite for understanding why vectorization exists and when it is safe to use.

---

### Traps & Limits

**Trap 1 — Modifying a list while iterating over it:**
```python
items = [1, 2, 3, 4]
for x in items:
    if x == 2:
        items.remove(x)   # skips the item after 2
print(items)
```

**Output:**

```
[1, 3, 4]
```

This looks correct here but fails unpredictably with duplicates. Always iterate over a copy: `for x in items[:]`.

**Trap 2 — Infinite while loop:**
```python
# count = 0
# while count < 5:
#     print(count)   # forgot: count += 1
```

**Mirror Mode:** The prep cook analogy implies every onion is identical. In real data, items are heterogeneous — some rows have null values, some have unexpected types. The loop does not automatically adapt. Defensive coding (checking types inside the loop body) is the responsibility of the programmer, not the loop itself.

---

### Application

1. Create a list of 8 numbers (a mix of positive, negative, and zero).
2. Loop through the list. For each number: print "positive" if above zero, "negative" if below zero, "zero" if exactly zero.
3. Count how many are positive. Print the count after the loop.

---

---

## Concept 4 — Functions

**Difficulty:** 2/5 | **Impact:** 5/5

---

### Hook

If a sous chef had to reinvent the Griot recipe from memory every single service, how long before the dish became inconsistent — or vanished entirely from the menu?

---

### Mental Image

**Restaurant skin:** A recipe is a function. It specifies the ingredients required (parameters), the preparation steps (the function body), and the finished dish (the return value). Once written, the recipe is reused every service. Changing the recipe changes the dish for every future execution. The recipe itself is not the dish — it is the *instructions for producing* the dish.

**Orchestra skin:** A musical score is a function. The conductor calls it (invokes it), the musicians execute it with the instruments provided (arguments), and the result is the performance (return value). The same score can be performed by different orchestras (called with different argument values) and produce different sound textures while following the same structure.

---

### Decryption

A function is a named, reusable block of code that accepts zero or more parameters and optionally returns a value. Functions enforce the DRY principle (Do Not Repeat Yourself) and are the primary unit of reusability in Python. In ML, functions encapsulate preprocessing steps, loss computations, evaluation metrics, and data augmentation pipelines.

---

### Minimal Code

```python
def marinate(protein, duration_hours, spice_level=2):
    flavor = protein + " marinated for " + str(duration_hours) + "h"
    return flavor + " (spice level: " + str(spice_level) + ")"

result1 = marinate("chicken", 4)
result2 = marinate("pork", 12, spice_level=5)
print(result1)
print(result2)
```

**Output:**

```
chicken marinated for 4h (spice level: 2)
pork marinated for 12h (spice level: 5)
```

---

### Analysis & Intuition

Default parameter values (`spice_level=2`) make functions flexible without forcing callers to specify every argument. However, never use a mutable default like a list or dictionary — Python evaluates default values once at function definition time, not at each call, which causes the infamous shared-state bug. Functions also define a local scope: variables created inside a function are not accessible outside it unless explicitly returned.

---

### Traps & Limits

**Trap 1 — Mutable default argument:**
```python
def add_topping(pizza, toppings=[]):
    toppings.append("cheese")
    return toppings

print(add_topping("margherita"))
print(add_topping("quattro stagioni"))   # Unexpected!
```

**Output:**

```
['cheese']
['cheese', 'cheese']
```

Use `toppings=None` and initialize inside the function body instead.

**Trap 2 — Forgetting `return`:**
A function without a `return` statement returns `None`. Assigning the result and using it downstream silently passes `None` through your pipeline.

**Mirror Mode:** A recipe specifies exact ingredient quantities. A function does not enforce parameter types — `marinate(42, "tomorrow")` will not raise an error at definition time. Python 3 supports type hints (`def marinate(protein: str, duration_hours: int)`) but they are documentation, not enforcement, unless a type checker like `mypy` is run separately.

---

### Application

1. Write a function `normalize(value, min_val, max_val)` that returns the value scaled to the range [0, 1] using the formula `(value - min_val) / (max_val - min_val)`.
2. Test it with `normalize(50, 0, 100)` (should return `0.5`) and `normalize(75, 50, 150)` (should return `0.25`).
3. Add a guard: if `max_val == min_val`, return `0.0` instead of dividing by zero.

---

---

## Concept 5 — Classes & Objects

**Difficulty:** 3/5 | **Impact:** 4/5

---

### Hook

How does a franchise like a fast food chain guarantee that every new location operates identically — same menu, same process, same equipment — while each location still has its own address, its own staff count, and its own daily revenue?

---

### Mental Image

**Restaurant skin:** A *class* is the franchise operations manual and architectural blueprint. It specifies what every location must have (attributes: address, seating capacity, daily revenue) and what every location can do (methods: open, take order, close). An *object* is a specific location — Port-au-Prince branch, Montreal branch — instantiated from that blueprint. Each object carries its own data, but shares the same behavioral contract.

**Video Game skin:** A class is the enemy type defined in the game engine: `Dragon`. It specifies health points, attack pattern, and drop rate. Each dragon encountered in the game world is an object — `cave_dragon`, `mountain_dragon` — each with independent health points but identical behavior rules.

---

### Decryption

A class is a template that bundles data (attributes) and behavior (methods) into a single unit. The `__init__` method is the constructor — it runs automatically when a new object is created and initializes its attributes. `self` is the reference to the current instance. Classes enable encapsulation (hiding internal state), inheritance (extending existing behavior), and polymorphism (different classes responding to the same method call differently).

---

### Minimal Code

```python
class Restaurant:
    def __init__(self, name, city, capacity):
        self.name = name
        self.city = city
        self.capacity = capacity
        self.revenue = 0.0

    def record_sale(self, amount):
        self.revenue += amount

    def report(self):
        return f"{self.name} ({self.city}): ${self.revenue:.2f}"

port_au_prince = Restaurant("Lakay", "Port-au-Prince", 40)
montreal = Restaurant("Lakay", "Montreal", 60)

port_au_prince.record_sale(450.00)
montreal.record_sale(780.00)
montreal.record_sale(320.00)

print(port_au_prince.report())
print(montreal.report())
```

**Output:**

```
Lakay (Port-au-Prince): $450.00
Lakay (Montreal): $1100.00
```

---

### Analysis & Intuition

The crucial insight: `montreal.revenue` and `port_au_prince.revenue` are completely independent despite belonging to the same class. This is the point of instantiation. In ML, you instantiate model objects — `model = LinearRegression()` — and each model object holds its own learned parameters. You can train two instances on different data and compare them without one overwriting the other.

---

### Traps & Limits

**Trap 1 — Class-level vs instance-level attributes:**
```python
class Counter:
    count = 0   # class attribute — shared across ALL instances

    def increment(self):
        Counter.count += 1

a = Counter()
b = Counter()
a.increment()
print(b.count)   # 1, not 0
```

**Output:**

```
1
```

To make it instance-level, initialize inside `__init__`: `self.count = 0`.

**Mirror Mode:** The franchise blueprint analogy implies all locations are identical at startup. In Python, you can add attributes to a single instance after creation (`port_au_prince.special_menu = True`) without that attribute appearing on any other instance. The blueprint is a starting point, not a hard constraint on object structure. This flexibility is powerful but makes debugging harder when instances diverge unexpectedly.

---

### Application

1. Define a class `DataPoint` with attributes `feature` (float) and `label` (string).
2. Add a method `describe()` that returns a formatted string: `"Feature: X | Label: Y"`.
3. Create three instances with different values. Store them in a list. Loop through the list and print each description.

---

---

## Concept 6 — File I/O

**Difficulty:** 2/5 | **Impact:** 3/5

---

### Hook

Every model you will ever train depends on data that lives somewhere outside of RAM. If you cannot read from and write to files reliably, you cannot do data science at all.

---

### Mental Image

**Restaurant skin:** The day's ingredient delivery comes in crates from the supplier (reading a file). The end-of-day inventory report is written by hand and filed in the office (writing a file). The context manager (`with` block) is the walk-in cooler door: it ensures the door is properly closed whether the job is done cleanly or interrupted by a problem.

**Construction skin:** Blueprints are read from the architect's binder (reading), and inspection reports are written back into the project folder (writing). The project folder persists after every worker goes home — files outlive the program that created them.

---

### Decryption

The `open()` function returns a file object. Mode `"r"` reads, `"w"` writes (overwrites), `"a"` appends, `"rb"` / `"wb"` handles binary data. The `with` statement (context manager) guarantees the file is closed even if an exception occurs mid-operation, preventing file handle leaks. In ML, you read CSV files, write model checkpoints, and log training metrics to text files constantly.

---

### Minimal Code

```python
# Write a simple CSV
with open("menu.csv", "w") as f:
    f.write("dish,price\n")
    f.write("Griot,18.00\n")
    f.write("Soup Joumou,12.00\n")

# Read it back line by line
with open("menu.csv", "r") as f:
    for line in f:
        print(line.strip())
```

**Output:**

```
dish,price
Griot,18.00
Soup Joumou,12.00
```

---

### Analysis & Intuition

Always use the `with` statement. Always call `.strip()` on lines read from text files — every line carries a newline character `\n` at the end, and failing to strip it is a common source of subtle string comparison bugs. For structured data, prefer the `csv` module or `pandas.read_csv()` over manual line parsing — they handle edge cases (quoted commas, encoding differences) that manual parsing misses.

---

### Traps & Limits

**Trap 1 — Opening with `"w"` destroys existing content:**
```python
# This erases everything already in the file before writing
with open("log.txt", "w") as f:
    f.write("new entry\n")
# Use "a" to append instead
```

**Trap 2 — FileNotFoundError on read:**
```python
with open("nonexistent.csv", "r") as f:   # raises FileNotFoundError
    pass
```

**Mirror Mode:** The walk-in cooler analogy suggests the door can only be open or closed. File objects have additional states — partially written, buffered but not flushed to disk. Python's `with` block flushes and closes on exit, but mid-write crashes before the `with` exits can leave a file truncated. For critical data, write to a temporary file and rename it atomically after completion.

---

### Application

1. Write a function `save_scores(filename, scores)` that takes a filename and a list of floats and writes each score on its own line.
2. Write a function `load_scores(filename)` that reads the file and returns a list of floats.
3. Save `[0.92, 0.87, 0.95, 0.78]`, then load and print the average.

---

---

## Concept 7 — Error Handling

**Difficulty:** 2/5 | **Impact:** 4/5

---

### Hook

A single unhandled exception in a data pipeline that has been running for six hours does not just stop the program — it erases every result computed so far unless you planned for failure.

---

### Mental Image

**Restaurant skin:** The expeditor at the pass is the try/except block. When a dish arrives, the expeditor checks it. If something is wrong (the exception), the expeditor does not let the server crash into the dining room with a ruined plate — they handle the situation: return it to the kitchen, substitute a component, and log the incident. The restaurant continues to operate.

**Human Body skin:** The immune system does not crash the entire body when it detects a pathogen. It isolates the threat (catches the exception), mounts a response (except block), and resumes normal operation (code continues after the try/except). The `finally` block is the body's baseline maintenance — the heart keeps beating regardless of whether an infection was detected.

---

### Decryption

The `try` block contains code that might raise an exception. The `except` block specifies what to do when a particular exception type is raised. The optional `else` block runs only if no exception occurred. The optional `finally` block always runs, with or without an exception — it is used for cleanup (closing files, releasing locks). Catching bare `Exception` is a last resort; catching specific exception types (`ValueError`, `FileNotFoundError`, `KeyError`) makes error handling precise and debuggable.

---

### Minimal Code

```python
def safe_divide(numerator, denominator):
    try:
        result = numerator / denominator
    except ZeroDivisionError:
        print("Cannot divide by zero. Returning None.")
        return None
    else:
        print(f"Success: {numerator} / {denominator} = {result:.4f}")
        return result
    finally:
        print("Operation attempted.")

safe_divide(10, 4)
print("---")
safe_divide(10, 0)
```

**Output:**

```
Success: 10 / 4 = 2.5000
Operation attempted.
---
Cannot divide by zero. Returning None.
Operation attempted.
```

---

### Analysis & Intuition

Error handling is documentation. The exceptions you catch declare your understanding of what can go wrong. Catching `Exception` broadly hides bugs — a typo in a variable name raises `NameError`, and a broad catch silently swallows it and returns `None`, making the downstream failure look like a data problem instead of a code problem. In ML pipelines, log the exception with `logging.exception(e)` inside the `except` block so the traceback is preserved even when execution continues.

---

### Traps & Limits

**Trap 1 — Catching too broadly:**
```python
try:
    result = compute_features(row)
except Exception:
    result = None   # Was it a data issue or a code bug? You will never know.
```

**Trap 2 — Silently swallowing the error without logging:**
```python
try:
    value = int(user_input)
except ValueError:
    pass   # The error vanishes. Debugging becomes archaeology.
```

**Mirror Mode:** The expeditor analogy implies that every problem has a known fix — send it back, substitute, continue. Some exceptions signal conditions that are genuinely unrecoverable (disk full, corrupted model weights, out of memory). Handling those by logging and returning `None` may allow the pipeline to proceed to a place where the missing value causes an even more confusing failure hundreds of steps later. Not every exception should be caught — sometimes letting the program halt and alerting the engineer is the correct response.

---

### Application

1. Write a function `parse_temperature(value)` that converts a string to a float (e.g., `"98.6"` -> `98.6`).
2. Handle `ValueError` (non-numeric string) and return `None` with a printed warning.
3. Test with `"98.6"`, `"hot"`, and `""`. Print the results.

---

---

## Module Summary

| Concept | Difficulty | Impact | Core Insight |
|---|---|---|---|
| Variables & Types | 1/5 | 5/5 | Names are pointers; types are behavioral contracts |
| Lists & Dictionaries | 2/5 | 5/5 | Access pattern determines the right container |
| Loops & Conditions | 2/5 | 5/5 | Loops are where scale and bugs compound together |
| Functions | 2/5 | 5/5 | Reusability requires naming the invariant |
| Classes & Objects | 3/5 | 4/5 | Blueprint vs instance is the core OOP distinction |
| File I/O | 2/5 | 3/5 | Data persists; programs do not |
| Error Handling | 2/5 | 4/5 | What you catch declares what you understand |

**Next:** Module 2 — NumPy & Linear Algebra Foundations
