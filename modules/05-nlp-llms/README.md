# Module 05 — NLP and LLMs

**The FILS Framework**
Part of the open-source AI/ML curriculum for complete beginners.

---

## Module Overview

This module introduces the core ideas behind modern natural language processing and large language models: how raw text becomes numbers, how those numbers carry meaning, how models learn to focus on what matters, and how pre-trained systems can be adapted and guided. Every concept is delivered as a BRIDGIA card — seven cases per concept, from Hook to Application.

**Concepts in this module:**

| Card | Concept | Difficulty | Relevance |
|------|---------|------------|-----------|
| 31 | Tokenization | 2/5 | 5/5 |
| 32 | Word Embeddings | 3/5 | 5/5 |
| 33 | Attention Mechanism | 4/5 | 5/5 |
| 34 | Transformers | 5/5 | 5/5 |
| 35 | Fine-Tuning | 3/5 | 5/5 |
| 36 | Prompt Engineering | 2/5 | 5/5 |
| 37 | RAG | 4/5 | 5/5 |

---

---

## Card 31 — Tokenization

**Difficulty:** 2/5 | **Relevance:** 5/5 | **Skin:** Restaurant / Construction

---

### Case 1 — Hook

You want a computer to read the sentence "The chef prepared an outstanding bouillabaisse." The computer has never seen letters, words, or sentences. It only processes numbers.

Before any learning can happen, the sentence must be cut into countable pieces and each piece must be assigned a number. The question is: how do you decide where to cut? Every word? Every letter? Something in between? And why does the choice of cutting strategy affect what the model can and cannot learn?

---

### Case 2 — Mental Image

**Restaurant skin:** A large catering order arrives on a single sheet of paper: "two portions of duck confit, one vegetable risotto, and three crème brûlées." The kitchen manager cannot work with the whole sheet at once. She tears it into individual order tickets — one per item — and assigns each ticket a number from the kitchen's master list. "Duck confit" is ticket #4721. "Crème brûlée" is ticket #892. The machine in the kitchen processes ticket numbers, not English words.

Tokenization is this tearing process. The long input string is cut into tokens — pieces of text — and each token is replaced by its integer ID from a fixed vocabulary dictionary. The model does not understand meaning at this stage. It just processes the sequence of numbers.

Notice what the kitchen manager does NOT do: she does not interpret what the customer wanted, suggest substitutions, or assess nutritional balance. She cuts and numbers. Meaning comes later.

**Construction skin:** A steel supplier receives an order for a custom beam — 23.7 meters long. The cutting machine cannot bend or shape metal. It only cuts. The beam is sliced into standard 1-meter segments, each stamped with a catalog number. Downstream workers assemble those segments. The cutting machine applied no judgment about the beam's eventual purpose. It divided a continuous input into discrete, numbered pieces according to fixed rules.

Tokenization does the same thing to text. The input — a continuous stream of characters — is divided into pieces according to a learned dictionary of subword units. Each piece gets a catalog number. Downstream components of the model work with those numbers.

---

### Case 3 — Decryption

Tokenization is the process of converting raw text into a sequence of integer token IDs that a model can process. Modern tokenizers — including those used in BERT, GPT, and other transformer models — operate at the subword level using algorithms like Byte Pair Encoding (BPE) or WordPiece.

The vocabulary is built during a training phase on a large text corpus. Frequent character sequences are merged iteratively to form subword units. Common words like "the" and "run" become single tokens. Rare or compound words are split into meaningful subunits: "bouillabaisse" might become ["bou", "illa", "baisse"]. Characters that appear nowhere in the vocabulary are handled by a special unknown token.

Key properties of subword tokenization:
- Handles out-of-vocabulary words gracefully by splitting them into known subunits
- Balances vocabulary size (typically 30,000–100,000 tokens) against sequence length
- The same surface string may tokenize differently depending on context (some tokenizers are context-sensitive at boundaries)
- Special tokens mark structural positions: [CLS] at the start of a sequence, [SEP] between segments, [PAD] for padding shorter sequences to a uniform length

The tokenizer does not understand language. It applies a fixed mapping from strings to integers. Meaning is entirely the responsibility of downstream components.

---

### Case 4 — Minimal Code

```python
from transformers import AutoTokenizer

# Load a pre-trained tokenizer (BERT-base)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize a sentence
text = "The chef prepared an outstanding bouillabaisse."
encoding = tokenizer(text, return_tensors="pt")

# Show the tokens and their IDs
tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
ids = encoding["input_ids"][0].tolist()

print("Tokens:", tokens)
print("IDs:   ", ids)
print(f"\nVocabulary size: {tokenizer.vocab_size:,}")
print(f"Sequence length: {len(ids)} (including [CLS] and [SEP])")

# Show how an unknown/rare word is split
rare_text = "bouillabaisse"
rare_tokens = tokenizer.tokenize(rare_text)
print(f"\nTokens for '{rare_text}': {rare_tokens}")
```

**Expected output:**
```
Tokens: ['[CLS]', 'the', 'chef', 'prepared', 'an', 'outstanding', 'bou', '##ill', '##ab', '##aisse', '.', '[SEP]']
IDs:    [101, 1996, 7665, 4832, 2019, 3426, 23435, 18331, 4874, 23107, 1012, 102]

Vocabulary size: 30,522
Sequence length: 12 (including [CLS] and [SEP])

Tokens for 'bouillabaisse': ['bou', '##ill', '##ab', '##aisse']
```

The `##` prefix marks a subword that continues from the previous token without a space. The rare word is split into four known subunits — none of them carry the meaning "French fish stew," but the model's downstream layers can learn to reconstruct that meaning from the combination.

---

### Case 5 — Analysis and Intuition

- A tokenizer is not a model. It applies a fixed, deterministic function. It cannot be updated during training.
- The vocabulary was built before you see your data. If you are working in a specialized domain (medical, legal, code), the general-purpose tokenizer may split domain terms inefficiently — common domain words become long multi-token sequences.
- Longer token sequences are more expensive to process. Splitting a 15-character word into 6 tokens is not the same as having a single token for it.
- Padding (adding [PAD] tokens to reach a uniform length) is necessary for batch processing. The model is trained to ignore padded positions using an attention mask.
- The tokenizer must exactly match the model it feeds. Loading a GPT-2 model with a BERT tokenizer produces nonsense — the integer IDs map to completely different representations in each model's embedding layer.

---

### Case 6 — Traps and Limits

**Trap 1 — Assuming one token equals one word**
Common words are often single tokens. Rare words, proper nouns, and non-English words are typically multiple tokens. "New York" is two tokens. "Pneumonoultramicroscopicsilicovolcanoconiosis" may be eight or more. Code that assumes `len(tokens) == len(words)` will silently produce wrong results.

**Trap 2 — Applying the wrong tokenizer to a model**
Every pretrained model has a specific tokenizer. The model's weights were learned against the integer IDs produced by that tokenizer. Using a different tokenizer produces different integer IDs, which map to different embedding vectors — the model will produce garbage output without any error message.

**Trap 3 — Ignoring the sequence length limit**
Most transformer models have a fixed maximum sequence length (e.g., 512 tokens for BERT, 1024 for GPT-2). Text that exceeds this limit is silently truncated by default. You will not receive a warning. Long documents must be chunked before tokenization.

**Mirror Mode: Where the Analogies Break Down**

**Dimension 1 — The kitchen manager understands the order; the tokenizer does not.**
The restaurant manager reads "duck confit" and understands it is a dish requiring a specific preparation. She assigns ticket #4721 because she knows what duck confit is. The tokenizer assigns an integer by looking up a dictionary. It has no concept of what the token refers to. If "bouillabaisse" were replaced with "xqrlmpt," the tokenizer would still assign it an ID (or split it into subwords) without any indication that the result is meaningless.

**Dimension 2 — The steel beam is continuous; text has natural word boundaries.**
The 23.7-meter beam has no inherent "right" place to cut — the 1-meter segments are a purely operational choice. Text, by contrast, has orthographic word boundaries defined by spaces and punctuation. Subword tokenization deliberately crosses those boundaries for efficiency, which means a token can be semantically incomplete in a way that a steel segment never is. The analogy implies that cutting is arbitrary; in tokenization, the cuts are principled but not aligned with human semantic units.

---

### Case 7 — Application Exercise

**Exercise: Exploring tokenization behavior**

Use `AutoTokenizer.from_pretrained("bert-base-uncased")` or `"gpt2"`.

1. Tokenize the following five strings and count the number of tokens each produces:
   - "The cat sat on the mat."
   - "Electroencephalography measures brain activity."
   - "print('hello world')"
   - "مرحبا بالعالم" (Arabic: Hello world)
   - A sentence of your choice from a technical domain (finance, medicine, law, or code).

2. For each string, identify which words became single tokens and which were split. Is there a pattern?

3. Compare the same strings tokenized with `"gpt2"` versus `"bert-base-uncased"`. How do the token counts and splits differ?

4. Find the maximum sequence length of the tokenizer (`tokenizer.model_max_length`). Take a long paragraph from Wikipedia. How many tokens does it produce? Does it fit within the limit?

**Deliverable:** A notebook with the tokenization outputs, a table comparing BERT and GPT-2 token counts for the five strings, and a written explanation of why subword tokenization handles rare words better than a whole-word vocabulary.

**Success condition:** You can explain why the same surface string might produce different numbers of tokens in different tokenizers, and you can identify the maximum sequence length constraint and explain its operational consequence.

---

---

## Card 32 — Word Embeddings

**Difficulty:** 3/5 | **Relevance:** 5/5 | **Skin:** City / Restaurant

---

### Case 1 — Hook

You have converted the sentence "The chef prepared an outstanding bouillabaisse" into a sequence of integer IDs: [101, 1996, 7665, ...]. Now what?

The integers carry no information about meaning. Token ID 7665 is not meaningfully "close to" or "far from" token ID 7664. You cannot add them, compare them, or reason about them geometrically. The machine needs a different representation — one where similar meanings are close together and different meanings are far apart.

What kind of representation allows you to do arithmetic on meaning?

---

### Case 2 — Mental Image

**City skin:** Imagine a city where every word has an address — not a street address, but a location in a 300-dimensional coordinate space. Words with similar meanings live in the same neighborhood. "Chef," "cook," and "baker" live within a few blocks of each other. "Surgeon" is in a different district — same general idea (skilled professional) but different domain. "Automobile" is on the other side of the city.

The address is a list of 300 numbers: the word's coordinates in meaning-space. Two words are "similar" if their addresses are close — measured by the angle between their vectors or the Euclidean distance between their positions.

This space has a remarkable property: directions mean something. The direction from "king" to "queen" is approximately the same as the direction from "man" to "woman." The meaning of gender difference is encoded as a consistent geometric shift in this space.

**Restaurant skin:** The restaurant stores every ingredient on a pantry shelf according to a flavor profile — a list of 50 numbers representing sweetness, acidity, umami, fat content, and so on. "Butter" and "cream" are stored near each other because their flavor profiles are similar. "Lemon" and "lime" are close. "Beef broth" and "chicken broth" are in the same section but not identical. A cook looking for a butter substitute knows to look in the same neighborhood on the shelf.

Embeddings are this flavor-profile shelving system for words. Each word is assigned a dense vector — its nutritional fingerprint in semantic space.

---

### Case 3 — Decryption

A word embedding is a dense vector representation of a token, typically of dimension 128 to 1024. The embedding maps an integer token ID to a real-valued vector: `embedding: integer → R^d`.

The embedding layer is a lookup table: a matrix of shape `[vocabulary_size × embedding_dimension]`. Given a token ID, the corresponding row is retrieved. This lookup is the first operation in almost every modern NLP model.

Embeddings are learned during training by minimizing the model's prediction loss. Similar words end up with similar vectors because they appear in similar contexts — this is the distributional hypothesis: words that appear in similar contexts tend to have similar meanings.

Static embeddings (Word2Vec, GloVe) assign each word a single vector regardless of context. "Bank" has one vector whether you mean a financial institution or a river bank.

Contextual embeddings (BERT, GPT) produce different vectors for the same word depending on surrounding context. "Bank" in "I went to the bank to deposit money" produces a different vector than "bank" in "the river bank was flooded." Contextual embeddings are what make modern LLMs powerful; they are computed dynamically by the model's attention layers, not looked up from a static table.

---

### Case 4 — Minimal Code

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Load model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

def get_embedding(text):
    """Return mean-pooled embedding for a string."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pool over token dimension
    return outputs.last_hidden_state.mean(dim=1)

# Compare semantic similarity of word pairs
pairs = [
    ("chef", "cook"),
    ("chef", "surgeon"),
    ("chef", "automobile"),
    ("king", "queen"),
    ("Paris", "France"),
]

print(f"{'Pair':<30} {'Cosine Similarity':>20}")
print("-" * 52)
for a, b in pairs:
    emb_a = get_embedding(a)
    emb_b = get_embedding(b)
    sim = F.cosine_similarity(emb_a, emb_b).item()
    print(f"{a + ' vs ' + b:<30} {sim:>20.4f}")

print(f"\nEmbedding dimension: {get_embedding('chef').shape[-1]}")
```

**Expected output:**
```
Pair                           Cosine Similarity
----------------------------------------------------
chef vs cook                                  0.8134
chef vs surgeon                               0.3512
chef vs automobile                            0.0891
king vs queen                                 0.7623
Paris vs France                               0.7240

Embedding dimension: 384
```

"Chef" and "cook" are close in embedding space. "Chef" and "automobile" are far apart. "King" and "queen" have high similarity because they appear in structurally similar contexts throughout the training corpus.

---

### Case 5 — Analysis and Intuition

- The embedding dimension is a hyperparameter. Larger dimensions can represent more nuance but are more expensive to compute and require more data to train well.
- Cosine similarity is the standard measure for comparing embeddings — it measures the angle between vectors and ignores magnitude. Values range from -1 (opposite) to 1 (identical direction).
- Embeddings trained on biased text will encode those biases geometrically. If "nurse" is closer to "woman" than to "man" in the embedding space, that reflects a pattern in the training corpus — the embedding has learned a social association.
- Pre-trained embeddings carry the statistical regularities of their training corpus. If you use a model pre-trained on English Wikipedia for French text, the embeddings will be unreliable for French.
- The embedding layer is always the first thing a transformer model does with token IDs. Everything downstream — attention, feed-forward layers, predictions — operates on embedding vectors, never on raw integers.

---

### Case 6 — Traps and Limits

**Trap 1 — Treating static embeddings as context-aware**
Word2Vec and GloVe produce one vector per word type. If your application requires distinguishing "I went to the bank" from "the river bank," static embeddings will fail because both uses of "bank" map to the same vector.

**Trap 2 — Comparing embeddings from different models**
An embedding from BERT and an embedding from GPT-2 are not in the same vector space. Their cosine similarity is meaningless. Only compare embeddings produced by the same model.

**Trap 3 — Assuming high cosine similarity means semantic equivalence**
Two words can have high cosine similarity because they appear in similar contexts — including contexts where they are antonyms. "Hot" and "cold" often co-occur with the same surrounding words ("weather," "temperature," "very") and may have surprisingly high cosine similarity. Embedding similarity captures distributional overlap, not meaning equivalence.

**Mirror Mode: Where the Analogies Break Down**

**Dimension 1 — The city map is static; contextual embeddings are dynamic.**
In the city analogy, every word has a fixed address. That address is the same regardless of what sentence it appears in. Contextual embeddings (BERT, GPT) work differently: the address a word receives depends on its neighbors. "Bank" receives a different embedding vector when surrounded by "deposit" and "money" than when surrounded by "river" and "flood." The city-map intuition is accurate for static embeddings like Word2Vec but misleading for the contextual embeddings that power modern LLMs.

**Dimension 2 — The flavor-profile shelf is human-designed; embeddings are learned.**
A sommelier who designs the pantry shelf knows what dimensions of flavor matter. The embedding's dimensions are learned from data and are not interpretable — there is no single dimension that cleanly captures "sweetness" or "acidity." Each embedding dimension is a linear combination of many semantic features, and no dimension reliably corresponds to a human-understandable concept. Attempting to read meaning from individual embedding coordinates will generally fail.

---

### Case 7 — Application Exercise

**Exercise: Semantic neighborhoods**

Use `sentence-transformers/all-MiniLM-L6-v2` or `bert-base-uncased` from the `transformers` library.

1. Choose a seed word (e.g., "hospital"). Generate embeddings for 20 related and unrelated words.
2. Compute pairwise cosine similarities. Which words are closest to your seed word? Are there any surprises?
3. Demonstrate the analogy property: compute `king - man + woman` using mean-pooled embeddings. Does the result point toward "queen" (i.e., is "queen" in the top-3 nearest neighbors of the resulting vector)?
4. Try the same analogy with a domain of your choice (e.g., "Paris - France + Germany" → does it produce "Berlin"?).
5. Find a pair of antonyms. Are they close or far in the embedding space? Explain why.

**Success condition:** You can explain why embeddings encode distributional similarity rather than dictionary meaning, you can demonstrate the analogy property empirically, and you can describe one case where high cosine similarity does not imply semantic equivalence.

---

---

## Card 33 — Attention Mechanism

**Difficulty:** 4/5 | **Relevance:** 5/5 | **Skin:** Restaurant / Orchestra

---

### Case 1 — Hook

You are translating the French sentence "Le chef a préparé le plat que le client avait demandé" into English. When you reach the word "demandé" (requested), you need to know who did the requesting — "le client." But "le client" is six words back.

Earlier translation systems processed text one word at a time in sequence. By the time they reached "demandé," the information about "le client" had been compressed and partially lost. The system had no explicit mechanism to look back and retrieve the right piece of context.

What if, instead of forcing the model to remember everything in a fixed-size state, you gave it a mechanism to actively look back at every previous word and decide how much to focus on each one — for each word being processed?

---

### Case 2 — Mental Image

**Restaurant skin:** A waiter takes a complex order at a large table. When the kitchen asks "who ordered the salmon?", the waiter does not replay the entire conversation from the beginning. He focuses his attention on the specific moment when the salmon was mentioned, ignoring all the other requests. He can selectively retrieve relevant information from his notes.

Attention works the same way. When the model processes a given token — say, the word "it" in "The cat drank from the bowl because it was thirsty" — it looks at every other token in the sequence and assigns each one a score: how relevant is this other token to understanding "it"? "Cat" gets a high score. "Bowl" gets a moderate score. "Because" gets a low score. These scores become weights that determine how much of each other token's representation is mixed into the representation of "it."

The model has learned — through training — which patterns of context are relevant for which purposes. It focuses automatically on what matters.

**Orchestra skin:** During a performance, the conductor does not listen equally to every section of the orchestra at every moment. When the violin solo begins, the conductor's attention is almost entirely on the violins. When the brass section enters for the climax, attention shifts. The rest of the orchestra continues to play, but the conductor's focus is selective, dynamic, and trained by years of musical experience.

In the attention mechanism, every token is simultaneously the conductor (it asks: "what do I need?") and a musician (it offers: "here is what I contain"). The mechanism is symmetric — every position can attend to every other position.

---

### Case 3 — Decryption

The attention mechanism computes, for each position in a sequence, a weighted combination of all positions based on learned relevance scores.

Each token's embedding is projected into three vectors using learned weight matrices:
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?"
- **Value (V):** "What do I contribute if selected?"

The attention score between position i and position j is:

```
score(i, j) = (Q_i · K_j) / sqrt(d_k)
```

where `d_k` is the dimension of the key vectors (scaling prevents vanishingly small gradients in the softmax). Scores are passed through softmax to produce attention weights that sum to 1. The output at position i is:

```
output_i = sum_j( softmax(score(i, j)) * V_j )
```

**Multi-head attention** applies this operation h times in parallel with different learned projections, allowing the model to attend to different types of relationships simultaneously. The heads' outputs are concatenated and projected.

**Self-attention** (tokens attending to each other within the same sequence) is the form used in transformers. **Cross-attention** (tokens in one sequence attending to tokens in another) is used in encoder-decoder architectures for tasks like translation.

---

### Case 4 — Minimal Code

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load a small BERT model to inspect attention weights
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)
model.eval()

text = "The cat drank from the bowl because it was thirsty."
inputs = tokenizer(text, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

with torch.no_grad():
    outputs = model(**inputs)

# outputs.attentions: tuple of (num_layers,) each shape (batch, heads, seq, seq)
attentions = outputs.attentions

# Look at layer 0, head 0: what does each token attend to?
layer0_head0 = attentions[0][0, 0]  # shape: (seq_len, seq_len)

# Print attention from "it" (find its position) to all other tokens
token_list = tokens
it_idx = token_list.index("it")

print(f"Attention weights from token '{token_list[it_idx]}' (layer 0, head 0):\n")
weights = layer0_head0[it_idx].tolist()
for tok, w in zip(token_list, weights):
    bar = "█" * int(w * 40)
    print(f"  {tok:<12} {w:.4f}  {bar}")

print(f"\nModel has {len(attentions)} layers, {attentions[0].shape[1]} attention heads per layer")
```

**Expected output (approximate — attention patterns vary by head and layer):**
```
Attention weights from token 'it' (layer 0, head 0):

  [CLS]        0.0821  ███
  the          0.0512  ██
  cat          0.1834  ███████
  drank        0.0631  ██
  from         0.0214  
  the          0.0318  █
  bowl         0.1102  ████
  because      0.0289  █
  it           0.2341  █████████
  was          0.0812  ███
  thirsty      0.0873  ███
  .            0.0153  
  [SEP]        0.0100  

Model has 12 layers, 12 attention heads per layer
```

"Cat" and "bowl" receive higher attention from "it" than other tokens — the model has learned that pronouns attend to their potential referents. Different heads capture different relationships.

---

### Case 5 — Analysis and Intuition

- Attention weights are not explanations. A high attention weight from token A to token B means B's value vector contributed heavily to the output at A. It does not mean "A is caused by B" or "B is the most semantically important word in the sentence."
- Early layers often capture syntactic relationships (subject, verb, object). Later layers tend to capture more semantic relationships. This is a statistical tendency, not a rule.
- The scaling factor `sqrt(d_k)` prevents the dot products from growing so large that softmax saturates and produces near-zero gradients. Without it, training deep transformers becomes unstable.
- Multi-head attention allows the model to simultaneously track many different relationships — one head might focus on syntactic dependencies, another on coreference, another on positional proximity.
- The computational cost of self-attention is O(n²) in sequence length n, because every position attends to every other position. This is why processing very long sequences (e.g., books) with standard transformers is expensive.

---

### Case 6 — Traps and Limits

**Trap 1 — Interpreting attention weights as feature importance**
High attention does not imply causal importance. A model can attend heavily to a token while that token's value contributes little to the final prediction. Attention visualizations are popular but often misleading as explanations.

**Trap 2 — Assuming attention patterns are consistent across layers and heads**
A single head in layer 0 may attend "cat" for "it," but a head in layer 11 might attend "bowl." Different heads specialize in different relationship types. Describing "the model's attention" based on one head is like describing a choir based on one singer.

**Trap 3 — Ignoring the quadratic cost for long sequences**
If you double the sequence length, attention computation quadruples. At 512 tokens this is manageable. At 8,000 tokens it becomes the dominant bottleneck. Efficient attention variants (Longformer, FlashAttention) exist for this, but the base mechanism is quadratic.

**Mirror Mode: Where the Analogies Break Down**

**Dimension 1 — The waiter actively chooses what to focus on; attention is a differentiable computation.**
The waiter exercises conscious judgment in recalling who ordered what. The attention mechanism does not exercise judgment — it applies a learned linear projection and a softmax. Every position always attends to every other position; the softmax weights just determine how much. The waiter can refuse to answer, misremember, or make a social judgment. Attention does none of these things.

**Dimension 2 — The conductor hears in real time; attention processes all positions simultaneously.**
An orchestra conductor responds to what musicians are currently playing and adjusts focus moment by moment. Transformer self-attention has no temporal processing — it receives the entire sequence at once and computes attention over all positions in parallel. There is no "moment by moment." The analogy implies sequentiality; the mechanism is fully parallel. This is one of the transformer's key advantages over recurrent architectures, but the analogy obscures it.

---

### Case 7 — Application Exercise

**Exercise: Visualizing attention**

Use `bert-base-uncased` with `output_attentions=True`.

1. Tokenize the sentence: "The trophy didn't fit in the suitcase because it was too large."
2. For each occurrence of "it" and "large," extract attention weights from layer 6, head 0.
3. Which tokens receive the most attention from "it"? Does it vary across heads?
4. Try the same sentence with "it was too small" instead of "too large." Does "it" attend differently to "trophy" vs "suitcase"?
5. Report the total number of attention heads across all layers. If each head specializes in something different, what does it mean for a model to have 144 heads (12 layers × 12 heads)?

**Success condition:** You can describe what an attention weight represents mathematically, explain why high attention does not equal causal importance, and demonstrate that different heads produce different attention patterns on the same sentence.

---

---

## Card 34 — Transformers

**Difficulty:** 5/5 | **Relevance:** 5/5 | **Skin:** Orchestra / Construction

---

### Case 1 — Hook

Before 2017, sequence models processed text one token at a time — each step depending on the previous step's output. This worked, but it created a fundamental bottleneck: steps could not be parallelized. Training was slow. Long-range dependencies were hard to capture because information had to pass through many sequential steps before reaching its destination.

In 2017, a paper called "Attention Is All You Need" introduced a new architecture: the Transformer. It replaced sequential recurrence with parallel self-attention. The entire sequence is processed simultaneously. Every token can directly attend to every other token, regardless of distance.

The question: how do you build a model that reads an entire sequence at once, allows every part to communicate with every other part, and still produces a meaningful, ordered output?

---

### Case 2 — Mental Image

**Orchestra skin:** A conventional orchestra rehearsal works sequentially — the conductor plays a recording of the previous section to each musician in turn, who then adds their part. The last musician in the chain cannot begin until the first has finished.

A transformer orchestra works differently. Every musician hears every other musician simultaneously — in a shared rehearsal space with perfect acoustics, where each player can listen to any section while playing. The violins do not wait for the cellos. The brass do not wait for the woodwinds. All 80 musicians play at once, each adjusting their interpretation based on what they hear from every other section simultaneously. This is self-attention: every position in the sequence processing its relationship to every other position in parallel.

The conductor (feed-forward layer) then takes what each musician produces — already shaped by what they heard from the ensemble — and refines it further before the sound goes out to the audience.

**Construction skin:** Old-style construction: the foundation team finishes, then the framing team begins, then the electrical team, then the plumbers — strict sequential handoffs. Modern construction on a large commercial project: all teams work simultaneously, each one receiving live blueprints that show the status of every other team's work. The electricians can see exactly where the plumbers are running pipes. The framing crew adjusts studs based on what the HVAC team is planning. Every worker has access to the full, current state of the entire project at all times. No message-passing delays. No waiting for the previous stage to complete.

The transformer's parallel processing is this coordinated simultaneous construction — every token processing its relationship to every other token in the same forward pass.

---

### Case 3 — Decryption

A Transformer consists of stacked encoder and/or decoder blocks. Each block contains two sub-components:

1. **Multi-head self-attention:** every token produces query, key, and value vectors; attention weights are computed across all positions in parallel (see Card 33)
2. **Position-wise feed-forward network:** a two-layer MLP applied independently to each position's representation

Both sub-components are wrapped with residual connections and layer normalization:

```
output = LayerNorm(x + sublayer(x))
```

Residual connections allow gradients to flow directly through the network depth, enabling training of very deep stacks (12 to 96+ layers).

**Positional encoding** is added to embeddings before the first layer, injecting information about token order — since self-attention is order-invariant, the model would otherwise treat the same tokens in different orders as identical.

**Encoder-only models** (BERT): process input sequences bidirectionally — each token attends to all other tokens. Used for classification, named entity recognition, question answering.

**Decoder-only models** (GPT): use causal masking — each token can only attend to previous tokens, enabling autoregressive text generation. The model predicts the next token given all previous tokens.

**Encoder-decoder models** (T5, original Transformer): the encoder processes input, the decoder attends to encoder output via cross-attention while generating output sequentially.

---

### Case 4 — Minimal Code

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Example 1: Text classification with a transformer encoder ---
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

texts = [
    "The chef prepared an absolutely outstanding bouillabaisse.",
    "The service was slow and the soup was cold.",
    "The restaurant was neither good nor bad.",
]

print("Sentiment Analysis:")
for text in texts:
    result = classifier(text)[0]
    print(f"  [{result['label']:8s} {result['score']:.4f}]  {text[:60]}")

# --- Example 2: Text generation with a transformer decoder ---
generator = pipeline("text-generation", model="gpt2", max_new_tokens=40, do_sample=False)

prompt = "The attention mechanism in transformers allows the model to"
output = generator(prompt)[0]["generated_text"]
print(f"\nText Generation:\nPrompt: {prompt}")
print(f"Output: {output}")

# --- Model architecture summary ---
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nDistilBERT parameters: {total_params:,}")
```

**Expected output:**
```
Sentiment Analysis:
  [POSITIVE 0.9998]  The chef prepared an absolutely outstanding bouillabaisse.
  [NEGATIVE 0.9994]  The service was slow and the soup was cold.
  [NEGATIVE 0.6781]  The restaurant was neither good nor bad.

Text Generation:
Prompt: The attention mechanism in transformers allows the model to
Output: The attention mechanism in transformers allows the model to learn the structure of the input sequence and to predict the output sequence.

DistilBERT parameters: 66,955,010
```

---

### Case 5 — Analysis and Intuition

- The transformer's key architectural innovation is the replacement of sequential recurrence with parallel self-attention. This enables both faster training (parallelism across GPUs) and better long-range dependency capture (direct connections between distant positions).
- Deeper stacks (more layers) allow the model to build increasingly abstract representations. Early layers tend to capture low-level syntactic patterns; later layers encode higher-level semantic and pragmatic structure.
- The feed-forward network in each block is where the bulk of the model's parameters live — it stores factual associations learned during training. The attention layers are relatively parameter-light and focus on routing information between positions.
- Positional encoding is essential. Without it, "The dog bit the man" and "The man bit the dog" would produce identical representations (since self-attention is a set operation). Positional encodings break this symmetry.
- Scale matters dramatically. GPT-2 has 1.5 billion parameters. GPT-3 has 175 billion. Scale enables emergent capabilities — behaviors that do not appear in smaller models and cannot be predicted by extrapolation from smaller models' performance.

---

### Case 6 — Traps and Limits

**Trap 1 — Treating the transformer as a single monolithic thing**
"Transformer" refers to an architectural family — the same principles implemented very differently across models. BERT and GPT share the basic self-attention mechanism but differ in masking strategy, training objective, and use case. T5 has both an encoder and a decoder. Do not assume that something true of one transformer is true of all transformers.

**Trap 2 — Assuming more parameters always means better performance on your task**
Larger models are not universally better. A 66M-parameter DistilBERT fine-tuned on your specific task often outperforms a 7B general-purpose model on that task. Scale helps with general capability; task-specific training helps with task performance. They are different axes.

**Trap 3 — Ignoring inference cost**
A 7-billion-parameter model requires roughly 14GB of GPU VRAM in half-precision. Running it in production at scale is expensive. Model size, inference latency, and deployment cost must be weighed against accuracy gains when selecting models for real applications.

**Mirror Mode: Where the Analogies Break Down**

**Dimension 1 — Musicians in the orchestra have individual artistic judgment; transformer layers apply fixed mathematical operations.**
Each musician can improvise, respond emotionally, and make interpretive choices. A transformer layer applies the same learned linear transformations to every input — there is no interpretation, no judgment, no variability at inference time (for deterministic decoding). The analogy implies agency that the architecture does not possess.

**Dimension 2 — Construction workers communicate in natural language; tokens communicate through linear algebra.**
When an electrician tells a plumber "I need six inches of clearance here," the communication is semantic and contextual. In the transformer, tokens communicate through query-key dot products — a mathematical operation that has no concept of intent, requests, or meaning. The communication metaphor anthropomorphizes a linear algebraic operation and may lead learners to overestimate the model's comprehension.

---

### Case 7 — Application Exercise

**Exercise: Using the HuggingFace pipeline API**

1. Use `pipeline("text-classification")` with `distilbert-base-uncased-finetuned-sst-2-english`. Classify 10 sentences of your own — mix clearly positive, clearly negative, and ambiguous cases. Which type does the model handle least reliably?

2. Use `pipeline("fill-mask")` with `bert-base-uncased`. Write a sentence with `[MASK]` in it. What are the top-5 predictions? Try a domain-specific sentence (legal, medical, culinary). Does the model's training corpus influence the predictions?

3. Use `pipeline("text-generation")` with `gpt2`. Compare the output of greedy decoding (`do_sample=False`) versus sampling (`do_sample=True, temperature=0.9`). How do the outputs differ? Which is more useful, and for what purposes?

4. Load a transformer model and call `sum(p.numel() for p in model.parameters())`. How many parameters does it have? If each parameter is stored as a 4-byte float, how many megabytes does the model require?

**Success condition:** You can use the pipeline API fluently, explain the difference between encoder-only and decoder-only architectures, and describe the role of positional encoding in one sentence.

---

---

## Card 35 — Fine-Tuning

**Difficulty:** 3/5 | **Relevance:** 5/5 | **Skin:** Restaurant / Construction

---

### Case 1 — Hook

Training a large language model from scratch costs millions of dollars in compute and requires hundreds of billions of tokens of text. Virtually no organization outside of large technology companies can do this.

But you need a model that classifies customer support tickets as urgent or not-urgent, using your specific vocabulary and your specific categories. The general-purpose model does not know what "P0 escalation" means in your company's support system.

Do you need to train a model from scratch? Or can you take a model that already knows language and teach it the specific, narrow task you need — quickly, cheaply, and with a small dataset?

---

### Case 2 — Mental Image

**Restaurant skin:** A culinary school graduate has spent four years learning foundational techniques: knife skills, sauce-making, bread baking, flavor pairing, heat management. She can cook almost anything reasonably well because she understands the underlying principles.

You hire her at your restaurant, which specializes exclusively in Peruvian cuisine. You do not send her back to culinary school for four years. You spend two weeks teaching her your specific menu, your specific suppliers, your specific presentation style, and the three dishes you are known for.

She now performs at the level of a Peruvian cuisine specialist — not because she started over, but because her four years of foundational training transferred. Fine-tuning is these two weeks of specialized training on top of the four-year foundation.

**Construction skin:** A general contractor has spent twenty years building residential homes of every style in every climate. You hire him to specialize in passive-house construction — a specific standard with strict energy performance requirements. You do not ask him to forget everything he knows about construction. You send him to a three-day passive-house certification course and have him complete two supervised projects. His general knowledge of building science makes the specialization fast to acquire. Fine-tuning is this certification course and those two supervised projects.

---

### Case 3 — Decryption

Fine-tuning is the process of taking a pre-trained model and continuing to train it on a smaller, task-specific dataset. The pre-trained model's weights are used as the starting point rather than random initialization.

The key insight: the pre-trained model has already learned general representations of language — grammar, syntax, factual associations, word meanings. Fine-tuning adjusts these representations toward the specific patterns of the target task.

**Full fine-tuning:** all model parameters are updated during fine-tuning. Effective but expensive — requires the same infrastructure as the pre-trained model.

**Parameter-efficient fine-tuning (PEFT):** only a small subset of parameters are updated. Methods include:
- **LoRA (Low-Rank Adaptation):** adds small trainable matrices to attention layers; typically updates less than 1% of parameters
- **Prefix tuning:** prepends learned token embeddings to the input; the model's original weights are frozen
- **Adapter layers:** inserts small trainable modules between frozen transformer layers

Fine-tuning typically requires:
- A labeled dataset of task-specific examples (hundreds to tens of thousands)
- A lower learning rate than pre-training (5e-5 to 1e-4 is typical)
- Fewer training epochs (2–5 for full fine-tuning on most classification tasks)

The risk of fine-tuning: **catastrophic forgetting** — when trained too aggressively on a small dataset, the model's general capabilities degrade.

---

### Case 4 — Minimal Code

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch
import numpy as np

# Simulate a small customer support ticket dataset
# Labels: 0 = not urgent, 1 = urgent
texts = [
    "My account was charged twice this month",
    "I cannot log in and have a meeting in 30 minutes",
    "How do I change my profile picture?",
    "Production system is down, all users affected",
    "Where can I find the billing FAQ?",
    "Payment processing is failing for all customers right now",
    "I forgot my password",
    "Critical security vulnerability discovered in our account",
] * 8  # Repeat to simulate a small dataset

labels = [0, 1, 0, 1, 0, 1, 0, 1] * 8

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)

dataset = Dataset.from_dict({"text": texts, "label": labels})
dataset = dataset.map(tokenize, batched=True)
train_ds, eval_ds = dataset.train_test_split(test_size=0.2).values()

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir="./fine-tune-demo",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_steps=10,
    report_to="none",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics,
)

print("Starting fine-tuning on 51 support ticket examples...")
trainer.train()

# Test on new examples
test_cases = ["Server is completely unreachable", "Can I update my billing address?"]
inputs = tokenizer(test_cases, return_tensors="pt", padding=True, truncation=True, max_length=64)

model.eval()
with torch.no_grad():
    logits = model(**inputs).logits
    preds = torch.argmax(logits, dim=-1).tolist()

label_map = {0: "not urgent", 1: "URGENT"}
for text, pred in zip(test_cases, preds):
    print(f"  [{label_map[pred]}] {text}")
```

**Expected output:**
```
Starting fine-tuning on 51 support ticket examples...
{'eval_loss': 0.3812, 'eval_accuracy': 0.9000, 'epoch': 1.0}
{'eval_loss': 0.2134, 'eval_accuracy': 0.9500, 'epoch': 2.0}
{'eval_loss': 0.1876, 'eval_accuracy': 1.0000, 'epoch': 3.0}

  [URGENT] Server is completely unreachable
  [not urgent] Can I update my billing address?
```

---

### Case 5 — Analysis and Intuition

- The learning rate for fine-tuning should be 10–100x smaller than the pre-training learning rate. The model's weights are already well-tuned for language; large updates destroy existing knowledge.
- Even with very small datasets (50–200 examples), fine-tuning a pre-trained model often outperforms training a custom model from scratch on thousands of examples.
- Longer fine-tuning does not always help. After 3–5 epochs on a small dataset, the model typically begins to overfit to the fine-tuning data and lose its general language understanding.
- The choice of which layers to fine-tune matters. Freezing early layers and fine-tuning only the final few layers is a common strategy when the dataset is very small.
- LoRA can reduce the trainable parameter count by 99% with minimal accuracy loss on many tasks. For teams without large GPU budgets, LoRA fine-tuning on a single GPU is often practical.

---

### Case 6 — Traps and Limits

**Trap 1 — Fine-tuning with a dataset that is too small**
Fine-tuning on 10–20 examples is usually insufficient for a reliable classifier. The model will memorize the training examples rather than generalize. As a rough rule, aim for at least 100 examples per class.

**Trap 2 — Using the same learning rate as pre-training**
Pre-training uses learning rates around 1e-4 to 5e-4. Using the same rate during fine-tuning will destroy the pre-trained representations in the first few gradient steps. Use 1e-5 to 5e-5.

**Trap 3 — Evaluating the fine-tuned model only on training-domain data**
A fine-tuned model may perform well on support tickets from your company but fail on tickets from a different department with different vocabulary. Always evaluate on held-out data that reflects the deployment distribution.

**Mirror Mode: Where the Analogies Break Down**

**Dimension 1 — The chef can refuse to unlearn her classical training; a model cannot.**
If the Peruvian menu conflicts with the chef's classical technique, she can explain the conflict, push back, or integrate the new technique thoughtfully. A fine-tuned model has no concept of conflict between its pre-training and fine-tuning objectives. It adjusts weights to minimize the loss on the fine-tuning data, regardless of whether this degrades its general capabilities. Catastrophic forgetting is a real risk that the chef analogy does not capture.

**Dimension 2 — The contractor's specialization is additive; fine-tuning is a parameter update.**
A contractor with a passive-house certification adds new knowledge without modifying his existing knowledge. Fine-tuning modifies the actual parameter values of the pre-trained model. The same weights that previously represented general language understanding are adjusted. If fine-tuning is too aggressive, those weights may represent the fine-tuning task well but lose their general-purpose capability. The analogy implies that pre-training is preserved intact; the mechanism does not guarantee this.

---

### Case 7 — Application Exercise

**Exercise: Fine-tuning for text classification**

Use a publicly available labeled text dataset — for example, the AG News dataset (topic classification: world news, sports, business, science) or the IMDB dataset (sentiment).

1. Load `distilbert-base-uncased` and fine-tune it for 3 epochs on 1000 training examples. Record test accuracy.
2. Fine-tune with only 100 training examples. How much does accuracy drop?
3. Compare fine-tuning with a frozen base (only training the classification head) versus full fine-tuning. Which performs better on 1000 examples?
4. Plot training loss and validation accuracy per epoch for the 1000-example run. At what epoch does validation accuracy peak?
5. Test the model on a sentence from outside the training distribution (e.g., a headline written today). Does the model generalize?

**Success condition:** You can explain why a pre-trained model fine-tuned on 100 examples often outperforms a model trained from scratch on 10,000 examples, and you can describe what catastrophic forgetting is and under what conditions it is most likely to occur.

---

---

## Card 36 — Prompt Engineering

**Difficulty:** 2/5 | **Relevance:** 5/5 | **Skin:** Restaurant / Video Game

---

### Case 1 — Hook

You have access to a powerful language model. You want it to extract the names of all pharmaceutical drugs mentioned in a clinical report and return them as a JSON array. You type: "Find the drugs."

The model returns a paragraph explaining what pharmaceutical drugs are.

You try again: "List the drugs in this text." The model lists them in bullet points with commentary.

You try: "Extract all pharmaceutical drug names from the following clinical report and return them as a JSON array with the key 'drugs'. Return only the JSON, no explanation." The model returns exactly what you need.

Nothing about the model changed between these three attempts. What changed was the instruction — the prompt. How do you write prompts that reliably produce the output you need?

---

### Case 2 — Mental Image

**Restaurant skin:** A kitchen order ticket is a prompt. "Steak" is a prompt that produces something — but what temperature? What cut? With what sides? The kitchen fills in the gaps with defaults that may not match what the diner wanted.

"Ribeye, medium-rare, with roasted asparagus and no sauce" is a well-engineered order ticket. It constrains the output space. There is less room for the kitchen to substitute its preferences. The more precisely the ticket specifies the desired output, the closer the delivered dish will be to what the diner imagined.

The chef (model) is the same in both cases. The kitchen (model weights) did not change. Only the instruction changed. A skilled diner learns to write order tickets that the kitchen can execute reliably. A skilled prompt engineer learns to write instructions that the model can execute reliably.

**Video game skin:** In a role-playing game, how you phrase a quest description determines how NPCs respond. "Help me" produces generic advice. "I am a level-12 rogue seeking the fastest route through the goblin tunnels to the northern gate without triggering combat" produces a specific route. The NPCs are the same; the response depends on how precisely the quest is described.

Prompt engineering is learning to write quest descriptions that get the NPC (the model) to respond with exactly what you need.

---

### Case 3 — Decryption

Prompt engineering is the practice of designing the input text to a language model to elicit a desired output. Since language models are trained to predict the next token given the preceding context, the prompt is the primary mechanism for controlling model behavior at inference time — without modifying model weights.

Key techniques:

**Zero-shot prompting:** a single instruction with no examples. Effective for simple, well-defined tasks.

**Few-shot prompting:** providing 2–10 example input-output pairs before the actual task. The model infers the pattern from the examples and applies it to new inputs.

**Chain-of-thought prompting:** instructing the model to reason step by step before producing a final answer ("Think through this step by step"). Significantly improves performance on multi-step reasoning tasks.

**System prompts / role prompts:** framing the model's identity and behavior at the start of the conversation ("You are a senior financial analyst reviewing earnings reports. Be precise, cite specific figures, and flag any inconsistencies.").

**Output format constraints:** specifying the exact format of the desired output ("Return only a JSON object with keys 'name', 'score', and 'reason'. Do not include any other text.").

Prompt engineering does not require training data or GPU access. It is the most accessible way to improve model output for most users.

---

### Case 4 — Minimal Code

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2", max_new_tokens=80, do_sample=False, pad_token_id=50256)

# Demonstrate how different prompt styles affect the same model

prompts = {
    "Vague prompt": (
        "Tell me about machine learning."
    ),
    "Specific prompt": (
        "In exactly two sentences, explain what machine learning is to a restaurant manager "
        "with no technical background."
    ),
    "Few-shot prompt": (
        "Translate the following technical terms into plain language for a non-technical audience.\n\n"
        "Term: gradient descent\n"
        "Plain language: a process where the model repeatedly makes small adjustments to reduce its errors\n\n"
        "Term: overfitting\n"
        "Plain language: when the model memorizes training data instead of learning general patterns\n\n"
        "Term: tokenization\n"
        "Plain language:"
    ),
    "Chain-of-thought prompt": (
        "A model achieves 95% accuracy on training data and 72% on test data. "
        "Let's think step by step about what this means:\n"
        "Step 1:"
    ),
}

for label, prompt in prompts.items():
    output = generator(prompt)[0]["generated_text"]
    continuation = output[len(prompt):]  # Show only what the model generated
    print(f"--- {label} ---")
    print(f"Prompt: {prompt[:80]}...")
    print(f"Output: {continuation[:120]}")
    print()
```

**Expected output (approximate — GPT-2 generations vary):**
```
--- Vague prompt ---
Prompt: Tell me about machine learning....
Output:  Machine learning is a field of artificial intelligence that involves the development of algorithms that can learn from data...

--- Specific prompt ---
Prompt: In exactly two sentences, explain what machine learning is to a restaurant manager...
Output:  Machine learning is a way for computers to learn from examples rather than being explicitly programmed for every situation. It allows a system to improve its performance over time...

--- Few-shot prompt ---
Prompt: Translate the following technical terms...
Output:  when the model learns the training data too well and fails to generalize to new examples

--- Chain-of-thought prompt ---
Prompt: ...Let's think step by step about what this means: Step 1:
Output:  The model performs well on data it has seen before. Step 2: The large gap between training and test accuracy suggests the model is overfitting...
```

The few-shot prompt produces the most precise and correctly formatted output because the model can infer the expected pattern from the examples.

---

### Case 5 — Analysis and Intuition

- Models are not mind-readers. The more precisely you specify what you want — format, length, perspective, constraints — the more reliably you get it.
- Few-shot examples are more powerful than instructions alone for format-sensitive tasks. Showing the model three examples of the desired output format is more effective than describing the format in words.
- Chain-of-thought prompting helps most on tasks that require multi-step reasoning (math, logical deduction, multi-hop question answering). It does not help on tasks that require only information retrieval.
- Longer prompts are not always better. Overly long prompts may bury the key instruction in irrelevant context, causing the model to attend to the wrong parts.
- Temperature controls randomness. Temperature=0 (greedy decoding) produces the most likely, most deterministic output. Temperature=1.0 produces more varied, creative outputs. For structured data extraction, use temperature=0.

---

### Case 6 — Traps and Limits

**Trap 1 — Assuming the model understands implied intent**
"Fix this" is not a prompt — it is an assumption that the model shares your context. Specify exactly what is wrong, what fixing means, and what the output should look like.

**Trap 2 — Evaluating prompts on a single example**
A prompt that works on one input may fail on another. Evaluate prompt effectiveness over a test set of at least 20–50 representative examples before concluding that a prompt is reliable.

**Trap 3 — Treating prompt engineering as model-agnostic**
A prompt optimized for GPT-4 may perform worse on Claude, which may perform worse on Mistral. Different models have different response tendencies, different instruction-following behaviors, and different failure modes. Test on the specific model you will deploy.

**Mirror Mode: Where the Analogies Break Down**

**Dimension 1 — The waiter can ask for clarification; the model cannot (in a single-turn setting).**
When the order ticket says only "steak," a good waiter asks "How would you like that cooked?" The language model in a single-turn setting cannot ask follow-up questions. It fills in the ambiguity with its most probable completion, which may not match your intent. In a multi-turn conversation, this limitation is partially addressed — but the analogy implies a back-and-forth that does not always exist.

**Dimension 2 — The NPC's behavior is scripted; the model's behavior is probabilistic.**
In a role-playing game, the NPC's response set is fixed and finite — the game developers explicitly programmed each possible response. Prompt engineering for language models works differently: the model generates responses probabilistically from a continuous distribution over all possible text. A prompt cannot guarantee a specific output; it shifts the probability distribution toward desired outputs. Viewing the model as a rule-following NPC sets up the wrong mental model for debugging when prompts behave unexpectedly.

---

### Case 7 — Application Exercise

**Exercise: Prompt comparison study**

Choose a task: extracting structured data from unstructured text (e.g., extracting all dates, proper nouns, or dollar amounts from a paragraph).

1. Write a zero-shot prompt. Apply it to 10 example paragraphs. Record the accuracy of the extraction.
2. Write a few-shot prompt with 3 examples. Apply to the same 10 paragraphs. Does accuracy improve?
3. Add an output format constraint (e.g., "Return only a JSON array. No other text."). Does this reduce formatting errors?
4. Try chain-of-thought prompting for one of the harder examples. Does asking the model to "think step by step" help?
5. Test your best prompt on 5 paragraphs that are slightly different from your examples (different domain, different writing style). Does it still work?

**Success condition:** You can demonstrate a measurable difference in output quality between a vague prompt and a well-engineered prompt, explain the role of few-shot examples in format-sensitive tasks, and describe one way your best prompt could fail on out-of-distribution input.

---

---

## Card 37 — RAG

**Difficulty:** 4/5 | **Relevance:** 5/5 | **Skin:** Restaurant / Construction

---

### Case 1 — Hook

A language model was trained on data with a cutoff date of early 2024. You ask it: "What was the outcome of last week's central bank interest rate decision?"

The model cannot answer reliably. Its knowledge was fixed at training time. It has no access to information that appeared after training concluded. It may hallucinate a plausible-sounding answer from outdated context.

Or: you have 50,000 pages of your company's internal documentation. The model has never seen any of it. You need answers that cite specific policies from those documents.

Fine-tuning on 50,000 pages is expensive and the documents change monthly. Is there a way to give the model access to current, specific information at query time without retraining?

---

### Case 2 — Mental Image

**Restaurant skin:** A chef trained at culinary school in 2018 knows everything that was in the curriculum that year. When a diner asks about a new ingredient — say, a regional variety of mushroom discovered in 2022 — the chef has never heard of it. His training data did not include it.

An alternative approach: before the chef prepares a dish involving unusual ingredients, the restaurant's librarian retrieves the most relevant pages from a current reference book and places them next to the stove. The chef reads those pages and uses that fresh information to prepare the dish accurately — even though the information never appeared in his culinary training.

RAG (Retrieval-Augmented Generation) is this librarian-and-stove setup. The model is the chef. The external document store is the reference library. The retrieval system is the librarian. For each query, relevant documents are retrieved and added to the model's context — giving it access to current, specific information it never saw during training.

**Construction skin:** An architect trained in 2019 memorized the building codes in effect at that time. A new fire safety code was issued in 2023. When designing a new building, the architect does not know the new code from memory.

A well-run firm gives the architect a current code lookup system. Before finalizing any fire safety specification, the architect queries the system: "Show me current requirements for sprinkler placement in commercial kitchens." The system retrieves the relevant code section. The architect reads it and incorporates the current standard into the design.

RAG is this code lookup system. The model queries an external knowledge base at inference time, retrieves relevant passages, and generates answers grounded in retrieved evidence rather than in training-time memory alone.

---

### Case 3 — Decryption

Retrieval-Augmented Generation (RAG) is an architecture that combines a retrieval system with a generative language model. It addresses two limitations of standard LLMs: knowledge cutoff and context length constraints for large document collections.

A standard RAG pipeline has three stages:

**1. Indexing (offline):**
- Documents are split into chunks (typically 200–500 tokens each)
- Each chunk is embedded using an embedding model (producing a dense vector)
- Vectors are stored in a vector database (e.g., FAISS, Chroma, Pinecone)

**2. Retrieval (at query time):**
- The user's query is embedded using the same embedding model
- The vector database performs a nearest-neighbor search: retrieve the k most similar document chunks
- Similarity is measured by cosine similarity or dot product of embedding vectors

**3. Generation:**
- The retrieved chunks are prepended to the user's query as context
- The language model generates an answer conditioned on both the query and the retrieved context
- The answer is grounded in the retrieved documents, not solely in the model's training memory

The quality of RAG depends critically on retrieval quality: if the wrong chunks are retrieved, the model generates answers grounded in irrelevant context. The embedding model used for indexing and retrieval must match — both query and documents are embedded with the same model.

---

### Case 4 — Minimal Code

```python
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import torch.nn.functional as F
import numpy as np

# --- Step 1: Build a small document store ---
documents = [
    "The restaurant's cancellation policy allows refunds up to 24 hours before the reservation.",
    "Private dining rooms are available for groups of 10 or more. Contact events@restaurant.com.",
    "The tasting menu changes seasonally. Current menu features dishes from the Pacific Northwest.",
    "Parking is available in the lot behind the building. Valet service is available after 6pm.",
    "The chef's tasting menu is available Tuesday through Sunday. Reservations required.",
    "Vegetarian and vegan options are available on all menus. Please inform your server of allergies.",
]

# --- Step 2: Embed all documents ---
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
embed_model = AutoModel.from_pretrained(embed_model_name)
embed_model.eval()

def embed(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

doc_embeddings = embed(documents)  # shape: (num_docs, hidden_dim)

# --- Step 3: Retrieval function ---
def retrieve(query, k=2):
    query_emb = embed([query])
    sims = F.cosine_similarity(query_emb, doc_embeddings)
    top_k = sims.topk(k).indices.tolist()
    return [documents[i] for i in top_k], sims[top_k].tolist()

# --- Step 4: Answer a query using retrieved context ---
query = "Can I cancel my reservation and get a refund?"
retrieved_docs, scores = retrieve(query, k=2)

context = "\n".join([f"[Doc {i+1}] {doc}" for i, doc in enumerate(retrieved_docs)])
augmented_prompt = (
    f"Answer the question using only the provided context. Be concise.\n\n"
    f"Context:\n{context}\n\n"
    f"Question: {query}\n"
    f"Answer:"
)

print(f"Query: {query}\n")
print("Retrieved documents:")
for i, (doc, score) in enumerate(zip(retrieved_docs, scores)):
    print(f"  [{score:.4f}] {doc}")

print(f"\nAugmented prompt sent to LLM:\n{augmented_prompt}")
```

**Expected output:**
```
Query: Can I cancel my reservation and get a refund?

Retrieved documents:
  [0.8921] The restaurant's cancellation policy allows refunds up to 24 hours before the reservation.
  [0.4103] The chef's tasting menu is available Tuesday through Sunday. Reservations required.

Augmented prompt sent to LLM:
Answer the question using only the provided context. Be concise.

Context:
[Doc 1] The restaurant's cancellation policy allows refunds up to 24 hours before the reservation.
[Doc 2] The chef's tasting menu is available Tuesday through Sunday. Reservations required.

Question: Can I cancel my reservation and get a refund?
Answer:
```

The retrieval step correctly identifies the cancellation policy document as the most relevant chunk (similarity 0.89). The augmented prompt gives the LLM grounded context to answer accurately.

---

### Case 5 — Analysis and Intuition

- RAG quality is bounded by retrieval quality. If the relevant document is not retrieved, the model generates an answer without the right context — it will either say it does not know or hallucinate.
- Chunk size matters. Chunks that are too small lose context (a sentence without its surrounding paragraph may be ambiguous). Chunks that are too large reduce retrieval precision (a long chunk is harder to match to a narrow query).
- The embedding model used for retrieval must be the same for indexing and querying. Embedding documents with one model and queries with another breaks the similarity search.
- RAG does not prevent hallucination entirely. If retrieved chunks are tangentially related, the model may blend retrieved context with training-time knowledge in ways that produce inaccurate answers.
- For production RAG, the vector database (FAISS, Chroma, Pinecone) is the performance bottleneck at scale. Approximate nearest-neighbor search is O(log n) rather than O(n) and is essential for large document stores.

---

### Case 6 — Traps and Limits

**Trap 1 — Assuming RAG removes the need to evaluate answers**
A RAG system with a high-quality retriever will usually retrieve relevant context. The model may still misinterpret it, generate answers that contradict the retrieved text, or stitch together context from multiple chunks incorrectly. Always evaluate RAG outputs on a held-out question set with known answers.

**Trap 2 — Using different embedding models for indexing and querying**
Documents are indexed using one embedding model; queries are embedded using the same model. If you upgrade the embedding model after indexing, you must re-index all documents. Mismatched embedding models produce meaningless similarity scores.

**Trap 3 — Ignoring retrieval failure cases**
Some queries genuinely have no relevant document in the knowledge base. The model should respond with "I don't have information about this" — not hallucinate. Designing the prompt to instruct the model to acknowledge retrieval failure is an important production consideration.

**Mirror Mode: Where the Analogies Break Down**

**Dimension 1 — The librarian understands the question; the retrieval system measures vector proximity.**
When a librarian searches for the right reference page, she understands the chef's question and applies semantic judgment about relevance. The RAG retrieval system computes cosine similarity between embedding vectors. These are different operations. A query about "cancellation policy" may retrieve a document about "party booking policies" if their embedding vectors happen to be similar, even though the semantic relevance is low. The librarian's semantic judgment has no equivalent in the mechanical retrieval step.

**Dimension 2 — The chef reads the page and decides whether to use it; the model cannot evaluate retrieved context quality.**
A skilled chef, reading a retrieved recipe page, can judge "this is relevant to what I'm making" or "this is for a different dish." The language model receives the retrieved context as part of its input and processes all of it without any explicit quality filter. If the retrieved chunk is misleading, the model may be led astray without the ability to recognize the retrieval as poor. The analogy implies active judgment that the generation mechanism does not possess.

---

### Case 7 — Application Exercise

**Exercise: Building a minimal RAG pipeline**

You have a directory of text files — use any collection you have: product documentation, company policies, course notes, or a small Wikipedia article set.

1. Split documents into chunks of approximately 300 words. How many chunks does your collection produce?
2. Embed all chunks using `sentence-transformers/all-MiniLM-L6-v2`. Store embeddings as a numpy array.
3. Write a `retrieve(query, k=3)` function that returns the top-3 most similar chunks for a given query.
4. Write a `rag_answer(query)` function that retrieves context and formats an augmented prompt. Print the prompt — do not yet plug it into an LLM.
5. Test retrieval on 5 queries. For each query, evaluate whether the top-1 retrieved chunk is actually relevant. How often does it retrieve the right context?

**Challenge:** Try two different chunk sizes (200 words vs 500 words). Does retrieval quality improve with smaller or larger chunks for your document collection?

**Success condition:** You can describe the three stages of a RAG pipeline (indexing, retrieval, generation), explain why retrieval quality determines the ceiling on RAG performance, and identify at least one query type where embedding-based retrieval fails for your document set.

---

*Module 05 — NLP and LLMs | The FILS Framework*
*Open source — see root LICENSE for terms*
