# Natural Language Processing - Assignment 1: Tokenization Explained

## 📚 Table of Contents
1. [What is This Assignment About?](#what-is-this-assignment-about)
2. [Understanding Tokenization](#understanding-tokenization)
3. [Task 1: Finding the Right Vocabulary Size](#task-1-finding-the-right-vocabulary-size)
4. [Task 2: Building a BPE Tokenizer](#task-2-building-a-bpe-tokenizer)
5. [Task 3: Analyzing the Tokenizer](#task-3-analyzing-the-tokenizer)
6. [Key Concepts Explained](#key-concepts-explained)
7. [How to Run the Code](#how-to-run-the-code)

---

## What is This Assignment About?

This assignment teaches you about **tokenization** - the process of breaking text into smaller pieces (tokens) that computers can understand. Think of it like cutting a sentence into words, but sometimes we need to cut words into even smaller pieces!

### Why Do We Need Tokenization?

Imagine you're teaching a computer to understand English. The computer doesn't know what words mean, so we need to:
1. **Break text into tokens** (words or subwords)
2. **Assign each token a number** (so the computer can work with numbers)
3. **Handle unknown words** (words the computer has never seen before)

### The Problem with Simple Word Tokenization

If we just split text by spaces, we face problems:
- **Vocabulary explosion**: Too many unique words (millions!)
- **Unknown words**: New words not in our vocabulary
- **Rare words**: Words that appear only once or twice

**Example**: If we see "unhappiness" but never saw it before, we can't handle it. But if we break it into "un-" + "happy" + "-ness", we can understand it!

---

## Understanding Tokenization

### Types of Tokenization

1. **Word-level**: Split by spaces
   - "Hello world" → ["Hello", "world"]
   - Problem: Can't handle new words

2. **Character-level**: Split into individual characters
   - "Hello" → ["H", "e", "l", "l", "o"]
   - Problem: Too many tokens, loses meaning

3. **Subword-level (BPE)**: Split into meaningful pieces
   - "Hello" → ["Hel", "lo"]
   - **Best of both worlds!**

### What is BPE (Byte-Pair Encoding)?

BPE is a smart way to tokenize text that:
- Starts with characters
- Learns common patterns (like "th", "ing", "ed")
- Creates a vocabulary of subwords
- Can handle any word, even if never seen before!

**Real-world example**: GPT models use BPE! It's why they can understand new words.

---

## Task 1: Finding the Right Vocabulary Size

### What Are We Doing?

We want to know: **How many words do we need to cover 90% of the text?**

### The Problem

In any text corpus (like books, articles), word frequencies follow **Zipf's Law**:
- A few words appear **very frequently** (like "the", "is", "a")
- Most words appear **rarely** (like "xylophone", "serendipity")

**Example from Brown Corpus**:
- Top 10 words: ~25% of all text
- Top 100 words: ~50% of all text
- Top 1000 words: ~75% of all text
- Top 10,000 words: ~90% of all text

### What is Coverage?

**Coverage** = (How many times top-k words appear) / (Total word occurrences)

**Example**:
- Total words in corpus: 1,000,000
- Top 100 words appear: 500,000 times
- Coverage(100) = 500,000 / 1,000,000 = 0.5 (50%)

### Step-by-Step: What Task 1 Does

1. **Load Brown Corpus**
   ```python
   words = brown.words()  # Get all words from the corpus
   ```
   - Brown Corpus: A collection of English texts from 1961
   - Contains ~1 million words
   - ~50,000 unique words

2. **Count Word Frequencies**
   ```python
   word_freq = Counter(words)
   # Result: {"the": 50000, "cat": 100, "xylophone": 1, ...}
   ```
   - Counts how many times each word appears
   - Most frequent: "the", "of", "and", etc.
   - Least frequent: rare words appearing once

3. **Calculate Coverage for Different Vocabulary Sizes**
   ```python
   coverage(k) = sum(frequencies of top-k words) / sum(all frequencies)
   ```
   - For k=100: How much of corpus is covered by top 100 words?
   - For k=1000: How much of corpus is covered by top 1000 words?
   - And so on...

4. **Plot the Results**
   - X-axis: Vocabulary size (number of words)
   - Y-axis: Coverage (0 to 1, or 0% to 100%)
   - Shows how coverage increases as we add more words
   - **Key observation**: Curve flattens (diminishing returns)

5. **Find Minimal Vocabulary Size for 90% Coverage**
   - Use binary search (efficient algorithm)
   - Find smallest k where coverage(k) ≥ 0.9
   - **Result**: Usually around 5,000-10,000 words

### Why Does Coverage Slow Down?

**Answer**: **Zipf's Law**

- Word frequency is inversely proportional to rank
- Formula: frequency(rank) ≈ constant / rank
- This means:
  - Rank 1 word: appears ~50,000 times
  - Rank 10 word: appears ~5,000 times
  - Rank 100 word: appears ~500 times
  - Rank 1000 word: appears ~50 times

**Visual Example**:
```
Rank 1-10:    ████████████████████ (huge frequency)
Rank 11-100:  ████████ (medium frequency)
Rank 101-1000: ███ (small frequency)
Rank 1001+:    █ (tiny frequency)
```

As we add more words, we're adding words with progressively lower frequencies, so each new word contributes less to total coverage.

---

## Task 2: Building a BPE Tokenizer

### What is BPE Tokenization?

BPE (Byte-Pair Encoding) is an algorithm that:
1. Starts with characters
2. Finds most frequent pairs of consecutive characters/subwords
3. Merges them into a single token
4. Repeats until desired vocabulary size

### The BPE Algorithm: Step by Step

#### Initialization

Start with character-level vocabulary:
```
Vocabulary: {a, b, c, ..., z, </w>}
Words: ["hello</w>", "world</w>"]
```

#### Training Process

**Iteration 1**:
- Count all pairs: ("h","e"), ("e","l"), ("l","l"), ("l","o"), ...
- Find most frequent: ("l","l") appears 2 times
- Merge: "ll" becomes a single token
- New vocabulary: {a, b, c, ..., z, "ll", </w>}
- Updated words: ["he" + "ll" + "o</w>", "wor" + "l" + "d</w>"]

**Iteration 2**:
- Count pairs again (now "ll" is a single token)
- Find most frequent: maybe ("e","l") or ("o","r")
- Merge the most frequent pair
- Continue...

**After Many Iterations**:
- Vocabulary contains: characters + common subwords
- Examples: "th", "ing", "ed", "tion", "the", "ing</w>", etc.

### Detailed Example

**Training Data**: ["low</w>", "lower</w>", "newest</w>", "widest</w>"]

**Step 1: Character-level**
```
l o w </w>
l o w e r </w>
n e w e s t </w>
w i d e s t </w>
```

**Step 2: Count pairs**
- ("e","s"): 2 times (in "newest" and "widest")
- ("e","s"): Most frequent!

**Step 3: Merge "es"**
```
l o w </w>
l o w e r </w>
n e w es t </w>
w i d es t </w>
```

**Step 4: Count pairs again**
- ("es","t"): 2 times
- Merge "est"

**Step 5: Continue until vocabulary size reached**

### How Tokenization Works (After Training)

Given a new word: "lowest"

1. Start with characters: `["l", "o", "w", "e", "s", "t", "</w>"]`
2. Apply merges in order:
   - Merge "es" → `["l", "o", "w", "es", "t", "</w>"]`
   - Merge "est" → `["l", "o", "w", "est", "</w>"]`
   - No more merges possible
3. Result: `["low", "est</w>"]` (2 tokens!)

### Why BPE is Powerful

1. **Handles Unknown Words**: Can tokenize any word using learned subwords
2. **Efficient**: Vocabulary size is controlled (e.g., 10,000 tokens)
3. **Meaningful**: Learns common patterns (prefixes, suffixes, roots)
4. **Language-agnostic**: Works for any language!

### Key Components of Our BPE Implementation

1. **`_pre_tokenize()`**: Splits text into words, adds `</w>` marker
2. **`_get_stats()`**: Counts frequency of all pairs
3. **`_merge_vocab()`**: Merges a pair in all words
4. **`train()`**: Main training loop (repeats merge process)
5. **`_apply_bpe()`**: Applies learned merges to tokenize new text
6. **`tokenize()`**: Public method to tokenize any text

---

## Task 3: Analyzing the Tokenizer

### What Are We Measuring?

After training the tokenizer, we want to know:
1. **Fertility**: How many tokens per word?
2. **Tokenized Length**: How long are tokenized sentences?

### What is Fertility?

**Fertility** = Number of tokens / Number of words

**Interpretation**:
- Fertility = 1.0: Each word becomes exactly 1 token (perfect!)
- Fertility = 1.5: Each word becomes 1.5 tokens on average
- Fertility = 2.0: Each word becomes 2 tokens on average

**Example**:
```
Original: "Hello world" (2 words)
Tokenized: ["Hel", "lo</w>", "world</w>"] (3 tokens)
Fertility = 3 / 2 = 1.5
```

**Why Does Fertility Matter?**
- Lower fertility = fewer tokens = faster processing
- Higher fertility = more tokens = more detailed representation
- BPE typically has fertility around 1.2-1.5 (good balance!)

### What is Tokenized Length?

Simply: **How many tokens are in a tokenized sentence?**

**Example**:
```
Original: "The quick brown fox jumps over the lazy dog."
Words: 9 words
Tokens: ["The</w>", "quick</w>", "brown</w>", "fox</w>", "jumps</w>", "over</w>", "the</w>", "lazy</w>", "dog</w>"]
Length: 9 tokens
```

But with BPE:
```
Tokens: ["The</w>", "qu", "ick</w>", "brown</w>", "fox</w>", "jumps</w>", "over</w>", "the</w>", "lazy</w>", "dog</w>"]
Length: 10 tokens
```

### What Task 3 Does

1. **Train BPE Tokenizer**
   - Uses vocabulary size from Task 1 (e.g., 5,000)
   - Trains on entire Brown corpus
   - Learns common subword patterns

2. **Get Sample Sentences**
   - Takes 1,000 sentences from Brown corpus
   - These are used for evaluation (not training)

3. **Calculate Fertility**
   - For each sentence:
     - Count words
     - Tokenize and count tokens
     - Calculate fertility = tokens / words
   - Calculate mean and standard deviation

4. **Calculate Tokenized Length**
   - For each sentence:
     - Tokenize it
     - Count tokens
   - Calculate mean and standard deviation

5. **Report Statistics**
   - Mean fertility: e.g., 1.35 (each word becomes 1.35 tokens)
   - Std fertility: e.g., 0.15 (variation)
   - Mean length: e.g., 12.5 tokens per sentence
   - Std length: e.g., 5.2 tokens (variation)

---

## Key Concepts Explained

### 1. Zipf's Law

**What it is**: A mathematical law describing word frequency distribution.

**The Law**: The frequency of a word is inversely proportional to its rank.

**Formula**: `frequency(rank) ≈ constant / rank`

**Example**:
- Rank 1 word ("the"): frequency = 50,000
- Rank 2 word ("of"): frequency = 25,000
- Rank 10 word: frequency = 5,000
- Rank 100 word: frequency = 500

**Why it matters**: Explains why coverage slows down as vocabulary size increases.

### 2. Cumulative Coverage

**What it is**: The percentage of text covered by the top-k most frequent words.

**Formula**:
```
coverage(k) = Σ(frequencies of top-k words) / Σ(all frequencies)
```

**Example**:
- Top 100 words cover 50% of text
- Top 1,000 words cover 75% of text
- Top 10,000 words cover 90% of text

**Why it matters**: Helps us choose vocabulary size efficiently.

### 3. Binary Search

**What it is**: An efficient algorithm to find a value in a sorted range.

**How it works**:
1. Check the middle value
2. If too small, search right half
3. If too large, search left half
4. Repeat until found

**Example**: Finding vocabulary size for 90% coverage
- Range: 1 to 50,000 words
- Check 25,000: coverage = 95% (too high, try smaller)
- Check 12,500: coverage = 88% (too low, try larger)
- Check 18,750: coverage = 92% (too high, try smaller)
- Continue until find minimal size with ≥90% coverage

**Why it matters**: Much faster than checking every possible size!

### 4. Subword Tokenization

**What it is**: Breaking words into smaller, meaningful pieces.

**Types**:
- **Prefixes**: "un-", "re-", "pre-"
- **Suffixes**: "-ing", "-ed", "-tion"
- **Roots**: "happy", "work", "play"
- **Common pairs**: "th", "er", "ly"

**Example**: "unhappiness"
- Word-level: 1 token (but might be unknown!)
- Character-level: 12 tokens (too many!)
- Subword-level: ["un-", "happy", "-ness"] (3 tokens, all known!)

**Why it matters**: Handles unknown words and rare words efficiently.

### 5. Vocabulary Size Trade-offs

**Small Vocabulary (e.g., 1,000 tokens)**:
- ✅ Fast processing
- ✅ Less memory
- ❌ High fertility (many tokens per word)
- ❌ May miss important patterns

**Large Vocabulary (e.g., 50,000 tokens)**:
- ✅ Low fertility (few tokens per word)
- ✅ Captures more patterns
- ❌ Slower processing
- ❌ More memory needed

**Sweet Spot (e.g., 5,000-10,000 tokens)**:
- ✅ Good balance
- ✅ Covers 90% of text
- ✅ Reasonable fertility (~1.3)
- ✅ Efficient processing

---

## How to Run the Code

### Prerequisites

Install required packages:
```bash
pip install -r requirements.txt
```

This installs:
- `nltk`: For Brown corpus
- `matplotlib`: For plotting
- `numpy`: For numerical operations

### Running Individual Tasks

#### Task 1
```bash
python task1.py
```
Or run the notebook cell for Task 1.

**What you'll see**:
- Loading message
- Total words and unique words count
- A plot showing coverage vs vocabulary size
- The appropriate vocabulary size (for 90% coverage)
- Answers to questions about Zipf's law

#### Task 2
```bash
python task2.py
```
Or run the notebook cell for Task 2.

**What you'll see**:
- BPE tokenizer implementation
- Example training on sample texts
- Test tokenization showing how words are split

#### Task 3
```bash
python task3.py
```
Or run the notebook cell for Task 3.

**What you'll see**:
- Training progress
- Vocabulary size and number of merges
- Fertility statistics (mean and std)
- Tokenized sentence length statistics (mean and std)

### Running the Complete Notebook

1. Open `Assignment1 (2).ipynb` in Jupyter Notebook
2. Run cells sequentially (Shift + Enter)
3. Each task builds on the previous one:
   - Task 1 finds vocabulary size
   - Task 2 implements BPE
   - Task 3 uses vocabulary size from Task 1 to train BPE

### Expected Outputs

**Task 1**:
```
Loading Brown corpus...
Total words in corpus: 1,161,192
Unique words in corpus: 56,057

Appropriate vocabulary size: 8,234
Actual coverage achieved: 0.9001 (90.01%)
```

**Task 2**:
```
BPE Tokenizer Implementation
Vocabulary size: 50
Number of merges: 25

Test text: 'The quick brown fox'
Tokens: ['th', 'e</w>', 'qu', 'ick</w>', 'brown</w>', 'fox</w>']
```

**Task 3**:
```
Vocabulary Size: 8,234
Fertility (tokens/word):
  Mean: 1.3245
  Std:  0.1523

Tokenized Sentence Length:
  Mean: 14.23
  Std:  6.45
```

---

## Common Questions

### Q: Why do we need 90% coverage? Why not 100%?

**A**: 100% coverage would require including every single unique word (50,000+ words). This is inefficient. 90% coverage gives us most of the text with a much smaller vocabulary. The remaining 10% consists of very rare words that can be handled by subword tokenization.

### Q: What happens to words not in the vocabulary?

**A**: BPE handles this! Even if a word isn't in the vocabulary, BPE can break it into subwords that are in the vocabulary. For example, "xylophone" might become ["xy", "lo", "phone</w>"], all of which are in the vocabulary.

### Q: Why does fertility vary between sentences?

**A**: Different sentences have different word lengths and complexities:
- Short, common words: "the cat" → low fertility (~1.0)
- Long, rare words: "xylophone" → high fertility (~3.0)
- Average sentence: fertility around 1.3-1.5

### Q: How is BPE different from just splitting words?

**A**:
- Simple splitting: "hello" → ["he", "llo"] (arbitrary)
- BPE: "hello" → ["hel", "lo</w>"] (learned from data, meaningful)

BPE learns which splits are most useful based on actual text data.

### Q: Can BPE work for other languages?

**A**: Yes! BPE is language-agnostic. It just learns patterns from whatever text you give it. It works for:
- English
- Chinese (learns character combinations)
- Arabic (learns letter combinations)
- Any language!

---

## Summary

This assignment teaches you:

1. **Word Frequency Analysis**: How words are distributed (Zipf's law)
2. **Vocabulary Selection**: Finding the right vocabulary size
3. **BPE Algorithm**: How subword tokenization works
4. **Tokenizer Evaluation**: Measuring tokenizer performance

**Key Takeaways**:
- Most text is covered by a small number of frequent words
- BPE is a powerful way to handle unknown words
- Subword tokenization balances efficiency and coverage
- Tokenizer performance can be measured with fertility and length statistics

**Real-world Applications**:
- GPT models use BPE-like tokenization
- Machine translation systems
- Text generation models
- Any NLP system that needs to handle diverse vocabulary

---

## Additional Resources

- [BPE Paper](https://arxiv.org/pdf/1508.07909): Original BPE research paper
- [Zipf's Law](https://en.wikipedia.org/wiki/Zipf%27s_law): More about word frequency distributions
- [Brown Corpus](https://en.wikipedia.org/wiki/Brown_Corpus): Information about the corpus used
- [Tokenization Guide](https://huggingface.co/docs/transformers/tokenizer_summary): Modern tokenization techniques

---

**Good luck with your assignment! 🚀**

If you have questions, refer back to the code comments or this explanation document.
