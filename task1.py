"""
Task 1: Data Preparation and Vocabulary Size Selection

This module loads the Brown corpus, calculates cumulative coverage,
plots coverage vs vocabulary size, and selects the appropriate vocabulary size.
"""

import nltk
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Download Brown corpus if not already downloaded
try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown', quiet=True)

from nltk.corpus import brown


def load_brown_corpus():
    """
    Load the Brown corpus and return all words.

    Returns:
        list: List of all words from the Brown corpus
    """
    # Get all words from Brown corpus
    words = brown.words()
    return words


def calculate_coverage(words, vocab_size):
    """
    Calculate cumulative coverage for a given vocabulary size.

    Coverage formula: coverage(k) = sum(frequencies of top-k words) / sum(all frequencies)

    This tells us what percentage of the corpus is covered by the top-k most frequent words.
    For example, if coverage(100) = 0.5, it means the top 100 words account for 50% of all
    word occurrences in the corpus.

    Args:
        words: List of words from the corpus
        vocab_size: Number of top-k tokens to include in vocabulary

    Returns:
        float: Cumulative coverage (0-1)
    """
    # Step 1: Count how many times each word appears in the corpus
    # Counter creates a dictionary: {word: frequency}
    # Example: {"the": 5000, "cat": 100, "dog": 200}
    word_freq = Counter(words)

    # Step 2: Calculate total frequency - sum of all word occurrences
    # This is the denominator in our coverage formula
    # Example: if we have 1,000,000 total word occurrences
    total_freq = sum(word_freq.values())

    # Step 3: Get the top-k most frequent words
    # most_common(vocab_size) returns a list of tuples: [(word, freq), ...]
    # Example: [("the", 5000), ("cat", 100), ("dog", 200)] for vocab_size=3
    top_k_words = word_freq.most_common(vocab_size)

    # Step 4: Sum up the frequencies of these top-k words
    # This is the numerator in our coverage formula
    # We extract just the frequencies (second element of each tuple) and sum them
    top_k_freq = sum(freq for _, freq in top_k_words)

    # Step 5: Calculate coverage as a ratio
    # If top_k_freq = 500,000 and total_freq = 1,000,000, then coverage = 0.5 (50%)
    coverage = top_k_freq / total_freq if total_freq > 0 else 0.0

    return coverage


def plot_coverage_vs_vocab_size(words, max_vocab_size=10000):
    """
    Plot cumulative coverage vs vocabulary size.

    Args:
        words: List of words from the corpus
        max_vocab_size: Maximum vocabulary size to plot (default: 10000)

    Returns:
        tuple: (vocab_sizes, coverages) - arrays of vocabulary sizes and coverages
    """
    # Count word frequencies
    word_freq = Counter(words)
    total_unique_words = len(word_freq)

    # Calculate coverage for different vocabulary sizes
    # We'll store vocabulary sizes and their corresponding coverages
    vocab_sizes = []
    coverages = []

    # Adaptive step size: use smaller steps when coverage changes rapidly (small vocab sizes),
    # and larger steps when coverage plateaus (large vocab sizes)
    # This gives us more data points where the curve is interesting (steep) and fewer
    # where it's flat, making the plot more informative
    current_size = 0

    while current_size < min(max_vocab_size, total_unique_words):
        # Determine step size based on current vocabulary size
        # Small vocab sizes: coverage changes quickly, so we sample more frequently (step=1)
        if current_size < 100:
            step = 1
        # Medium-small: still changing, sample every 5 words
        elif current_size < 500:
            step = 5
        # Medium: changes slower, sample every 10 words
        elif current_size < 1000:
            step = 10
        # Medium-large: changes very slowly, sample every 50 words
        elif current_size < 5000:
            step = 50
        # Large: coverage barely changes, sample every 100 words
        else:
            step = 100

        # Move to next vocabulary size
        current_size += step

        # Don't exceed the total number of unique words
        if current_size > total_unique_words:
            current_size = total_unique_words

        # Calculate coverage for this vocabulary size
        coverage = calculate_coverage(words, current_size)

        # Store the data point
        vocab_sizes.append(current_size)
        coverages.append(coverage)

        # Stop if we've covered all unique words
        if current_size >= total_unique_words:
            break

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(vocab_sizes, coverages, 'b-', linewidth=2)
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% Coverage')
    plt.xlabel('Vocabulary Size', fontsize=12)
    plt.ylabel('Cumulative Coverage', fontsize=12)
    plt.title('Cumulative Coverage vs Vocabulary Size (Brown Corpus)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return np.array(vocab_sizes), np.array(coverages)


def find_appropriate_vocab_size(words, target_coverage=0.9):
    """
    Find the minimal vocabulary size that covers at least target_coverage of words.

    Uses binary search to efficiently find the answer. Instead of checking every possible
    vocabulary size (which could be thousands), we use binary search to find the answer
    in O(log n) time.

    Args:
        words: List of words from the corpus
        target_coverage: Target coverage threshold (default: 0.9, meaning 90%)

    Returns:
        int: Minimal vocabulary size that achieves target coverage
    """
    # Step 1: Count word frequencies and get total unique words
    word_freq = Counter(words)
    total_unique_words = len(word_freq)

    # Step 2: Binary search setup
    # We search in the range [1, total_unique_words]
    # left = smallest possible vocab size (1 word)
    # right = largest possible vocab size (all unique words)
    left, right = 1, total_unique_words
    result = total_unique_words  # Default: use all words if we can't find a smaller solution

    # Step 3: Binary search loop
    # The idea: coverage increases as vocab_size increases (more words = more coverage)
    # So we can use binary search to find the smallest vocab_size with coverage >= 0.9
    while left <= right:
        # Check the middle point
        mid = (left + right) // 2

        # Calculate coverage for this vocabulary size
        coverage = calculate_coverage(words, mid)

        # If we've reached our target coverage (e.g., >= 90%)
        if coverage >= target_coverage:
            # This vocab_size works! But maybe a smaller one also works?
            # So we record this as a candidate and search left (smaller vocab sizes)
            result = mid
            right = mid - 1  # Try smaller vocab sizes
        else:
            # Coverage is too low, we need more words
            # So we search right (larger vocab sizes)
            left = mid + 1

    # Return the minimal vocabulary size that achieves target coverage
    return result


if __name__ == "__main__":
    # Load Brown corpus
    print("Loading Brown corpus...")
    words = load_brown_corpus()
    print(f"Total words in corpus: {len(words):,}")
    print(f"Unique words in corpus: {len(set(words)):,}")

    # Plot coverage vs vocabulary size
    print("\nPlotting cumulative coverage vs vocabulary size...")
    vocab_sizes, coverages = plot_coverage_vs_vocab_size(words)

    # Find appropriate vocabulary size
    print("\nFinding appropriate vocabulary size (90% coverage)...")
    vocab_size = find_appropriate_vocab_size(words, target_coverage=0.9)
    actual_coverage = calculate_coverage(words, vocab_size)

    print(f"\nAppropriate vocabulary size: {vocab_size}")
    print(f"Actual coverage achieved: {actual_coverage:.4f} ({actual_coverage*100:.2f}%)")

    # Answer questions
    print("\n" + "="*60)
    print("ANSWERS TO QUESTIONS:")
    print("="*60)
    print("\n1. Why does the coverage slow down as vocabulary size increases?")
    print("   Answer: The coverage slows down because of Zipf's law - word")
    print("   frequencies follow a power-law distribution. The most frequent")
    print("   words (top-ranked) account for a large portion of the corpus,")
    print("   while less frequent words (lower-ranked) appear much less often.")
    print("   As we add more words to the vocabulary, we're adding words with")
    print("   progressively lower frequencies, so each additional word contributes")
    print("   less to the total coverage.")

    print("\n2. Which empirical law explains the slowing down increase of coverage?")
    print("   Answer: Zipf's law (or Zipf's distribution) explains this phenomenon.")
    print("   Zipf's law states that the frequency of a word is inversely")
    print("   proportional to its rank in the frequency table. This means that")
    print("   a small number of high-frequency words account for most of the")
    print("   corpus, while many low-frequency words account for very little.")
    print("   This power-law distribution causes the coverage curve to flatten")
    print("   as vocabulary size increases.")
