"""
Task 3: Tokenizer Training and Analysis

This module trains the BPE tokenizer on the Brown corpus and calculates
statistics on fertility and tokenized sentence length.
"""

import nltk
import numpy as np
from task1 import load_brown_corpus, find_appropriate_vocab_size
from task2 import BPETokenizer

# Download Brown corpus if needed
try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown', quiet=True)

from nltk.corpus import brown


def get_sentences_from_brown(num_sentences=1000):
    """
    Get a specified number of sentences from the Brown corpus.

    Args:
        num_sentences: Number of sentences to retrieve (default: 1000)

    Returns:
        List of sentence strings
    """
    sentences = brown.sents()[:num_sentences]
    # Join tokens into sentence strings
    sentence_strings = [' '.join(sent) for sent in sentences]
    return sentence_strings


def calculate_fertility(tokenizer: BPETokenizer, sentences: list) -> np.ndarray:
    """
    Calculate fertility for each sentence.

    Fertility measures how many tokens the tokenizer produces per word.
    It's an important metric because:
    - Fertility = 1.0 means each word becomes exactly 1 token (perfect compression)
    - Fertility > 1.0 means words are split into multiple tokens (subword tokenization)
    - Higher fertility = more tokens = potentially more information but also more computation

    Formula: fertility = num_tokens / num_words

    Example:
        Sentence: "Hello world" (2 words)
        Tokens: ["Hello", "world"] (2 tokens) -> fertility = 2/2 = 1.0
        OR
        Tokens: ["Hel", "lo", "world"] (3 tokens) -> fertility = 3/2 = 1.5

    Args:
        tokenizer: Trained BPETokenizer instance
        sentences: List of sentence strings

    Returns:
        Array of fertility values for each sentence
    """
    fertilities = []

    for sentence in sentences:
        # Step 1: Count original words in the sentence
        # Split by whitespace gives us word count
        # Example: "Hello world!" -> ["Hello", "world!"] -> 2 words
        num_words = len(sentence.split())

        # Step 2: Tokenize the sentence using BPE
        # This splits words into subword tokens
        # Example: "Hello world!" -> ["Hel", "lo</w>", "world</w>", "!</w>"] -> 4 tokens
        tokens = tokenizer.tokenize(sentence)
        num_tokens = len(tokens)

        # Step 3: Calculate fertility ratio
        # If num_words = 2 and num_tokens = 4, fertility = 4/2 = 2.0
        # This means on average, each word becomes 2 tokens
        if num_words > 0:
            fertility = num_tokens / num_words
        else:
            # Handle edge case: empty sentence
            fertility = 0.0

        fertilities.append(fertility)

    return np.array(fertilities)


def calculate_tokenized_length(tokenizer: BPETokenizer, sentences: list) -> np.ndarray:
    """
    Calculate the length (number of tokens) of each tokenized sentence.

    Args:
        tokenizer: Trained BPETokenizer instance
        sentences: List of sentence strings

    Returns:
        Array of tokenized sentence lengths
    """
    lengths = []

    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        lengths.append(len(tokens))

    return np.array(lengths)


def train_and_analyze(vocab_size: int, num_samples: int = 1000):
    """
    Train the BPE tokenizer and perform analysis.

    Args:
        vocab_size: Vocabulary size for the tokenizer
        num_samples: Number of sentences to use for analysis (default: 1000)

    Returns:
        Dictionary containing statistics
    """
    print("=" * 60)
    print("Task 3: Tokenizer Training and Analysis")
    print("=" * 60)

    # Load Brown corpus for training
    print("\n1. Loading Brown corpus for training...")
    words = load_brown_corpus()

    # Convert words to sentences for training
    # We'll use all sentences from Brown corpus for training
    print("   Converting to sentences...")
    all_sentences = brown.sents()
    training_texts = [' '.join(sent) for sent in all_sentences]

    print(f"   Total training sentences: {len(training_texts):,}")

    # Initialize and train tokenizer
    print(f"\n2. Training BPE tokenizer with vocabulary size: {vocab_size}")
    tokenizer = BPETokenizer()
    tokenizer.train(training_texts, vocab_size=vocab_size)

    print(f"   Actual vocabulary size: {len(tokenizer.vocab)}")
    print(f"   Number of BPE merges: {len(tokenizer.merges)}")

    # Get sample sentences for analysis
    print(f"\n3. Getting {num_samples} sample sentences for analysis...")
    sample_sentences = get_sentences_from_brown(num_sentences=num_samples)

    # Calculate fertility
    print("\n4. Calculating fertility statistics...")
    fertilities = calculate_fertility(tokenizer, sample_sentences)
    fertility_mean = np.mean(fertilities)
    fertility_std = np.std(fertilities)

    print(f"   Mean fertility: {fertility_mean:.4f}")
    print(f"   Std fertility: {fertility_std:.4f}")

    # Calculate tokenized sentence length
    print("\n5. Calculating tokenized sentence length statistics...")
    lengths = calculate_tokenized_length(tokenizer, sample_sentences)
    length_mean = np.mean(lengths)
    length_std = np.std(lengths)

    print(f"   Mean tokenized sentence length: {length_mean:.2f}")
    print(f"   Std tokenized sentence length: {length_std:.2f}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Vocabulary Size: {len(tokenizer.vocab)}")
    print(f"\nFertility (tokens/word):")
    print(f"  Mean: {fertility_mean:.4f}")
    print(f"  Std:  {fertility_std:.4f}")
    print(f"\nTokenized Sentence Length:")
    print(f"  Mean: {length_mean:.2f}")
    print(f"  Std:  {length_std:.2f}")

    return {
        'tokenizer': tokenizer,
        'fertility_mean': fertility_mean,
        'fertility_std': fertility_std,
        'length_mean': length_mean,
        'length_std': length_std,
        'fertilities': fertilities,
        'lengths': lengths
    }


if __name__ == "__main__":
    # First, find appropriate vocabulary size from Task 1
    print("Finding appropriate vocabulary size (90% coverage)...")
    words = load_brown_corpus()
    vocab_size = find_appropriate_vocab_size(words, target_coverage=0.9)
    print(f"Using vocabulary size: {vocab_size}\n")

    # Train and analyze
    results = train_and_analyze(vocab_size=vocab_size, num_samples=1000)
