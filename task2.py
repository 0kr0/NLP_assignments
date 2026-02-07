"""
Task 2: Implement Byte-Pair Encoding (BPE) Tokenizer

This module implements a Byte-Pair Encoding (BPE) tokenizer as described in
the paper: "Neural Machine Translation of Rare Words with Subword Units"
(https://arxiv.org/pdf/1508.07909)
"""

import re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict


class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) Tokenizer implementation.

    BPE is a data compression algorithm that iteratively replaces the most frequent
    pair of consecutive bytes (or characters) with a single, unused byte. In NLP,
    it's adapted to work with subword units instead of bytes.
    """

    def __init__(self):
        """Initialize the BPE tokenizer."""
        # Vocabulary: maps token to its index
        self.vocab = {}

        # Reverse vocabulary: maps index to token
        self.vocab_reverse = {}

        # BPE merges: list of (token1, token2) pairs that were merged
        self.merges = []

        # Special tokens
        self.unk_token = "<UNK>"
        self.end_of_word_token = "</w>"  # Marks end of word

    def _pre_tokenize(self, text: str) -> List[str]:
        """
        Pre-tokenize text into words and add end-of-word markers.

        This is the first step in BPE tokenization. We need to:
        1. Split text into words (pre-tokenization)
        2. Add a special marker to indicate word boundaries (</w>)

        Why add </w>? BPE works on character/subword level, so we need to know where
        words end. Otherwise, "cat" and "cats" might merge incorrectly with other words.

        Args:
            text: Input text string

        Returns:
            List of pre-tokenized words with end-of-word markers
        """
        # Step 1: Split text into words
        # \S+ matches one or more non-whitespace characters (words)
        # .lower() converts to lowercase for consistency
        # Example: "Hello world!" -> ["hello", "world!"]
        words = re.findall(r'\S+', text.lower())

        # Step 2: Add end-of-word marker to each word
        # This marker helps BPE know where words end during merging
        # Example: ["hello", "world!"] -> ["hello</w>", "world!</w>"]
        # Later, BPE will learn that "lo</w>" is different from "lo" (which could be part of "long")
        words = [word + self.end_of_word_token for word in words]

        return words

    def _get_word_freqs(self, texts: List[str]) -> Dict[str, int]:
        """
        Get word frequencies from a list of texts.

        Args:
            texts: List of text strings

        Returns:
            Dictionary mapping words (with </w>) to their frequencies
        """
        word_freqs = Counter()

        for text in texts:
            words = self._pre_tokenize(text)
            word_freqs.update(words)

        return dict(word_freqs)

    def _get_stats(self, word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """
        Get statistics of pairs of consecutive symbols.

        This is the core of BPE: we need to find which pairs of consecutive characters/subwords
        appear most frequently. These frequent pairs will be merged into a single token.

        Example: If "th" appears 1000 times and "he" appears 800 times, we'll merge "th" first.

        Args:
            word_freqs: Dictionary mapping words to their frequencies
                       Example: {"hello</w>": 5, "world</w>": 3}

        Returns:
            Dictionary mapping pairs (char1, char2) to their total frequency
            Example: {("h", "e"): 5, ("e", "l"): 5, ("l", "l"): 5, ("l", "o"): 5, ...}
        """
        # Initialize a dictionary to count pair frequencies
        # defaultdict(int) means if a key doesn't exist, it defaults to 0
        pairs = defaultdict(int)

        # For each word and its frequency in the corpus
        for word, freq in word_freqs.items():
            # Step 1: Split word into individual characters/symbols
            # Example: "hello</w>" -> ["h", "e", "l", "l", "o", "<", "/", "w", ">"]
            symbols = list(word)

            # Step 2: Count all pairs of consecutive symbols
            # We iterate through adjacent pairs: (symbols[0], symbols[1]), (symbols[1], symbols[2]), etc.
            # Example: for "hello</w>", we get pairs: ("h","e"), ("e","l"), ("l","l"), ("l","o"), ...
            for i in range(len(symbols) - 1):
                # Increment the count for this pair by the word's frequency
                # If "hello</w>" appears 5 times, each pair in it gets +5
                pairs[(symbols[i], symbols[i + 1])] += freq

        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], word_freqs: Dict[str, int]) -> Dict[str, int]:
        """
        Merge the most frequent pair in the vocabulary.

        This function takes a pair like ("h", "e") and replaces all occurrences of "he"
        in all words with a single merged token "he". This is the actual "merging" step.

        Example:
            Before: {"hello</w>": 5, "help</w>": 3}
            Pair: ("h", "e")
            After: {"he" + "llo</w>": 5, "he" + "lp</w>": 3}
            Result: {"hello</w>": 5, "help</w>": 3} -> {"hello</w>": 5, "help</w>": 3}
            Wait, that's the same... Actually, if we merge "he", we get:
            {"hello</w>": 5} -> {"he" + "llo</w>": 5} -> {"hello</w>": 5}
            But the internal representation changes: "h"+"e" becomes "he"

        Args:
            pair: Tuple of (token1, token2) to merge, e.g., ("h", "e")
            word_freqs: Current word frequencies dictionary

        Returns:
            Updated word frequencies dictionary with merged pairs
        """
        # Create a new dictionary to store updated words
        new_word_freqs = {}

        # Create the merged token by joining the pair
        # Example: ("h", "e") -> "he"
        merged_token = ''.join(pair)

        # For each word in the current vocabulary
        for word in word_freqs:
            # Replace all occurrences of the pair with the merged token
            # Example: word = "h" + "e" + "l" + "l" + "o</w>"
            #          pair = ("h", "e")
            #          After replace: "he" + "l" + "l" + "o</w>"
            # Note: replace() replaces ALL occurrences, not just the first one
            new_word = word.replace(''.join(pair), merged_token)

            # Keep the same frequency for the updated word
            new_word_freqs[new_word] = word_freqs[word]

        return new_word_freqs

    def train(self, texts: List[str], vocab_size: int):
        """
        Train the BPE tokenizer on a list of texts.

        The training process:
        1. Start with character-level vocabulary (each character is a token)
        2. Iteratively find the most frequent pair of consecutive tokens
        3. Merge that pair into a single token
        4. Repeat until we reach the desired vocabulary size

        Args:
            texts: List of training text strings
            vocab_size: Desired vocabulary size (including special tokens)
        """
        # Step 1: Pre-tokenize texts and count word frequencies
        # This gives us a dictionary: {"hello</w>": 5, "world</w>": 3, ...}
        word_freqs = self._get_word_freqs(texts)

        # Step 2: Initialize vocabulary with all unique characters
        # BPE starts at the character level - each character is initially a separate token
        vocab = set()
        for word in word_freqs.keys():
            for char in word:
                vocab.add(char)  # Add each unique character
        # Example: vocab = {"h", "e", "l", "o", "<", "/", "w", ">", ...}

        # Step 3: Convert to sorted list for consistent ordering
        vocab = sorted(list(vocab))

        # Step 4: Add special tokens (like <UNK> for unknown words)
        if self.unk_token not in vocab:
            vocab.insert(0, self.unk_token)

        # Step 5: Create vocabulary dictionaries
        # vocab: maps token -> index (e.g., {"h": 0, "e": 1, ...})
        # vocab_reverse: maps index -> token (e.g., {0: "h", 1: "e", ...})
        self.vocab = {token: idx for idx, token in enumerate(vocab)}
        self.vocab_reverse = {idx: token for token, idx in self.vocab.items()}

        # Step 6: Calculate how many merges we need to perform
        # If we want vocab_size=1000 and we have 100 characters, we need 900 merges
        num_merges = vocab_size - len(self.vocab)

        # Step 7: Perform BPE merges iteratively
        self.merges = []  # Store the order of merges (important for tokenization later)

        for i in range(num_merges):
            # Step 7a: Get statistics of all pairs of consecutive tokens
            # This tells us which pairs appear most frequently
            pairs = self._get_stats(word_freqs)

            # If no pairs left, we can't merge anymore (shouldn't happen normally)
            if not pairs:
                break

            # Step 7b: Find the most frequent pair
            # max() with key=pairs.get finds the pair with the highest frequency
            # Example: if pairs = {("h","e"): 1000, ("e","l"): 500}, best_pair = ("h","e")
            best_pair = max(pairs, key=pairs.get)

            # Step 7c: Merge this pair in all words
            # This updates word_freqs: "h"+"e" becomes "he" everywhere
            word_freqs = self._merge_vocab(best_pair, word_freqs)

            # Step 7d: Add the merged token to our vocabulary
            merged_token = ''.join(best_pair)  # ("h","e") -> "he"
            if merged_token not in self.vocab:
                # Assign it the next available index
                self.vocab[merged_token] = len(self.vocab)
                self.vocab_reverse[len(self.vocab) - 1] = merged_token

            # Step 7e: Record this merge (we need this order for tokenization)
            # The order matters! We apply merges in the same order during tokenization
            self.merges.append(best_pair)

    def _apply_bpe(self, word: str) -> List[str]:
        """
        Apply BPE merges to a single word.

        This function takes a word and applies all the learned BPE merges in order.
        It's like "replaying" the training process but for a single word.

        Example:
            Word: "hello</w>"
            Merges learned: [("h","e"), ("l","l"), ("he","l")]
            Step 1: ["h","e","l","l","o","<","/","w",">"]
            Step 2: Apply ("h","e"): ["he","l","l","o","<","/","w",">"]
            Step 3: Apply ("l","l"): ["he","ll","o","<","/","w",">"]
            Step 4: Apply ("he","l"): ["hel","l","o","<","/","w",">"]

        Args:
            word: Input word (should include </w> marker)

        Returns:
            List of BPE tokens
        """
        # Step 1: Ensure word has end-of-word marker
        word = word + self.end_of_word_token if not word.endswith(self.end_of_word_token) else word

        # Step 2: Start with individual characters
        # This is our initial state - each character is a separate token
        tokens = list(word)
        # Example: "hello</w>" -> ["h", "e", "l", "l", "o", "<", "/", "w", ">"]

        # Step 3: Apply all learned merges in the same order they were learned
        # This is crucial! The order matters because later merges depend on earlier ones
        for pair in self.merges:
            new_tokens = []  # Will store tokens after applying this merge
            i = 0  # Index to traverse current tokens

            # Go through tokens and look for the pair to merge
            while i < len(tokens):
                # Check if current token and next token form the pair we want to merge
                # Example: tokens = ["h", "e", "l"], pair = ("h", "e")
                #          At i=0: tokens[0]="h" == pair[0]="h" ✓
                #                  tokens[1]="e" == pair[1]="e" ✓
                #          So we merge them!
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    # Found the pair! Merge them into a single token
                    new_tokens.append(''.join(pair))
                    i += 2  # Skip both tokens since we merged them
                else:
                    # No merge here, keep the token as is
                    new_tokens.append(tokens[i])
                    i += 1  # Move to next token

            # Update tokens for next iteration
            tokens = new_tokens

        return tokens

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a text string using the trained BPE tokenizer.

        Args:
            text: Input text string

        Returns:
            List of tokenized subwords
        """
        # Pre-tokenize into words
        words = self._pre_tokenize(text)

        # Apply BPE to each word
        tokens = []
        for word in words:
            word_tokens = self._apply_bpe(word)
            tokens.extend(word_tokens)

        return tokens

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text string

        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        token_ids = []

        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # Handle unknown tokens
                token_ids.append(self.vocab.get(self.unk_token, 0))

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text string
        """
        tokens = [self.vocab_reverse.get(idx, self.unk_token) for idx in token_ids]

        # Remove end-of-word markers and join
        text = ''.join(tokens)
        text = text.replace(self.end_of_word_token, ' ')

        return text.strip()


if __name__ == "__main__":
    # Example usage
    print("BPE Tokenizer Implementation")
    print("=" * 50)

    # Sample training texts
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing is fascinating.",
        "Tokenization is an important preprocessing step."
    ]

    # Initialize tokenizer
    tokenizer = BPETokenizer()

    # Train with small vocabulary size for demonstration
    print("\nTraining tokenizer...")
    tokenizer.train(sample_texts, vocab_size=50)

    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print(f"Number of merges: {len(tokenizer.merges)}")

    # Test tokenization
    test_text = "The quick brown fox"
    tokens = tokenizer.tokenize(test_text)
    token_ids = tokenizer.encode(test_text)

    print(f"\nTest text: '{test_text}'")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")

    # Decode
    decoded = tokenizer.decode(token_ids)
    print(f"Decoded: '{decoded}'")
