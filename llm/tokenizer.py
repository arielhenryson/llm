import tiktoken
from abc import ABC, abstractmethod
from typing import List

class Tokenizer(ABC):
    """
    Abstract base class for tokenizers.
    
    Think of this as a "Contract" or "Blueprint". It doesn't do the work itself,
    but it forces any class that inherits from it to follow specific rules.
    This ensures that whether you use a simple Character tokenizer or a 
    complex GPT-4 tokenizer, your main code can treat them exactly the same.
    """

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Converts a string of text into a list of numbers (tokens).
        
        Args:
            text (str): The input text to process.
            
        Returns:
            List[int]: A list of integers representing the text.
        """
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """
        Converts a list of numbers (tokens) back into a string of text.
        
        Args:
            tokens (List[int]): The list of integers to convert back.
            
        Returns:
            str: The reconstructed text.
        """
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """
        Returns the total number of unique tokens this tokenizer knows.
        
        This is crucial for defining the size of the Neural Network's 
        input and output layers.
        """
        pass

class CharTokenizer(Tokenizer):
    """
    Simple character-level tokenizer.
    
    HOW IT WORKS:
    1. It looks at every unique character in your text (a, b, c, 1, 2, space, etc.).
    2. It assigns a unique number to each character.
    
    PROS: 
    - Very simple to understand and implement.
    - Great for learning how Transformers work from scratch.
    
    CONS: 
    - Produces very long sequences of tokens (e.g., "Hello" is 5 tokens).
    - The model has to work harder to learn relationships between distant parts of words.
    """

    def __init__(self, text: str = None, chars: List[str] = None):
        """
        Initialize the tokenizer.
        
        You must provide EITHER:
        1. 'text': A large string (like a book). The tokenizer will find all unique 
           characters in this string to create its vocabulary.
        OR
        2. 'chars': A list of characters defining the vocabulary directly.
        """
        if chars is None and text is None:
            raise ValueError("Either 'text' or 'chars' must be provided to build vocabulary.")
        
        # If text is provided, find all unique characters and sort them
        if chars is None:
            chars = sorted(list(set(text)))
        
        self.chars = chars
        self.vocab_size = len(self.chars) # Total unique characters
        
        # Create mappings:
        # stoi (String TO Integer): maps 'a' -> 1
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        # itos (Integer TO String): maps 1 -> 'a'
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> List[int]:
        """
        Translates text to integers using the 'stoi' dictionary.
        Example: "Hi" -> [33, 45]
        """
        return [self.stoi[c] for c in text]

    def decode(self, tokens: List[int]) -> str:
        """
        Translates integers back to text using the 'itos' dictionary.
        Example: [33, 45] -> "Hi"
        """
        return ''.join([self.itos[i] for i in tokens])
    
    # The @property decorator allows us to access this like a variable: tokenizer.vocab_size
    # instead of a function: tokenizer.vocab_size()
    @property 
    def vocab_size(self) -> int:
        return len(self.chars)

    def save(self, path: str):
        """
        Placeholder for saving the vocabulary to disk.
        In a real project, you would save the 'chars' list to a JSON or Pickle file
        so you can load the same tokenizer later without re-reading the text data.
        """
        pass

class TiktokenTokenizer(Tokenizer):
    """
    Wrapper around OpenAI's 'tiktoken' library.
    
    HOW IT WORKS:
    This uses Byte Pair Encoding (BPE). Instead of looking at single characters,
    it looks for common chunks of text (sub-words).
    - Common words like " the " might be 1 token.
    - Rare words might be split into 2-3 tokens.
    
    PROS:
    - Much more efficient. "Hello world" is 2 tokens here, but 11 tokens in CharTokenizer.
    - This is what real production models (GPT-3, GPT-4) use.
    """

    def __init__(self, encoding_name: str = "gpt2"):
        """
        Args:
            encoding_name: The name of the OpenAI model encoding to use.
                           "gpt2" is standard for older models.
                           "cl100k_base" is used for GPT-4.
        """
        # Load the pre-trained tokenizer from OpenAI
        self.enc = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> List[int]:
        """Uses the optimized C code inside tiktoken to encode text."""
        return self.enc.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """Uses the optimized C code inside tiktoken to decode text."""
        return self.enc.decode(tokens)

    @property
    def vocab_size(self) -> int:
        """
        Returns the vocab size defined by OpenAI for this model.
        For GPT-2, this is roughly 50,257 tokens.
        """
        return self.enc.n_vocab