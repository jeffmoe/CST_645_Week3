# CST_645_Week3
NLP assignment dealing with N-grams and tokenization in NLTK.

---
## Overview

This document provides a comprehensive guide to implementing N-grams using the Natural Language Toolkit (NLTK) in Python. It covers the theoretical foundations of N-grams, practical implementation steps, text preprocessing techniques, and visualization of results.

---

## Dataset

The sample dataset consists of three sentences:

sample = [
    "So do all who live to see such times.",  
    "But that is not for them to decide.",  
    "All we have to decide is what to do with the time that is given us."  
]

---

## Methodology

The implementation follows these key steps:

- Text Preprocessing: Function created for text preprocessing and tokenization in one
- Input Handling: Logic to handle strings or lists with error handling for improper inputs
- Text Normalization: Expands contractions, removes punctuation and special characters, and normalizes text by lowercasing
- Tokenization: Tokenizes input into sentences and words for comparison analysis
- N-gram Generation: Uses NLTK library for N-gram creation with different N values
- Visualization: Represents number of elements in each N-gram for word tokenization

## Library Installation and Setup

### First, install and import the required libraries:

```python
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import contractions
```

### Download required NLTK data
```python
nltk.download('popular')
```

### Preprocessing Function
The preprocessing function handles text cleaning and tokenization:
```python
def preprocess(text: str | list) -> list:
    '''
    Function to preprocess the text for N-gram creation.
    - outputs a list of tokens.
    '''
    try:
        # Check the input for dtype. Raise error if not string or list
        if isinstance(text, list):
            text = ' '.join(text)
        elif not isinstance(text, str):
            raise ValueError("Input must be a list or string")
        
        text = contractions.fix(text)  # Expand contractions first
        normalized_tokens_sentences = nltk.sent_tokenize(text)
        
        # Normalize sentences
        normalized_tokens_sentences = [' '.join(re.sub(r'[^a-zA-Z0-9]', '', token.lower()) for token in normalized_tokens_sentences.split())
                                       for normalized_tokens_sentences in normalized_tokens_sentences
                                       ]
        
        # Normalize words
        text = [re.sub(r'[^a-zA-Z0-9]', '', token.lower()) for token in text.split()]
        text = ' '.join(text)  # Join back into a string for tokenization
        normalized_tokens_words = nltk.word_tokenize(text)
        
        return normalized_tokens_words, normalized_tokens_sentences
    except Exception as e:
        print(f"Error in preprocessing: {e}")
```
