import logging
from collections import Counter
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

def encode_text(text: str) -> List[int]:
    """
    Encode text to UTF-8 bytes.
    
    Args:
        text: Input text string
        
    Returns:
        List of integers representing UTF-8 bytes
    """
    byte_tokens = list(text.encode('utf-8'))
    logger.debug(f"Encoded text length: {len(text)} -> {len(byte_tokens)} bytes")
    return byte_tokens

def get_most_common_pair(encoded_bytes: List[int]) -> Optional[Tuple[int, int]]:
    """
    Find the most frequently occurring pair of adjacent bytes.
    
    Args:
        encoded_bytes: List of byte values
        
    Returns:
        Most common pair of bytes, or None if no pairs exist
    """
    if len(encoded_bytes) < 2:
        return None
        
    pair_stats = Counter()
    for a, b in zip(encoded_bytes, encoded_bytes[1:]):
        pair_stats[(a, b)] += 1
    
    if not pair_stats:
        return None
        
    most_common = pair_stats.most_common(1)[0]
    logger.debug(f"Most common pair: {most_common}")
    return most_common[0]

def merge_pair(encoded_bytes: List[int], pair: Tuple[int, int], new_token: int) -> List[int]:
    """
    Merge all occurrences of a byte pair into a new token.
    
    Args:
        encoded_bytes: List of byte values
        pair: Tuple of two bytes to merge
        new_token: New token value to replace the pair
        
    Returns:
        New list with merged tokens
    """
    n = 0
    result = []
    while n < len(encoded_bytes):
        if n < len(encoded_bytes) - 1 and (encoded_bytes[n], encoded_bytes[n + 1]) == pair:
            result.append(new_token)
            n += 2
        else:
            result.append(encoded_bytes[n])
            n += 1
    
    logger.debug(f"Merged {pair} -> {new_token}, new length: {len(result)}")
    return result

def learn_bpe(text: str, num_merges: int, start_token: int = 256) -> Tuple[List[int], List[Tuple[Tuple[int, int], int]]]:
    """
    Learn BPE merges from input text.
    
    Args:
        text: Input text to learn from
        num_merges: Maximum number of merges to perform
        start_token: Starting value for new tokens (default: 256)
        
    Returns:
        Tuple of (final byte sequence, list of merges performed)
    """
    logger.info(f"Starting BPE learning on text of length {len(text)}")
    
    # Initial encoding
    byte_tokens = encode_text(text)
    merges = []
    new_token = start_token
    
    # Perform merges
    for i in range(num_merges):
        pair = get_most_common_pair(byte_tokens)
        if not pair:
            logger.info(f"No more pairs to merge after {i} merges")
            break
            
        byte_tokens = merge_pair(byte_tokens, pair, new_token)
        merges.append((pair, new_token))
        
        logger.info(f"Merge {i+1}: {pair} -> {new_token}, sequence length: {len(byte_tokens)}")
        new_token += 1
        
        if len(byte_tokens) == 1:
            logger.info("Reached single token, stopping merges")
            break
    
    return byte_tokens, merges

def tokenize(text: str, merges: List[Tuple[Tuple[int, int], int]]) -> List[int]:
    """
    Apply learned BPE merges to new text.
    
    Args:
        text: Input text to encode
        merges: List of merge operations as (pair, new_token) tuples
        
    Returns:
        List of tokens after applying merges
    """
    logger.debug(f"Applying {len(merges)} merges to text of length {len(text)}")
    tokens = encode_text(text)
    
    for pair, new_token in merges:
        tokens = merge_pair(tokens, pair, new_token)
        
    logger.debug(f"Final token sequence length: {len(tokens)}")
    return tokens

def decode_tokens(tokens: List[int], merges: List[Tuple[Tuple[int, int], int]]) -> str:
    """
    Decode a sequence of tokens back to text using the merge operations in reverse.
    
    Args:
        tokens: List of tokens to decode
        merges: List of merge operations as (pair, new_token) tuples
        
    Returns:
        Decoded text string
    """
    logger.debug(f"Decoding token sequence of length {len(tokens)}")
    
    # Create reverse mapping from new_token to original pair
    token_to_pair = {new_token: pair for pair, new_token in merges}
    
    def expand_token(token: int) -> List[int]:
        """Recursively expand a token to its original bytes."""
        if token in token_to_pair:
            a, b = token_to_pair[token]
            return expand_token(a) + expand_token(b)
        return [token]
    
    # Expand all tokens back to original bytes
    bytes_list = []
    for token in tokens:
        bytes_list.extend(expand_token(token))
    
    # Convert bytes back to text
    try:
        text = bytes(bytes_list).decode('utf-8')
        logger.debug(f"Successfully decoded to text of length {len(text)}")
        return text
    except UnicodeDecodeError as e:
        logger.error(f"Failed to decode bytes: {e}")
        raise

def calculate_compression_ratio(raw_tokens: List[int], compressed_tokens: List[int]) -> float:
    """
    Calculate the compression ratio achieved by BPE tokenization.
    
    Args:
        raw_tokens: List of original byte tokens before compression
        compressed_tokens: List of tokens after applying BPE merges
        
    Returns:
        Compression ratio as a float (original_size/compressed_size)
        A ratio > 1 indicates compression, < 1 indicates expansion
    """
    original_size = len(raw_tokens)
    compressed_size = len(compressed_tokens)
    
    ratio = original_size / compressed_size if compressed_size > 0 else 0
    
    logger.info("Compression Analysis:")
    logger.info(f"Original size: {original_size} tokens")
    logger.info(f"Compressed size: {compressed_size} tokens")
    logger.info(f"Compression ratio: {ratio:.2f}x")
    logger.info(f"Space saving: {((1 - 1/ratio) * 100):.1f}%" if ratio > 0 else "N/A")
    
    return ratio
