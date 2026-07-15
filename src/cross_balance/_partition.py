"""Strategies for slicing the majority class into balanced chunks."""

import numpy as np


def bagging_chunks(n_majority, chunk_size, n_chunks, random_state=None):
    """Sequential, wrap-around chunks that together cover the majority class
    roughly evenly (each sample is reused only when ``chunk_size * n_chunks``
    exceeds ``n_majority``).
    """
    rng = np.random.RandomState(random_state)
    order = rng.permutation(n_majority)
    chunks = []
    for i in range(n_chunks):
        start = (i * chunk_size) % n_majority
        positions = (np.arange(start, start + chunk_size)) % n_majority
        chunks.append(order[positions])
    return chunks


def weighted_chunks(n_majority, chunk_size, n_chunks, weights, random_state=None):
    """Weighted-resampling chunks: each chunk is drawn with replacement using
    per-sample ``weights`` (higher weight -> more likely to be included).
    Used by the cost-sensitive ``'boosting'`` ensemble type, where weights are
    updated between chunks to focus later members on majority samples the
    ensemble-so-far struggles with.
    """
    rng = np.random.RandomState(random_state)
    chunks = []
    for _ in range(n_chunks):
        p = weights / weights.sum()
        idx = rng.choice(n_majority, size=chunk_size, replace=True, p=p)
        chunks.append(idx)
    return chunks
