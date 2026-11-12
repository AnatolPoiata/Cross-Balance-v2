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


def self_paced_chunk(hardness, chunk_size, alpha, rng, k_bins=5):
    """Hardness-harmonised majority sample (self-paced under-sampling, Liu
    et al., 2020). ``hardness`` is the ensemble-so-far probability of the
    minority class for every majority sample; samples are binned into
    ``k_bins`` equal-width hardness bins and each bin receives a share of
    the chunk proportional to ``1 / (mean_bin_hardness + alpha)``. Early in
    training (``alpha`` near 0) easy and hard regions are balanced; as
    ``alpha`` grows the share approaches uniform-per-bin, so the few hard
    (borderline / noisy) samples are never allowed to dominate a chunk —
    the failure mode of the multiplicative ``'boosting'`` update.
    """
    n = len(hardness)
    if hardness.max() == hardness.min():
        return rng.choice(n, chunk_size, replace=chunk_size > n)
    try:
        pops, edges = np.histogram(hardness, bins=k_bins)
    except ValueError:  # range too small for k finite-width bins: hardness is effectively constant
        return rng.choice(n, chunk_size, replace=chunk_size > n)
    bins, contrib = [], np.zeros(k_bins)
    for b in range(k_bins):
        upper = hardness <= edges[b + 1] if b == k_bins - 1 else hardness < edges[b + 1]
        idx = np.where((hardness >= edges[b]) & upper)[0]
        bins.append(idx)
        if len(idx):
            contrib[b] = hardness[idx].mean()
    with np.errstate(divide="ignore"):
        w = 1.0 / (contrib + alpha)
    w[(pops == 0) | ~np.isfinite(w)] = 0.0
    n_bins = np.floor(chunk_size * w / w.sum()).astype(int)
    order, i = np.argsort(-pops), 0
    while n_bins.sum() < chunk_size:
        if pops[order[i % k_bins]] > 0:
            n_bins[order[i % k_bins]] += 1
        i += 1
    parts = [rng.choice(bins[b], n_bins[b], replace=n_bins[b] > pops[b])
             for b in range(k_bins) if n_bins[b] > 0]
    return np.concatenate(parts)
