"""
Diagnostic script for optC embedding quality.

Investigates why optC silhouette scores (0.04-0.07) are much lower than optA (0.13)
despite optC being a sparsified version of optA (top-32 of 127 experts per layer).

Checks:
  1. optC sparsity (exactly 32 non-zeros per layer per doc)
  2. optC values match optA where non-zero
  3. Pairwise cosine similarity distributions for optA vs optC
  4. Embedding value statistics (mean, std, max, min for non-zero entries)
  5. PCA components needed for 95% variance for both optA and optC
  6. Effect of PCA centering on sparse optC embeddings

Usage:
    python diagnose_optc.py
"""

import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

DATA_DIR = "claude_outputs/analysis/router_clustering_pretraining"
NUM_LAYERS = 16
NUM_EXPERTS = 127
TOP_K_SPARSE = 32


def load_embeddings():
    """Load optA and optC embeddings."""
    optA_path = os.path.join(DATA_DIR, "embeddings_optA_avgprob.npy")
    optC_path = os.path.join(DATA_DIR, "embeddings_optC_top32sparse.npy")

    print(f"Loading optA from {optA_path}")
    optA = np.load(optA_path)
    print(f"  shape={optA.shape}, dtype={optA.dtype}")

    print(f"Loading optC from {optC_path}")
    optC = np.load(optC_path)
    print(f"  shape={optC.shape}, dtype={optC.dtype}")

    return optA, optC


def check_sparsity(optC, n_sample=500):
    """Verify optC has exactly TOP_K_SPARSE non-zeros per layer per doc."""
    print(f"\n{'='*70}")
    print("CHECK 1: optC sparsity (sample of {n_sample} docs)")
    print(f"{'='*70}")

    N = optC.shape[0]
    sample_idx = np.random.default_rng(42).choice(N, min(n_sample, N), replace=False)
    sample = optC[sample_idx]

    # Reshape to (n_sample, num_layers, num_experts)
    reshaped = sample.reshape(-1, NUM_LAYERS, NUM_EXPERTS)
    nnz_per_layer = (reshaped != 0).sum(axis=2)  # (n_sample, num_layers)

    # Check if exactly TOP_K_SPARSE per layer
    all_exact = (nnz_per_layer == TOP_K_SPARSE).all()
    print(f"  All layers have exactly {TOP_K_SPARSE} non-zeros: {all_exact}")

    if not all_exact:
        # Show distribution of non-zero counts
        unique, counts = np.unique(nnz_per_layer, return_counts=True)
        print("  Distribution of non-zero counts per layer:")
        for u, c in zip(unique, counts):
            print(f"    {u} non-zeros: {c} occurrences ({c / nnz_per_layer.size:.1%})")
        # Show which layers/docs are problematic
        bad_mask = nnz_per_layer != TOP_K_SPARSE
        bad_docs, bad_layers = np.where(bad_mask)
        print(f"  Number of (doc, layer) pairs with wrong count: {bad_mask.sum()}")
        if bad_mask.sum() > 0:
            for i in range(min(10, len(bad_docs))):
                d, layer_idx = bad_docs[i], bad_layers[i]
                print(
                    f"    doc={sample_idx[d]}, layer={layer_idx}, nnz={nnz_per_layer[d, layer_idx]}"
                )
    else:
        print(
            f"  Passed: all {len(sample_idx)} sampled docs x {NUM_LAYERS} layers have exactly {TOP_K_SPARSE} non-zeros"
        )

    # Also check overall sparsity
    total_elements = sample.size
    total_nonzero = np.count_nonzero(sample)
    density = total_nonzero / total_elements
    expected_density = TOP_K_SPARSE / NUM_EXPERTS
    print(f"  Overall density: {density:.4f} (expected: {expected_density:.4f})")


def check_value_match(optA, optC, n_sample=500):
    """Verify optC values match optA where optC is non-zero."""
    print(f"\n{'='*70}")
    print(f"CHECK 2: optC values match optA where non-zero (sample of {n_sample} docs)")
    print(f"{'='*70}")

    N = optA.shape[0]
    sample_idx = np.random.default_rng(42).choice(N, min(n_sample, N), replace=False)

    a_sample = optA[sample_idx].astype(np.float32)
    c_sample = optC[sample_idx].astype(np.float32)

    nonzero_mask = c_sample != 0
    if nonzero_mask.sum() == 0:
        print("  ERROR: optC has no non-zero values!")
        return

    # Compare values where optC is non-zero
    a_vals = a_sample[nonzero_mask]
    c_vals = c_sample[nonzero_mask]

    abs_diff = np.abs(a_vals - c_vals)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    exact_match = (abs_diff == 0).all()

    print(f"  Exact match (all diffs == 0): {exact_match}")
    print(f"  Max absolute difference: {max_diff:.8f}")
    print(f"  Mean absolute difference: {mean_diff:.8f}")
    print(f"  Number of non-zero entries compared: {nonzero_mask.sum()}")

    # Check if optC keeps the TOP values (not random values)
    # For a sample of docs, verify the kept values are indeed the top-32 per layer
    print(f"\n  Verifying optC keeps the TOP-{TOP_K_SPARSE} values per layer:")
    n_verify = min(100, len(sample_idx))
    mismatches = 0
    for i in range(n_verify):
        a_reshaped = a_sample[i].reshape(NUM_LAYERS, NUM_EXPERTS)
        c_reshaped = c_sample[i].reshape(NUM_LAYERS, NUM_EXPERTS)
        for layer in range(NUM_LAYERS):
            a_layer = a_reshaped[layer]
            c_layer = c_reshaped[layer]
            # Get top-32 indices from optA
            top32_from_a = set(np.argsort(a_layer)[-TOP_K_SPARSE:])
            # Get non-zero indices from optC
            nonzero_from_c = set(np.where(c_layer != 0)[0])
            if top32_from_a != nonzero_from_c:
                mismatches += 1
                if mismatches <= 5:
                    only_in_a = top32_from_a - nonzero_from_c
                    only_in_c = nonzero_from_c - top32_from_a
                    print(f"    MISMATCH doc={sample_idx[i]} layer={layer}:")
                    print(f"      In A's top-32 but zeroed in C: {only_in_a}")
                    print(f"      In C's non-zero but not in A's top-32: {only_in_c}")
                    # Check if this is a float16 tie-breaking issue
                    if only_in_a and only_in_c:
                        for idx_a in list(only_in_a)[:3]:
                            for idx_c in list(only_in_c)[:3]:
                                print(
                                    f"        A[{idx_a}]={a_layer[idx_a]:.6f} vs A[{idx_c}]={a_layer[idx_c]:.6f}"
                                )

    total_checks = n_verify * NUM_LAYERS
    print(f"  Mismatches: {mismatches}/{total_checks} layer checks ({mismatches/total_checks:.2%})")
    if mismatches > 0:
        print("  NOTE: Mismatches may be due to float16 tie-breaking in argsort.")
        print("  optC was computed on float16 values; this check uses float32 from optA.")


def compare_cosine_similarity(optA, optC, n_sample=1000):
    """Compare pairwise cosine similarity distributions for optA vs optC."""
    print(f"\n{'='*70}")
    print(f"CHECK 3: Pairwise cosine similarity distributions (sample of {n_sample} docs)")
    print(f"{'='*70}")

    N = optA.shape[0]
    sample_idx = np.random.default_rng(42).choice(N, min(n_sample, N), replace=False)

    a_sample = optA[sample_idx].astype(np.float32)
    c_sample = optC[sample_idx].astype(np.float32)

    # Compute pairwise cosine distances (1 - cosine_similarity)
    print("  Computing pairwise cosine distances for optA...")
    cos_dist_a = pairwise_distances(a_sample, metric="cosine")
    cos_sim_a = 1.0 - cos_dist_a

    print("  Computing pairwise cosine distances for optC...")
    cos_dist_c = pairwise_distances(c_sample, metric="cosine")
    cos_sim_c = 1.0 - cos_dist_c

    # Extract upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(len(sample_idx), k=1)
    sims_a = cos_sim_a[triu_idx]
    sims_c = cos_sim_c[triu_idx]

    print("\n  OptA cosine similarity distribution:")
    print(
        f"    mean={sims_a.mean():.4f}  std={sims_a.std():.4f}  "
        f"min={sims_a.min():.4f}  max={sims_a.max():.4f}"
    )
    print(
        f"    percentiles: 5%={np.percentile(sims_a, 5):.4f}  "
        f"25%={np.percentile(sims_a, 25):.4f}  50%={np.percentile(sims_a, 50):.4f}  "
        f"75%={np.percentile(sims_a, 75):.4f}  95%={np.percentile(sims_a, 95):.4f}"
    )

    print("\n  OptC cosine similarity distribution:")
    print(
        f"    mean={sims_c.mean():.4f}  std={sims_c.std():.4f}  "
        f"min={sims_c.min():.4f}  max={sims_c.max():.4f}"
    )
    print(
        f"    percentiles: 5%={np.percentile(sims_c, 5):.4f}  "
        f"25%={np.percentile(sims_c, 25):.4f}  50%={np.percentile(sims_c, 50):.4f}  "
        f"75%={np.percentile(sims_c, 75):.4f}  95%={np.percentile(sims_c, 95):.4f}"
    )

    # Key diagnostic: if optC similarities are much higher on average, it means
    # sparsification made embeddings more uniform (less discriminative)
    print("\n  DIAGNOSTIC:")
    print(f"    Mean cosine sim difference (optC - optA): {sims_c.mean() - sims_a.mean():.4f}")
    print(f"    Std cosine sim difference (optC - optA): {sims_c.std() - sims_a.std():.4f}")
    if sims_c.mean() > sims_a.mean() + 0.01:
        print("    WARNING: optC embeddings are MORE similar to each other than optA.")
        print("    This means sparsification reduced inter-document discrimination,")
        print("    which directly explains lower silhouette scores.")
    if sims_c.std() < sims_a.std() - 0.01:
        print("    WARNING: optC similarity spread is NARROWER than optA.")
        print("    Documents are harder to tell apart in optC space.")


def embedding_statistics(optA, optC):
    """Print statistics about embedding values."""
    print(f"\n{'='*70}")
    print("CHECK 4: Embedding value statistics")
    print(f"{'='*70}")

    a = optA.astype(np.float32)
    c = optC.astype(np.float32)

    # Overall stats
    print("\n  OptA (all values):")
    print(f"    mean={a.mean():.6f}  std={a.std():.6f}  min={a.min():.6f}  max={a.max():.6f}")

    print("\n  OptC (all values, including zeros):")
    print(f"    mean={c.mean():.6f}  std={c.std():.6f}  min={c.min():.6f}  max={c.max():.6f}")

    # Non-zero stats for optC
    c_nonzero = c[c != 0]
    print(
        f"\n  OptC (non-zero values only, {c_nonzero.size} of {c.size} = {c_nonzero.size/c.size:.2%}):"
    )
    print(
        f"    mean={c_nonzero.mean():.6f}  std={c_nonzero.std():.6f}  "
        f"min={c_nonzero.min():.6f}  max={c_nonzero.max():.6f}"
    )

    # Per-layer statistics
    print("\n  Per-layer sum statistics (should be ~1.0 for optA, less for optC):")
    a_layers = a.reshape(-1, NUM_LAYERS, NUM_EXPERTS).sum(axis=2)  # (N, 16)
    c_layers = c.reshape(-1, NUM_LAYERS, NUM_EXPERTS).sum(axis=2)  # (N, 16)
    print(f"    OptA per-layer sum: mean={a_layers.mean():.4f}  std={a_layers.std():.4f}")
    print(f"    OptC per-layer sum: mean={c_layers.mean():.4f}  std={c_layers.std():.4f}")

    # How much of the probability mass does top-32 capture?
    mass_ratio = c_layers / (a_layers + 1e-10)
    print(f"\n  Probability mass captured by top-{TOP_K_SPARSE} (optC_sum / optA_sum):")
    print(
        f"    mean={mass_ratio.mean():.4f}  std={mass_ratio.std():.4f}  "
        f"min={mass_ratio.min():.4f}  max={mass_ratio.max():.4f}"
    )

    # Variance of zero vs non-zero entries in optC
    print("\n  Variance decomposition for optC:")
    zero_fraction = (c == 0).mean()
    print(f"    Fraction of zeros: {zero_fraction:.4f}")
    print(f"    Variance of all values: {c.var():.8f}")
    print(f"    Variance of non-zero values only: {c_nonzero.var():.8f}")
    print(f"    NOTE: PCA centering will shift zeros to -mean ({-c.mean():.6f}),")
    print("    making them large negative values. This 'zero structure' dominates PCA variance.")


def pca_analysis(optA, optC):
    """Compare PCA components needed for 95% variance."""
    print(f"\n{'='*70}")
    print("CHECK 5: PCA components for 95% variance")
    print(f"{'='*70}")

    a = optA.astype(np.float32)
    c = optC.astype(np.float32)

    # Fit PCA on optA
    print(f"\n  Fitting PCA on optA ({a.shape})...")
    pca_a = PCA(n_components=min(a.shape[0], a.shape[1]), svd_solver="randomized", random_state=42)
    pca_a.fit(a)
    cumvar_a = np.cumsum(pca_a.explained_variance_ratio_)
    n_comp_a_95 = int(np.searchsorted(cumvar_a, 0.95)) + 1
    print(f"    Components for 95% variance: {n_comp_a_95}")
    print(f"    Variance at that point: {cumvar_a[n_comp_a_95-1]:.4f}")
    print(f"    Top-5 component variances: {pca_a.explained_variance_ratio_[:5]}")

    # Fit PCA on optC
    print(f"\n  Fitting PCA on optC ({c.shape})...")
    pca_c = PCA(n_components=min(c.shape[0], c.shape[1]), svd_solver="randomized", random_state=42)
    pca_c.fit(c)
    cumvar_c = np.cumsum(pca_c.explained_variance_ratio_)
    n_comp_c_95 = int(np.searchsorted(cumvar_c, 0.95)) + 1
    print(f"    Components for 95% variance: {n_comp_c_95}")
    print(f"    Variance at that point: {cumvar_c[n_comp_c_95-1]:.4f}")
    print(f"    Top-5 component variances: {pca_c.explained_variance_ratio_[:5]}")

    print("\n  DIAGNOSTIC:")
    print(f"    PCA components ratio (optC/optA): {n_comp_c_95/n_comp_a_95:.2f}")
    if n_comp_c_95 < n_comp_a_95:
        print("    optC needs FEWER PCA components. This suggests the zero structure")
        print("    is dominating — PCA captures 'which experts are active' (binary-like)")
        print("    rather than the fine-grained probability differences between them.")
    if pca_c.explained_variance_ratio_[0] > 0.3:
        print(
            f"    WARNING: First PCA component of optC explains {pca_c.explained_variance_ratio_[0]:.1%}"
        )
        print("    of variance. This likely captures the shared sparsity pattern (the zeros),")
        print("    not meaningful routing differences.")

    # Compare what PCA centering does
    print("\n  Effect of PCA centering (mean subtraction):")
    a_mean = a.mean(axis=0)
    c_mean = c.mean(axis=0)
    print(
        f"    OptA feature means: range [{a_mean.min():.6f}, {a_mean.max():.6f}], "
        f"std={a_mean.std():.6f}"
    )
    print(
        f"    OptC feature means: range [{c_mean.min():.6f}, {c_mean.max():.6f}], "
        f"std={c_mean.std():.6f}"
    )

    # Show the fraction of features that are always zero in optC
    always_zero = (c == 0).all(axis=0).sum()
    sometimes_zero = ((c == 0).any(axis=0) & ~(c == 0).all(axis=0)).sum()
    never_zero = (~(c == 0).any(axis=0)).sum()
    print("\n    OptC feature analysis across all docs:")
    print(f"      Always zero: {always_zero}/{c.shape[1]} features")
    print(f"      Sometimes zero: {sometimes_zero}/{c.shape[1]} features")
    print(f"      Never zero: {never_zero}/{c.shape[1]} features")


def post_transform_comparison(optA, optC):
    """Apply the same pca_l2 transform to both and compare."""
    print(f"\n{'='*70}")
    print("CHECK 6: Post-transform (pca_l2) comparison")
    print(f"{'='*70}")

    a = optA.astype(np.float32)
    c = optC.astype(np.float32)

    # Apply PCA (95% var) + L2 norm to both
    print("\n  Transforming optA with pca_l2...")
    pca_a = PCA(n_components=min(a.shape[0], a.shape[1]), svd_solver="randomized", random_state=42)
    pca_a.fit(a)
    cumvar_a = np.cumsum(pca_a.explained_variance_ratio_)
    n_a = int(np.searchsorted(cumvar_a, 0.95)) + 1
    pca_a_final = PCA(n_components=n_a, random_state=42)
    a_reduced = pca_a_final.fit_transform(a)
    a_normed = normalize(a_reduced, norm="l2")

    print("  Transforming optC with pca_l2...")
    pca_c = PCA(n_components=min(c.shape[0], c.shape[1]), svd_solver="randomized", random_state=42)
    pca_c.fit(c)
    cumvar_c = np.cumsum(pca_c.explained_variance_ratio_)
    n_c = int(np.searchsorted(cumvar_c, 0.95)) + 1
    pca_c_final = PCA(n_components=n_c, random_state=42)
    c_reduced = pca_c_final.fit_transform(c)
    c_normed = normalize(c_reduced, norm="l2")

    print(f"  OptA: {a.shape} -> PCA({n_a}) -> L2 norm -> {a_normed.shape}")
    print(f"  OptC: {c.shape} -> PCA({n_c}) -> L2 norm -> {c_normed.shape}")

    # Compare pairwise distance distributions in the transformed space
    n_sample = 1000
    N = min(optA.shape[0], optC.shape[0])
    idx = np.random.default_rng(42).choice(N, min(n_sample, N), replace=False)

    print(f"\n  Pairwise Euclidean distances in transformed space (sample={min(n_sample, N)}):")
    dist_a = pairwise_distances(a_normed[idx], metric="euclidean")
    dist_c = pairwise_distances(c_normed[idx], metric="euclidean")
    triu = np.triu_indices(len(idx), k=1)

    da = dist_a[triu]
    dc = dist_c[triu]
    print(
        f"    OptA: mean={da.mean():.4f}  std={da.std():.4f}  range=[{da.min():.4f}, {da.max():.4f}]"
    )
    print(
        f"    OptC: mean={dc.mean():.4f}  std={dc.std():.4f}  range=[{dc.min():.4f}, {dc.max():.4f}]"
    )

    # Coefficient of variation (std/mean) — higher = more separable
    cv_a = da.std() / da.mean()
    cv_c = dc.std() / dc.mean()
    print("\n    Distance coefficient of variation (std/mean):")
    print(f"      OptA: {cv_a:.4f}")
    print(f"      OptC: {cv_c:.4f}")
    if cv_c < cv_a:
        print("    WARNING: OptC has lower distance CV, meaning distances are more")
        print("    uniform (less spread). This directly causes lower silhouette scores")
        print("    because K-means can't distinguish clusters as well.")

    # Alternative: try l2-only (no PCA) on optC
    print("\n  Alternative: L2-only (no PCA) on optC:")
    c_l2only = normalize(c, norm="l2")
    dist_c_l2 = pairwise_distances(c_l2only[idx], metric="euclidean")
    dc_l2 = dist_c_l2[triu]
    cv_c_l2 = dc_l2.std() / dc_l2.mean()
    print(f"    OptC (L2 only): mean={dc_l2.mean():.4f}  std={dc_l2.std():.4f}  CV={cv_c_l2:.4f}")

    # Alternative: try standardize_pca_l2 on optC
    print("\n  Alternative: standardize + PCA + L2 on optC:")
    from sklearn.preprocessing import StandardScaler

    c_scaled = StandardScaler().fit_transform(c)
    pca_cs = PCA(
        n_components=min(c_scaled.shape[0], c_scaled.shape[1]),
        svd_solver="randomized",
        random_state=42,
    )
    pca_cs.fit(c_scaled)
    cumvar_cs = np.cumsum(pca_cs.explained_variance_ratio_)
    n_cs = int(np.searchsorted(cumvar_cs, 0.95)) + 1
    pca_cs_final = PCA(n_components=n_cs, random_state=42)
    c_std_reduced = pca_cs_final.fit_transform(c_scaled)
    c_std_normed = normalize(c_std_reduced, norm="l2")
    dist_c_std = pairwise_distances(c_std_normed[idx], metric="euclidean")
    dc_std = dist_c_std[triu]
    cv_c_std = dc_std.std() / dc_std.mean()
    print(
        f"    OptC (std+PCA+L2, {n_cs} comps): mean={dc_std.mean():.4f}  std={dc_std.std():.4f}  CV={cv_c_std:.4f}"
    )


def main():
    print("OptC Diagnostic Script")
    print(f"Data directory: {DATA_DIR}")
    print(f"Expected shape: (N, {NUM_LAYERS * NUM_EXPERTS})")
    print()

    optA, optC = load_embeddings()

    assert optA.shape == optC.shape, f"Shape mismatch: optA={optA.shape} vs optC={optC.shape}"
    assert (
        optA.shape[1] == NUM_LAYERS * NUM_EXPERTS
    ), f"Unexpected dim: {optA.shape[1]} != {NUM_LAYERS * NUM_EXPERTS}"

    check_sparsity(optC, n_sample=500)
    check_value_match(optA, optC, n_sample=500)
    compare_cosine_similarity(optA, optC, n_sample=1000)
    embedding_statistics(optA, optC)
    pca_analysis(optA, optC)
    post_transform_comparison(optA, optC)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(
        """
Likely root causes for low optC silhouette scores:

1. PCA CENTERING ON SPARSE DATA: PCA subtracts the mean from each feature.
   For optC, ~75% of entries are zero. After centering, these zeros become
   -mean (a substantial negative value), and the non-zero entries shift too.
   The first PCA components likely capture the zero/non-zero structure rather
   than the fine-grained probability differences that distinguish domains.

2. REDUCED DISCRIMINATION: By zeroing 95 of 127 experts per layer, optC
   throws away the subtle differences in the tail of the softmax distribution.
   Two documents from different domains might have identical top-32 experts
   but differ in their lower-ranked experts. OptA captures this; optC does not.

3. FLOAT16 PRECISION: The argsort for top-32 selection operates on float16
   values. Many experts have very similar softmax probabilities (all near
   1/127 ~ 0.0079). Float16 has ~3 decimal digits of precision, so ties
   in the 32nd-33rd expert are likely, making the sparsity boundary noisy.

Potential fixes:
  - Use 'l2' transform (no PCA) for optC, since PCA centering destroys sparsity
  - Use 'standardize_pca_l2' which z-scores first (may handle zeros better)
  - Compute optC on float32 values before casting to float16
  - Try a higher TOP_K_SPARSE (e.g., 48 or 64) to retain more signal
  - Consider TF-IDF-like reweighting: upweight rare experts, downweight common ones
"""
    )


if __name__ == "__main__":
    main()
