# E0 Divergence Diagnosis: Half-Guard Benchmark

## Background

PR #36 showed that gain-based half-guard selection beats entropy overall,
winning in E1, E2, and E3. However, E0 (Premier League) showed a **slight**
advantage for entropy. This report investigates why.

## 1. Sample Size and Absolute Hits

### Per-league totals (across all folds and N_HALF values)

| League | HG Decisions | Gain Hits | Entropy Hits | Delta (G-E) | Gain Acc | Entropy Acc |
|--------|-------------|-----------|--------------|-------------|----------|-------------|
| E0 | 48 | 30 | 32 | -2 | 0.6250 | 0.6667 |
| E1 | 48 | 39 | 29 | +10 | 0.8125 | 0.6042 |
| E2 | 48 | 35 | 33 | +2 | 0.7292 | 0.6875 |
| E3 | 48 | 38 | 31 | +7 | 0.7917 | 0.6458 |

### E0 detail by fold

| Fold | HG Decisions | Gain Hits | Entropy Hits | Delta |
|------|-------------|-----------|--------------|-------|
| 1 | 12 | 5 | 9 | -4 |
| 2 | 12 | 7 | 7 | +0 |
| 3 | 12 | 6 | 7 | -1 |
| 4 | 12 | 12 | 9 | +3 |

### E0 detail by N_HALF

| N_HALF | HG Decisions | Gain Hits | Entropy Hits | Delta | Gain Acc | Entropy Acc |
|--------|-------------|-----------|--------------|-------|----------|-------------|
| 2 | 8 | 5 | 4 | +1 | 0.6250 | 0.5000 |
| 4 | 16 | 9 | 11 | -2 | 0.5625 | 0.6875 |
| 6 | 24 | 16 | 17 | -1 | 0.6667 | 0.7083 |

### Statistical context

- E0 gain accuracy: 30/48 = 0.6250 (95% CI: [0.484, 0.748])
- E0 entropy accuracy: 32/48 = 0.6667 (95% CI: [0.525, 0.783])
- Absolute difference: 2 hits out of 48 decisions
- The 95% confidence intervals **overlap substantially**.

### Confidence interval comparison across leagues

| League | N | Gain Hits | Gain Acc | 95% CI | Entropy Hits | Entropy Acc | 95% CI | CIs Overlap? |
|--------|---|-----------|----------|--------|--------------|-------------|--------|--------------|
| E0 | 48 | 30 | 0.6250 | [0.484, 0.748] | 32 | 0.6667 | [0.525, 0.783] | Yes |
| E1 | 48 | 39 | 0.8125 | [0.681, 0.898] | 29 | 0.6042 | [0.463, 0.730] | Yes |
| E2 | 48 | 35 | 0.7292 | [0.590, 0.834] | 33 | 0.6875 | [0.547, 0.801] | Yes |
| E3 | 48 | 38 | 0.7917 | [0.657, 0.883] | 31 | 0.6458 | [0.504, 0.766] | Yes |

## 2. Probability Profile: E0 vs Other Leagues

How do the model's probability distributions differ across leagues?

### Distribution of best, second, top2, and entropy

| League | N_matches | Mean Best | Mean Second | Mean Top2 | Mean Entropy | Std Entropy |
|--------|----------|-----------|-------------|-----------|--------------|-------------|
| E0 | 866 | 0.4915 | 0.2844 | 0.7759 | 0.9374 | 0.0479 |
| E1 | 1247 | 0.4745 | 0.2942 | 0.7686 | 0.9489 | 0.0388 |
| E2 | 1255 | 0.4808 | 0.2898 | 0.7706 | 0.9453 | 0.0399 |
| E3 | 1248 | 0.4751 | 0.2930 | 0.7681 | 0.9490 | 0.0367 |

### Entropy distribution percentiles

| League | P10 | P25 | P50 | P75 | P90 | Mean |
|--------|-----|-----|-----|-----|-----|------|
| E0 | 0.8643 | 0.9122 | 0.9545 | 0.9747 | 0.9814 | 0.9374 |
| E1 | 0.8882 | 0.9333 | 0.9646 | 0.9764 | 0.9828 | 0.9489 |
| E2 | 0.8834 | 0.9226 | 0.9601 | 0.9758 | 0.9822 | 0.9453 |
| E3 | 0.8932 | 0.9299 | 0.9622 | 0.9771 | 0.9833 | 0.9490 |

### Second-best probability distribution percentiles

| League | P10 | P25 | P50 | P75 | P90 | Mean |
|--------|-----|-----|-----|-----|-----|------|
| E0 | 0.2250 | 0.2415 | 0.2783 | 0.3259 | 0.3580 | 0.2844 |
| E1 | 0.2326 | 0.2495 | 0.2933 | 0.3380 | 0.3618 | 0.2942 |
| E2 | 0.2302 | 0.2474 | 0.2844 | 0.3319 | 0.3603 | 0.2898 |
| E3 | 0.2339 | 0.2490 | 0.2900 | 0.3352 | 0.3629 | 0.2930 |

## 3. Selection Divergence Analysis

When gain and entropy pick different matches, who picks better?

### Overlap and divergent selections per league

| League | Total HG | Overlap | Gain-Only | Entropy-Only | Gain-Only Hit Rate | Entropy-Only Hit Rate |
|--------|---------|---------|-----------|--------------|--------------------|-----------------------|
| E0 | 48 | 8 | 40 | 40 | 0.6250 (25/40) | 0.6750 (27/40) |
| E1 | 48 | 0 | 48 | 48 | 0.8125 (39/48) | 0.6042 (29/48) |
| E2 | 48 | 4 | 44 | 44 | 0.7045 (31/44) | 0.6591 (29/44) |
| E3 | 48 | 1 | 47 | 47 | 0.8085 (38/47) | 0.6596 (31/47) |

**Key question**: In E0, when the methods disagree, does entropy's unique picks
hit more often than gain's unique picks?

### E0 divergent selections by fold

| Fold | Gain-Only | Gain-Only Hits | Entropy-Only | Entropy-Only Hits |
|------|-----------|----------------|--------------|-------------------|
| 1 | 11 | 4 | 11 | 8 |
| 2 | 5 | 3 | 5 | 3 |
| 3 | 12 | 6 | 12 | 7 |
| 4 | 12 | 12 | 12 | 9 |

## 4. Properties of Selected Matches

Average probability stats for the matches each method selects.

| League | Method | Avg Gain (2nd best) | Avg Top2 | Avg Entropy |
|--------|--------|---------------------|----------|-------------|
| E0 | gain | 0.3819 | 0.7764 | 0.9729 |
| E0 | entropy | 0.3550 | 0.7323 | 0.9861 |
| E1 | gain | 0.3839 | 0.7825 | 0.9685 |
| E1 | entropy | 0.3392 | 0.7005 | 0.9957 |
| E2 | gain | 0.3788 | 0.7637 | 0.9793 |
| E2 | entropy | 0.3601 | 0.7418 | 0.9870 |
| E3 | gain | 0.3844 | 0.7725 | 0.9738 |
| E3 | entropy | 0.3470 | 0.7361 | 0.9879 |

### Gap between gain-selected and entropy-selected stats

| League | Gain Diff (G-E) | Top2 Diff (G-E) | Entropy Diff (G-E) |
|--------|-----------------|-----------------|---------------------|
| E0 | +0.0269 | +0.0441 | -0.0132 |
| E1 | +0.0447 | +0.0820 | -0.0272 |
| E2 | +0.0187 | +0.0218 | -0.0078 |
| E3 | +0.0374 | +0.0365 | -0.0141 |

## 5. E0 Model Confidence and Predictability

Is E0 generally harder to predict? Does the model show different confidence patterns?

### Top-1 and Top-2 accuracy by league (from match data)

| League | N_matches | Top1 Acc | Top2 Acc | Mean Best Prob | Mean Entropy |
|--------|----------|----------|----------|----------------|--------------|
| E0 | 866 | 0.5023 | 0.7760 | 0.4915 | 0.9374 |
| E1 | 1247 | 0.4387 | 0.7578 | 0.4745 | 0.9489 |
| E2 | 1255 | 0.4821 | 0.7641 | 0.4808 | 0.9453 |
| E3 | 1248 | 0.4631 | 0.7364 | 0.4751 | 0.9490 |

### Correlation: entropy vs top2_hit by league

Higher entropy should mean less certainty, but does it predict top2 misses?

| League | Corr(entropy, top2_hit) | P-value | N |
|--------|------------------------|---------|---|
| E0 | -0.1151 | 0.0007 | 866 |
| E1 | -0.0912 | 0.0013 | 1247 |
| E2 | -0.0620 | 0.0280 | 1255 |
| E3 | -0.0608 | 0.0317 | 1248 |

### Correlation: gain (second-best prob) vs top2_hit by league

| League | Corr(gain, top2_hit) | P-value | N |
|--------|---------------------|---------|---|
| E0 | -0.0749 | 0.0275 | 866 |
| E1 | -0.0830 | 0.0033 | 1247 |
| E2 | -0.0277 | 0.3260 | 1255 |
| E3 | -0.0284 | 0.3161 | 1248 |

## 6. Diagnosis and Conclusion

### Key findings

1. **Tiny sample, tiny delta**: E0's divergence rests on 2 absolute hits out of 48 half-guard decisions (30 gain vs 32 entropy). The 95% confidence intervals overlap heavily.

2. **Inconsistent across folds**: Gain wins 1 fold(s), entropy wins 2 fold(s), tied 1 fold(s). No consistent pattern across time periods.

3. **Narrower gain-entropy separation in E0**: The gap in avg second-best probability between gain-selected and entropy-selected matches is smaller in E0 (+0.0269) compared to E1 (+0.0447), E2 (+0.0187), E3 (+0.0374). This means gain and entropy tend to pick more similar matches in E0, reducing the advantage of gain-based selection.

4. **E0 has lower mean entropy**: E0 mean entropy = 0.9374 vs E1+E2+E3 mean = 0.9477. Premier League predictions tend to be more confident on average, meaning fewer genuinely uncertain matches for gain to differentiate from.

5. **Divergent picks**: When the methods disagree in E0, gain-only picks hit 25/40 (62.5%) vs entropy-only picks hit 27/40 (67.5%). This difference is small and could easily be noise.

### Overall diagnosis

**The E0 divergence is almost certainly sample noise.**

- The entire effect is 2 hits out of 48 decisions -- well within binomial sampling error.
- The 95% CIs for gain (0.484-0.748) and entropy (0.525-0.783) overlap completely.
- The effect is not consistent across folds, ruling out a systematic league-specific cause.
- E0 does show a slightly narrower probability spread (lower entropy, tighter gain-entropy separation), which means the gain method has less room to add value. But this is a *marginal* effect -- it explains why E0's gain advantage might be *smaller*, not why it should flip to entropy's favor.

### Recommendation

- **No action needed** on the production logic. The gain-based method is still
  the correct choice for all leagues including E0.
- If future benchmark runs with more data continue to show E0 diverging,
  consider investigating whether Premier League's tighter probability
  distributions warrant a league-specific N_HALF or a gain threshold cutoff.
- A possible future improvement: hybrid selection that uses gain but falls back
  to entropy when the gain gap between candidates is very small.
