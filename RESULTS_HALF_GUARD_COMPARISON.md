# Half-Guard Selection Benchmark: Entropy vs Gain

## Setup

- Walk-forward folds: 2
- Total test matches: 3848
- Half-guards per fold (N_HALF): 4
- Leagues: E0, E1, E2, E3

## Per-Fold Results

| Fold | N | Mode | Acc_Top1 | Acc_Top2_HG | Combined | LogLoss | Brier |
|------|---|------|----------|-------------|----------|---------|-------|
| 1 | 1931 | gain | 0.4801 | 1.0000 | 0.4811 | 1.0417 | 0.6262 |
| 1 | 1931 | entropy | 0.4801 | 0.2500 | 0.4801 | 1.0417 | 0.6262 |
| 2 | 1917 | gain | 0.4684 | 1.0000 | 0.4695 | 1.0455 | 0.6298 |
| 2 | 1917 | entropy | 0.4684 | 0.5000 | 0.4690 | 1.0455 | 0.6298 |

## Aggregate (Mean across folds)

| Metric | Entropy | Gain | Delta (gain - entropy) | Better |
|--------|---------|------|------------------------|--------|
| Acc_Top1 | 0.4743 | 0.4743 | +0.0000 | tie |
| Acc_Top2_HG | 0.3750 | 1.0000 | +0.6250 | gain |
| Combined | 0.4745 | 0.4753 | +0.0008 | gain |
| LogLoss | 1.0436 | 1.0436 | +0.0000 | tie |
| Brier | 0.6280 | 0.6280 | +0.0000 | tie |

## Selection Divergence

| Fold | N_half | Selections that differ |
|------|--------|-----------------------|
| 1 | 4 | 8 |
| 2 | 4 | 8 |
| **Total** | 8 | 16 |

## Average Stats for Selected Half-Guards

| Fold | Mode | Avg Gain | Avg Top2 | Avg Entropy |
|------|------|----------|----------|-------------|
| 1 | gain | 0.3823 | 0.7651 | 0.9789 |
| 1 | entropy | 0.3773 | 0.7613 | 0.9805 |
| 2 | gain | 0.3743 | 0.7507 | 0.9848 |
| 2 | entropy | 0.3628 | 0.7338 | 0.9903 |

## Conclusion

**Gain-based selection is better** for half-guard accuracy (Acc_Top2_HG: 1.0000 vs 0.3750, delta +0.6250). The combined ticket hit rate difference is small (+0.0008) because N_HALF=4 is a small fraction of total matches, but the gain-based method picks half-guards that are much more likely to hit.

- Acc_Top2_HG delta (gain - entropy): +0.6250
- Combined hit rate delta (gain - entropy): +0.0008
- Selections that differed: 16 / 8
