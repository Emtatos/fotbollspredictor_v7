# Half-Guard Benchmark: Expanded Entropy vs Gain Comparison

## Setup

- Walk-forward folds: 5 (using 4 test folds)
- Leagues: E0, E1, E2, E3
- N_HALF values tested: 2, 4, 6
- League subsets: all (combined), E0, E1, E2, E3 (individual)
- Total benchmark cells: 60
- Total test matches (all-league cells): 13848
- Total half-guard decisions (all-league cells): 48

## Overall Summary (all leagues combined)

| Metric | Entropy | Gain | Delta | Better |
|--------|---------|------|-------|--------|
| Acc_Top2_HG | 0.5417 | 0.6667 | +0.1250 | gain |
| Combined | 0.4701 | 0.4700 | -0.0001 | tie |
| Acc_Top1 | 0.4690 | 0.4690 | +0.0000 | tie |

- Divergent selections: 96 / 48
- Mean gain (gain-selected): 0.3806
- Mean gain (entropy-selected): 0.3521
- Mean top2 (gain-selected): 0.7643
- Mean top2 (entropy-selected): 0.7334

## Results by N_HALF

| N_HALF | HG Decisions | Entropy Acc_HG | Gain Acc_HG | Delta | Better |
|--------|-------------|----------------|-------------|-------|--------|
| 2 | 8 | 0.2500 | 0.6250 | +0.3750 | gain |
| 4 | 16 | 0.5625 | 0.6875 | +0.1250 | gain |
| 6 | 24 | 0.6250 | 0.6667 | +0.0417 | gain |

## Results by League

| League | HG Decisions | Entropy Acc_HG | Gain Acc_HG | Delta | Combined Delta | Better |
|--------|-------------|----------------|-------------|-------|----------------|--------|
| E0 | 48 | 0.6667 | 0.6250 | -0.0417 | +0.0023 | entropy |
| E1 | 48 | 0.6042 | 0.8125 | +0.2083 | +0.0019 | gain |
| E2 | 48 | 0.6875 | 0.7292 | +0.0417 | +0.0032 | gain |
| E3 | 48 | 0.6458 | 0.7917 | +0.1458 | +0.0032 | gain |

## Results by Fold (all leagues)

| Fold | N_matches | HG Decisions | Entropy Acc_HG | Gain Acc_HG | Delta | Better |
|------|----------|-------------|----------------|-------------|-------|--------|
| 1 | 3462 | 12 | 0.5833 | 1.0000 | +0.4167 | gain |
| 2 | 3465 | 12 | 0.3333 | 0.2500 | -0.0833 | entropy |
| 3 | 3483 | 12 | 0.5000 | 0.5000 | +0.0000 | tie |
| 4 | 3438 | 12 | 0.7500 | 0.9167 | +0.1667 | gain |

## Results by League x N_HALF

| League | N_HALF | HG Decisions | Entropy Acc_HG | Gain Acc_HG | Delta | Better |
|--------|--------|-------------|----------------|-------------|-------|--------|
| E0 | 2 | 8 | 0.5000 | 0.6250 | +0.1250 | gain |
| E0 | 4 | 16 | 0.6875 | 0.5625 | -0.1250 | entropy |
| E0 | 6 | 24 | 0.7083 | 0.6667 | -0.0417 | entropy |
| E1 | 2 | 8 | 0.3750 | 0.7500 | +0.3750 | gain |
| E1 | 4 | 16 | 0.6250 | 0.8125 | +0.1875 | gain |
| E1 | 6 | 24 | 0.6667 | 0.8333 | +0.1667 | gain |
| E2 | 2 | 8 | 0.6250 | 0.6250 | +0.0000 | tie |
| E2 | 4 | 16 | 0.6875 | 0.7500 | +0.0625 | gain |
| E2 | 6 | 24 | 0.7083 | 0.7500 | +0.0417 | gain |
| E3 | 2 | 8 | 0.6250 | 0.8750 | +0.2500 | gain |
| E3 | 4 | 16 | 0.6875 | 0.8125 | +0.1250 | gain |
| E3 | 6 | 24 | 0.6250 | 0.7500 | +0.1250 | gain |

## Conclusion

**Gain-based selection is better overall** for half-guard accuracy (Acc_Top2_HG: 0.6667 vs 0.5417, delta +0.1250) across 48 half-guard decisions.

### Per-segment breakdown

- Leagues where gain wins: 3
- Leagues where entropy wins: 1
- Leagues tied: 0
- N_HALF settings where gain wins: 3
- N_HALF settings where entropy wins: 0
- N_HALF settings tied: 0

- Overall Acc_Top2_HG delta (gain - entropy): +0.1250
- Overall Combined delta (gain - entropy): -0.0001
- Total half-guard decisions: 48
- Total divergent selections: 96
