# Half-Guard Coupon Evaluation: Entropy vs Gain

## Setup

### Coupon definition

- **Coupon size**: 8 matches per coupon
- Matches are grouped chronologically into consecutive coupons within each test fold
- Incomplete coupons (< coupon_size) are dropped
- Half-guard selection is done **per coupon** (not globally), matching real usage

### Evaluation rules

- **Single matches** (not half-guarded): correct if top-1 prediction matches actual result
- **Half-guard matches**: correct if actual result is in top-2 predictions
- **Coupon hit**: ALL matches in the coupon are correct
- **HG rescue**: coupon passes with half-guards but would have failed with singles only

### Parameters

- Walk-forward folds: 5
- N_HALF values: 2, 4, 6
- Leagues: E0, E1, E2, E3
- Total coupons evaluated: 1152

## Overall Summary

| Metric | Entropy | Gain | Delta (gain-entropy) | Better |
|--------|---------|------|----------------------|--------|
| Ticket hit rate | 0.0165 | 0.0165 | +0.0000 | tie |
| Mean correct/coupon | 4.76 | 4.77 | +0.0061 | tie |
| HG rescue rate | 0.0148 | 0.0148 | +0.0000 | tie |
| HG rescues (count) | 17 | 17 | +0 | tie |

- Coupons where gain and entropy selected different half-guards: **352** / 1152 (30.6%)

## Results by N_HALF

| N_HALF | Coupons | Entropy Hit Rate | Gain Hit Rate | Delta | Entropy Rescues | Gain Rescues | Divergent |
|--------|---------|-----------------|-------------- |-------|-----------------|--------------|-----------|
| 2 | 576 | 0.0069 | 0.0104 | +0.0035 | 3 | 5 | 201 |
| 4 | 576 | 0.0260 | 0.0226 | -0.0035 | 14 | 12 | 151 |

## Mean Correct per Coupon by N_HALF

| N_HALF | Entropy Mean | Gain Mean | Delta |
|--------|-------------|-----------|-------|
| 2 | 4.46 | 4.48 | +0.0191 |
| 4 | 5.06 | 5.05 | -0.0069 |

## Results by Fold

| Fold | Coupons | Entropy Hit Rate | Gain Hit Rate | Delta | Better |
|------|---------|-----------------|-------------- |-------|--------|
| 1 | 288 | 0.0174 | 0.0174 | +0.0000 | tie |
| 2 | 288 | 0.0104 | 0.0139 | +0.0035 | gain |
| 3 | 290 | 0.0276 | 0.0276 | +0.0000 | tie |
| 4 | 286 | 0.0105 | 0.0070 | -0.0035 | entropy |

## Detailed Per-Cell Results

| Fold | N_HALF | Coupons | Entropy Hits | Gain Hits | E Rescues | G Rescues | Divergent |
|------|--------|---------|-------------|---------- |-----------|-----------|-----------|
| 1 | 2 | 144 | 1/144 | 2/144 | 0 | 1 | 73 |
| 1 | 4 | 144 | 4/144 | 3/144 | 3 | 2 | 42 |
| 2 | 2 | 144 | 0/144 | 1/144 | 0 | 1 | 43 |
| 2 | 4 | 144 | 3/144 | 3/144 | 3 | 3 | 37 |
| 3 | 2 | 145 | 3/145 | 3/145 | 3 | 3 | 34 |
| 3 | 4 | 145 | 5/145 | 5/145 | 5 | 5 | 27 |
| 4 | 2 | 143 | 0/143 | 0/143 | 0 | 0 | 51 |
| 4 | 4 | 143 | 3/143 | 2/143 | 3 | 2 | 45 |

## Conclusion

### Verdict: unchanged

**No meaningful difference** between gain and entropy in practical coupon outcomes.

### Key numbers

- Ticket hit rate delta (gain - entropy): **+0.0000**
- Mean correct/coupon delta: **+0.0061**
- HG rescue rate delta: **+0.0000**
- HG rescue count: gain=17, entropy=17
- Divergent selections: 352/1152 (30.6%)

### Per-N_HALF verdict

- N_HALF=2: ticket hit delta +0.0035 -> **gain**
- N_HALF=4: ticket hit delta -0.0035 -> **entropy**
