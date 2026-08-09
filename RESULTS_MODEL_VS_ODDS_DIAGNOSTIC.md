# Diagnostic: Model vs Bookmaker Odds

## Scope

- Diagnostic only. No production defaults, weights, UI, parser, Elo, Poisson, or streck logic changed.
- Requested base SHA: `10b2d438090ae513de2dfb23c12c96bafb77ff1b`.
- Walk-forward date segments: 4; evaluated folds: [1, 2, 3].
- Paired out-of-fold evaluation rows: 2978.
- Evaluation date range: 2025-01-16 to 2026-05-24.
- Paired bootstrap resamples per delta: 2000.
- All three variants are evaluated on exactly the same rows with valid historical odds.
- `Odds` uses the fair implied probabilities already produced by FeatureBuilder (Bet365 when complete, otherwise Pinnacle).
- `Model` uses current `FEATURE_COLUMNS` without odds.
- `Model+Odds` is separately trained inside every fold with `ALL_FEATURE_COLUMNS`.
- Negative Delta_LogLoss/Brier_vs_Odds means the candidate is better than odds.

## Known limitation — League encoding and production parity

`FeatureBuilder` returns numeric `League` values `0.0–3.0`. Current production training and inference pass those numeric values through `encode_league()`, which only recognizes the string values `E0–E3`; the production League feature therefore collapses to the constant value `−1` and is functionally unused.

This diagnostic decodes the numeric values back to `E0–E3` before the benchmark trainer encodes them to `0–3`. Consequently, the `Model` and `Model+Odds` variants in this report use an active ordinal League feature and are not exact production-parity measurements.

The results should therefore be interpreted as a benchmark of the current feature set with an active League representation, not as a bit-for-bit evaluation of the currently deployed production model.

## Per-fold metrics and warm-up diagnostic

| Fold | Train_N | Test_N | Paired_N | Variant | N | Accuracy | LogLoss | Brier | X_precision | X_recall | Delta_LogLoss_vs_Odds | Delta_Brier_vs_Odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1094 | 942 | 942 | Model | 942 | 0.4682 | 1.0508 | 0.6325 | 0.0000 | 0.0000 | 0.0272 | 0.0188 |
| 1 | 1094 | 942 | 942 | Odds | 942 | 0.4851 | 1.0236 | 0.6138 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1 | 1094 | 942 | 942 | Model+Odds | 942 | 0.4788 | 1.0400 | 0.6242 | 0.0000 | 0.0000 | 0.0164 | 0.0104 |
| 2 | 2036 | 1055 | 1055 | Model | 1055 | 0.4664 | 1.0503 | 0.6326 | 0.0000 | 0.0000 | 0.0136 | 0.0090 |
| 2 | 2036 | 1055 | 1055 | Odds | 1055 | 0.4758 | 1.0367 | 0.6236 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 2036 | 1055 | 1055 | Model+Odds | 1055 | 0.4768 | 1.0406 | 0.6255 | 0.0000 | 0.0000 | 0.0039 | 0.0019 |
| 3 | 3091 | 981 | 981 | Model | 981 | 0.4873 | 1.0395 | 0.6254 | 0.0000 | 0.0000 | 0.0255 | 0.0171 |
| 3 | 3091 | 981 | 981 | Odds | 981 | 0.5046 | 1.0140 | 0.6083 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 3091 | 981 | 981 | Model+Odds | 981 | 0.4985 | 1.0201 | 0.6126 | 0.0000 | 0.0000 | 0.0061 | 0.0043 |

Early folds have materially less historical training data because this is an expanding-window test. If a model deficit versus odds shrinks as fold index and Train_N increase, treat data starvation/warm-up as a plausible explanation rather than concluding from the aggregate alone that the model has no signal.

## Per-league metrics

| League | Variant | N | Accuracy | LogLoss | Brier | X_precision | X_recall | Delta_LogLoss_vs_Odds | Delta_Brier_vs_Odds | Delta_LogLoss_CI95_L | Delta_LogLoss_CI95_U | Delta_Brier_CI95_L | Delta_Brier_CI95_U |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | Model | 553 | 0.5081 | 1.0339 | 0.6210 | 0.0000 | 0.0000 | 0.0370 | 0.0246 | 0.0168 | 0.0574 | 0.0099 | 0.0394 |
| E0 | Odds | 553 | 0.5172 | 0.9969 | 0.5963 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E0 | Model+Odds | 553 | 0.5009 | 1.0119 | 0.6059 | 0.0000 | 0.0000 | 0.0150 | 0.0096 | 0.0020 | 0.0272 | 0.0007 | 0.0182 |
| E1 | Model | 793 | 0.4401 | 1.0638 | 0.6427 | 0.0000 | 0.0000 | 0.0244 | 0.0164 | 0.0096 | 0.0382 | 0.0059 | 0.0261 |
| E1 | Odds | 793 | 0.4704 | 1.0394 | 0.6263 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E1 | Model+Odds | 793 | 0.4578 | 1.0497 | 0.6329 | 0.0000 | 0.0000 | 0.0103 | 0.0066 | 0.0023 | 0.0180 | 0.0012 | 0.0119 |
| E2 | Model | 816 | 0.4816 | 1.0389 | 0.6244 | 0.0000 | 0.0000 | 0.0194 | 0.0139 | 0.0072 | 0.0310 | 0.0053 | 0.0222 |
| E2 | Odds | 816 | 0.4975 | 1.0195 | 0.6105 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E2 | Model+Odds | 816 | 0.4890 | 1.0277 | 0.6158 | 0.0000 | 0.0000 | 0.0082 | 0.0053 | 0.0011 | 0.0147 | 0.0006 | 0.0102 |
| E3 | Model | 816 | 0.4755 | 1.0473 | 0.6302 | 0.0000 | 0.0000 | 0.0115 | 0.0074 | 0.0009 | 0.0227 | 0.0000 | 0.0153 |
| E3 | Odds | 816 | 0.4767 | 1.0358 | 0.6227 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E3 | Model+Odds | 816 | 0.4951 | 1.0388 | 0.6241 | 0.0000 | 0.0000 | 0.0030 | 0.0014 | -0.0036 | 0.0103 | -0.0034 | 0.0061 |

## Per-league evidence summary

- **E0 / Model**: worse on both; paired 95% CIs above 0 (N=553, ΔLogLoss=0.0370, ΔBrier=0.0246).
- **E0 / Model+Odds**: worse on both; paired 95% CIs above 0 (N=553, ΔLogLoss=0.0150, ΔBrier=0.0096).
- **E1 / Model**: worse on both; paired 95% CIs above 0 (N=793, ΔLogLoss=0.0244, ΔBrier=0.0164).
- **E1 / Model+Odds**: worse on both; paired 95% CIs above 0 (N=793, ΔLogLoss=0.0103, ΔBrier=0.0066).
- **E2 / Model**: worse on both; paired 95% CIs above 0 (N=816, ΔLogLoss=0.0194, ΔBrier=0.0139).
- **E2 / Model+Odds**: worse on both; paired 95% CIs above 0 (N=816, ΔLogLoss=0.0082, ΔBrier=0.0053).
- **E3 / Model**: worse on both; paired 95% CIs above 0 (N=816, ΔLogLoss=0.0115, ΔBrier=0.0074).
- **E3 / Model+Odds**: worse on both point estimates; uncertainty overlaps 0 (N=816, ΔLogLoss=0.0030, ΔBrier=0.0014).

## Per-season metrics

| Season | League | Variant | N | Accuracy | LogLoss | Brier | X_precision | X_recall | Delta_LogLoss_vs_Odds | Delta_Brier_vs_Odds | Delta_LogLoss_CI95_L | Delta_LogLoss_CI95_U | Delta_Brier_CI95_L | Delta_Brier_CI95_U |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2425 | E0 | Model | 173 | 0.5607 | 1.0293 | 0.6161 | 0.0000 | 0.0000 | 0.0800 | 0.0531 | 0.0390 | 0.1236 | 0.0241 | 0.0839 |
| 2425 | E0 | Odds | 173 | 0.5780 | 0.9493 | 0.5630 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2425 | E0 | Model+Odds | 173 | 0.5491 | 0.9977 | 0.5940 | 0.0000 | 0.0000 | 0.0483 | 0.0310 | 0.0165 | 0.0790 | 0.0086 | 0.0521 |
| 2526 | E0 | Model | 380 | 0.4842 | 1.0360 | 0.6231 | 0.0000 | 0.0000 | 0.0175 | 0.0116 | -0.0056 | 0.0421 | -0.0049 | 0.0290 |
| 2526 | E0 | Odds | 380 | 0.4895 | 1.0185 | 0.6115 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2526 | E0 | Model+Odds | 380 | 0.4789 | 1.0184 | 0.6114 | 0.0000 | 0.0000 | -0.0002 | -0.0001 | -0.0128 | 0.0123 | -0.0081 | 0.0078 |
| 2425 | E1 | Model | 241 | 0.4730 | 1.0514 | 0.6334 | 0.0000 | 0.0000 | 0.0169 | 0.0122 | -0.0100 | 0.0431 | -0.0065 | 0.0305 |
| 2425 | E1 | Odds | 241 | 0.4896 | 1.0344 | 0.6212 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2425 | E1 | Model+Odds | 241 | 0.4855 | 1.0421 | 0.6258 | 0.0000 | 0.0000 | 0.0076 | 0.0046 | -0.0106 | 0.0247 | -0.0079 | 0.0170 |
| 2526 | E1 | Model | 552 | 0.4257 | 1.0692 | 0.6468 | 0.0000 | 0.0000 | 0.0277 | 0.0182 | 0.0091 | 0.0449 | 0.0052 | 0.0303 |
| 2526 | E1 | Odds | 552 | 0.4620 | 1.0416 | 0.6285 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2526 | E1 | Model+Odds | 552 | 0.4457 | 1.0530 | 0.6360 | 0.0000 | 0.0000 | 0.0115 | 0.0075 | 0.0033 | 0.0199 | 0.0022 | 0.0127 |
| 2425 | E2 | Model | 264 | 0.4659 | 1.0419 | 0.6271 | 0.0000 | 0.0000 | 0.0210 | 0.0147 | -0.0035 | 0.0434 | -0.0029 | 0.0308 |
| 2425 | E2 | Odds | 264 | 0.4848 | 1.0209 | 0.6124 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2425 | E2 | Model+Odds | 264 | 0.4773 | 1.0296 | 0.6178 | 0.0000 | 0.0000 | 0.0088 | 0.0055 | -0.0067 | 0.0236 | -0.0051 | 0.0163 |
| 2526 | E2 | Model | 552 | 0.4891 | 1.0374 | 0.6231 | 0.0000 | 0.0000 | 0.0186 | 0.0134 | 0.0059 | 0.0325 | 0.0046 | 0.0233 |
| 2526 | E2 | Odds | 552 | 0.5036 | 1.0188 | 0.6097 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2526 | E2 | Model+Odds | 552 | 0.4946 | 1.0267 | 0.6149 | 0.0000 | 0.0000 | 0.0079 | 0.0052 | 0.0009 | 0.0148 | 0.0008 | 0.0097 |
| 2425 | E3 | Model | 264 | 0.4053 | 1.0731 | 0.6480 | 0.0000 | 0.0000 | 0.0080 | 0.0063 | -0.0115 | 0.0280 | -0.0075 | 0.0207 |
| 2425 | E3 | Odds | 264 | 0.4205 | 1.0651 | 0.6417 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2425 | E3 | Model+Odds | 264 | 0.4280 | 1.0761 | 0.6489 | 0.0000 | 0.0000 | 0.0110 | 0.0072 | -0.0016 | 0.0253 | -0.0019 | 0.0161 |
| 2526 | E3 | Model | 552 | 0.5091 | 1.0349 | 0.6216 | 0.0000 | 0.0000 | 0.0131 | 0.0080 | 0.0002 | 0.0265 | -0.0010 | 0.0172 |
| 2526 | E3 | Odds | 552 | 0.5036 | 1.0218 | 0.6137 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2526 | E3 | Model+Odds | 552 | 0.5272 | 1.0210 | 0.6123 | 0.0000 | 0.0000 | -0.0008 | -0.0014 | -0.0088 | 0.0076 | -0.0070 | 0.0041 |

## Interpretation guardrails

- Accuracy is secondary; LogLoss and Brier are the primary probability-quality metrics.
- X precision/recall are reported separately because draw performance can be hidden by overall accuracy.
- A point estimate alone is not treated as proof. When bootstrap is enabled, paired intervals show uncertainty in the delta vs odds.
- This report does not activate `USE_ODDS_FEATURES`, change `combined_probability.py`, or recommend production weights.
