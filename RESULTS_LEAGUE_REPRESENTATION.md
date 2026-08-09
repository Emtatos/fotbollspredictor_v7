# Diagnostic: League representation

## Scope

- Base SHA: `63bd1c471156a789ae7f7c3a3775d765485c65e7`.
- Diagnostic only; no production setting or model artifact is changed.
- Six independently trained model variants: three League representations for base features and the same three for base+odds features.
- `league_none` keeps the League column and sets every value to `-1`, matching current production behaviour exactly.
- `league_ordinal` uses E0-E3 encoded as 0-3.
- `league_onehot` removes League from X and uses four fixed binary columns.
- All variants and the bookmaker reference use exactly the same valid-odds test rows in every fold.
- Paired evaluation rows: 2978.
- Evaluation date range: 2025-01-16 to 2026-05-24.
- Paired bootstrap resamples: 2000.
- Strict PR #43 sample parity enforced: True.
- Negative delta means the candidate is better than the named reference.

## Overall metrics

| Variant | N | Accuracy | LogLoss | Brier | X_top2_rate | X_mean_prob | X_actual_rate | X_brier | Delta_LogLoss_vs_Odds | Delta_Brier_vs_Odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Odds | 2978 | 0.4882 | 1.0251 | 0.6154 | 0.5735 | 0.2617 | 0.2545 | 0.1897 | 0.0000 | 0.0000 |
| base/league_none | 2978 | 0.4738 | 1.0468 | 0.6301 | 0.4543 | 0.2700 | 0.2545 | 0.1903 | 0.0217 | 0.0147 |
| base/league_ordinal | 2978 | 0.4738 | 1.0469 | 0.6302 | 0.4553 | 0.2703 | 0.2545 | 0.1903 | 0.0218 | 0.0148 |
| base/league_onehot | 2978 | 0.4718 | 1.0465 | 0.6299 | 0.4694 | 0.2711 | 0.2545 | 0.1904 | 0.0214 | 0.0145 |
| with_odds/league_none | 2978 | 0.4822 | 1.0337 | 0.6208 | 0.5658 | 0.2698 | 0.2545 | 0.1902 | 0.0086 | 0.0054 |
| with_odds/league_ordinal | 2978 | 0.4846 | 1.0337 | 0.6208 | 0.5625 | 0.2699 | 0.2545 | 0.1902 | 0.0086 | 0.0054 |
| with_odds/league_onehot | 2978 | 0.4805 | 1.0337 | 0.6209 | 0.5668 | 0.2701 | 0.2545 | 0.1902 | 0.0086 | 0.0054 |

## Per-fold metrics

| Fold | Train_N | Test_N | Paired_N | Variant | N | Accuracy | LogLoss | Brier | X_top2_rate | X_mean_prob | X_actual_rate | X_brier | Delta_LogLoss_vs_Odds | Delta_Brier_vs_Odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1094 | 942 | 942 | Odds | 942 | 0.4851 | 1.0236 | 0.6138 | 0.6093 | 0.2620 | 0.2463 | 0.1850 | 0.0000 | 0.0000 |
| 1 | 1094 | 942 | 942 | base/league_none | 942 | 0.4660 | 1.0507 | 0.6325 | 0.5669 | 0.2856 | 0.2463 | 0.1867 | 0.0271 | 0.0187 |
| 1 | 1094 | 942 | 942 | base/league_ordinal | 942 | 0.4682 | 1.0508 | 0.6325 | 0.5679 | 0.2858 | 0.2463 | 0.1868 | 0.0272 | 0.0188 |
| 1 | 1094 | 942 | 942 | base/league_onehot | 942 | 0.4660 | 1.0491 | 0.6312 | 0.5998 | 0.2880 | 0.2463 | 0.1869 | 0.0255 | 0.0175 |
| 1 | 1094 | 942 | 942 | with_odds/league_none | 942 | 0.4766 | 1.0401 | 0.6243 | 0.6656 | 0.2841 | 0.2463 | 0.1866 | 0.0165 | 0.0106 |
| 1 | 1094 | 942 | 942 | with_odds/league_ordinal | 942 | 0.4788 | 1.0400 | 0.6242 | 0.6603 | 0.2841 | 0.2463 | 0.1866 | 0.0164 | 0.0104 |
| 1 | 1094 | 942 | 942 | with_odds/league_onehot | 942 | 0.4745 | 1.0399 | 0.6243 | 0.6805 | 0.2847 | 0.2463 | 0.1867 | 0.0163 | 0.0106 |
| 2 | 2036 | 1055 | 1055 | Odds | 1055 | 0.4758 | 1.0367 | 0.6236 | 0.5498 | 0.2647 | 0.2550 | 0.1898 | 0.0000 | 0.0000 |
| 2 | 2036 | 1055 | 1055 | base/league_none | 1055 | 0.4692 | 1.0504 | 0.6327 | 0.4322 | 0.2662 | 0.2550 | 0.1904 | 0.0137 | 0.0091 |
| 2 | 2036 | 1055 | 1055 | base/league_ordinal | 1055 | 0.4664 | 1.0503 | 0.6326 | 0.4332 | 0.2664 | 0.2550 | 0.1904 | 0.0136 | 0.0090 |
| 2 | 2036 | 1055 | 1055 | base/league_onehot | 1055 | 0.4607 | 1.0512 | 0.6333 | 0.4379 | 0.2663 | 0.2550 | 0.1904 | 0.0145 | 0.0097 |
| 2 | 2036 | 1055 | 1055 | with_odds/league_none | 1055 | 0.4758 | 1.0406 | 0.6254 | 0.5327 | 0.2674 | 0.2550 | 0.1905 | 0.0039 | 0.0019 |
| 2 | 2036 | 1055 | 1055 | with_odds/league_ordinal | 1055 | 0.4768 | 1.0406 | 0.6255 | 0.5270 | 0.2674 | 0.2550 | 0.1905 | 0.0039 | 0.0019 |
| 2 | 2036 | 1055 | 1055 | with_odds/league_onehot | 1055 | 0.4758 | 1.0404 | 0.6254 | 0.5289 | 0.2676 | 0.2550 | 0.1904 | 0.0037 | 0.0018 |
| 3 | 3091 | 981 | 981 | Odds | 981 | 0.5046 | 1.0140 | 0.6083 | 0.5647 | 0.2582 | 0.2620 | 0.1940 | 0.0000 | 0.0000 |
| 3 | 3091 | 981 | 981 | base/league_none | 981 | 0.4862 | 1.0391 | 0.6251 | 0.3700 | 0.2592 | 0.2620 | 0.1936 | 0.0251 | 0.0168 |
| 3 | 3091 | 981 | 981 | base/league_ordinal | 981 | 0.4873 | 1.0395 | 0.6254 | 0.3710 | 0.2597 | 0.2620 | 0.1936 | 0.0255 | 0.0171 |
| 3 | 3091 | 981 | 981 | base/league_onehot | 981 | 0.4893 | 1.0389 | 0.6250 | 0.3782 | 0.2600 | 0.2620 | 0.1936 | 0.0249 | 0.0167 |
| 3 | 3091 | 981 | 981 | with_odds/league_none | 981 | 0.4944 | 1.0201 | 0.6126 | 0.5056 | 0.2588 | 0.2620 | 0.1934 | 0.0062 | 0.0043 |
| 3 | 3091 | 981 | 981 | with_odds/league_ordinal | 981 | 0.4985 | 1.0201 | 0.6126 | 0.5066 | 0.2588 | 0.2620 | 0.1934 | 0.0061 | 0.0043 |
| 3 | 3091 | 981 | 981 | with_odds/league_onehot | 981 | 0.4913 | 1.0204 | 0.6127 | 0.4985 | 0.2589 | 0.2620 | 0.1934 | 0.0064 | 0.0044 |

## Per-league metrics

| League | Variant | N | Accuracy | LogLoss | Brier | X_top2_rate | X_mean_prob | X_actual_rate | X_brier | Delta_LogLoss_vs_Odds | Delta_Brier_vs_Odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | Odds | 553 | 0.5172 | 0.9969 | 0.5963 | 0.6203 | 0.2406 | 0.2495 | 0.1875 | 0.0000 | 0.0000 |
| E0 | base/league_none | 553 | 0.5063 | 1.0342 | 0.6212 | 0.4901 | 0.2669 | 0.2495 | 0.1888 | 0.0373 | 0.0248 |
| E0 | base/league_ordinal | 553 | 0.5081 | 1.0339 | 0.6210 | 0.4901 | 0.2674 | 0.2495 | 0.1889 | 0.0370 | 0.0246 |
| E0 | base/league_onehot | 553 | 0.5009 | 1.0327 | 0.6201 | 0.5027 | 0.2680 | 0.2495 | 0.1893 | 0.0358 | 0.0238 |
| E0 | with_odds/league_none | 553 | 0.5009 | 1.0119 | 0.6059 | 0.6582 | 0.2659 | 0.2495 | 0.1887 | 0.0150 | 0.0096 |
| E0 | with_odds/league_ordinal | 553 | 0.5009 | 1.0119 | 0.6059 | 0.6564 | 0.2658 | 0.2495 | 0.1887 | 0.0150 | 0.0096 |
| E0 | with_odds/league_onehot | 553 | 0.4991 | 1.0118 | 0.6059 | 0.6673 | 0.2662 | 0.2495 | 0.1887 | 0.0149 | 0.0095 |
| E1 | Odds | 793 | 0.4704 | 1.0394 | 0.6263 | 0.5939 | 0.2660 | 0.2560 | 0.1915 | 0.0000 | 0.0000 |
| E1 | base/league_none | 793 | 0.4388 | 1.0637 | 0.6426 | 0.4262 | 0.2715 | 0.2560 | 0.1921 | 0.0243 | 0.0163 |
| E1 | base/league_ordinal | 793 | 0.4401 | 1.0638 | 0.6427 | 0.4363 | 0.2720 | 0.2560 | 0.1921 | 0.0244 | 0.0164 |
| E1 | base/league_onehot | 793 | 0.4401 | 1.0634 | 0.6424 | 0.4515 | 0.2741 | 0.2560 | 0.1924 | 0.0240 | 0.0161 |
| E1 | with_odds/league_none | 793 | 0.4552 | 1.0497 | 0.6329 | 0.5473 | 0.2716 | 0.2560 | 0.1918 | 0.0103 | 0.0066 |
| E1 | with_odds/league_ordinal | 793 | 0.4578 | 1.0497 | 0.6329 | 0.5448 | 0.2717 | 0.2560 | 0.1918 | 0.0103 | 0.0066 |
| E1 | with_odds/league_onehot | 793 | 0.4565 | 1.0489 | 0.6324 | 0.5586 | 0.2726 | 0.2560 | 0.1917 | 0.0096 | 0.0061 |
| E2 | Odds | 816 | 0.4975 | 1.0195 | 0.6105 | 0.5294 | 0.2650 | 0.2463 | 0.1849 | 0.0000 | 0.0000 |
| E2 | base/league_none | 816 | 0.4841 | 1.0387 | 0.6242 | 0.4547 | 0.2708 | 0.2463 | 0.1861 | 0.0192 | 0.0137 |
| E2 | base/league_ordinal | 816 | 0.4816 | 1.0389 | 0.6244 | 0.4461 | 0.2709 | 0.2463 | 0.1861 | 0.0194 | 0.0139 |
| E2 | base/league_onehot | 816 | 0.4804 | 1.0383 | 0.6240 | 0.4645 | 0.2709 | 0.2463 | 0.1860 | 0.0188 | 0.0135 |
| E2 | with_odds/league_none | 816 | 0.4865 | 1.0279 | 0.6160 | 0.5282 | 0.2706 | 0.2463 | 0.1862 | 0.0084 | 0.0055 |
| E2 | with_odds/league_ordinal | 816 | 0.4890 | 1.0277 | 0.6158 | 0.5208 | 0.2705 | 0.2463 | 0.1862 | 0.0082 | 0.0053 |
| E2 | with_odds/league_onehot | 816 | 0.4841 | 1.0278 | 0.6160 | 0.5196 | 0.2702 | 0.2463 | 0.1861 | 0.0083 | 0.0055 |
| E3 | Odds | 816 | 0.4767 | 1.0358 | 0.6227 | 0.5662 | 0.2686 | 0.2647 | 0.1942 | 0.0000 | 0.0000 |
| E3 | base/league_none | 816 | 0.4755 | 1.0469 | 0.6298 | 0.4571 | 0.2700 | 0.2647 | 0.1937 | 0.0111 | 0.0071 |
| E3 | base/league_ordinal | 816 | 0.4755 | 1.0473 | 0.6302 | 0.4596 | 0.2701 | 0.2647 | 0.1937 | 0.0115 | 0.0074 |
| E3 | base/league_onehot | 816 | 0.4743 | 1.0475 | 0.6302 | 0.4694 | 0.2704 | 0.2647 | 0.1936 | 0.0117 | 0.0075 |
| E3 | with_odds/league_none | 816 | 0.4914 | 1.0388 | 0.6240 | 0.5588 | 0.2701 | 0.2647 | 0.1938 | 0.0029 | 0.0013 |
| E3 | with_odds/league_ordinal | 816 | 0.4951 | 1.0388 | 0.6241 | 0.5576 | 0.2701 | 0.2647 | 0.1938 | 0.0030 | 0.0014 |
| E3 | with_odds/league_onehot | 816 | 0.4877 | 1.0395 | 0.6247 | 0.5539 | 0.2704 | 0.2647 | 0.1938 | 0.0037 | 0.0019 |

## Direct League-representation comparisons

These are the decision-bearing paired comparisons. Deltas against odds alone cannot establish whether League itself helps.

| League | FeatureSet | LeagueRepresentation | N | Delta_LogLoss_vs_None | Delta_LogLoss_vs_None_CI95_L | Delta_LogLoss_vs_None_CI95_U | Delta_Brier_vs_None | Delta_Brier_vs_None_CI95_L | Delta_Brier_vs_None_CI95_U | Delta_LogLoss_vs_Ordinal | Delta_LogLoss_vs_Ordinal_CI95_L | Delta_LogLoss_vs_Ordinal_CI95_U | Delta_Brier_vs_Ordinal | Delta_Brier_vs_Ordinal_CI95_L | Delta_Brier_vs_Ordinal_CI95_U |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | base | league_ordinal | 553 | -0.0003 | -0.0013 | 0.0008 | -0.0002 | -0.0009 | 0.0006 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E0 | base | league_onehot | 553 | -0.0015 | -0.0037 | 0.0006 | -0.0010 | -0.0025 | 0.0005 | -0.0012 | -0.0030 | 0.0006 | -0.0008 | -0.0021 | 0.0004 |
| E0 | with_odds | league_ordinal | 553 | -0.0000 | -0.0003 | 0.0003 | -0.0000 | -0.0002 | 0.0002 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E0 | with_odds | league_onehot | 553 | -0.0001 | -0.0016 | 0.0013 | -0.0001 | -0.0011 | 0.0009 | -0.0001 | -0.0015 | 0.0013 | -0.0001 | -0.0011 | 0.0009 |
| E1 | base | league_ordinal | 793 | 0.0001 | -0.0005 | 0.0006 | 0.0001 | -0.0003 | 0.0004 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E1 | base | league_onehot | 793 | -0.0003 | -0.0020 | 0.0014 | -0.0002 | -0.0014 | 0.0009 | -0.0004 | -0.0019 | 0.0012 | -0.0003 | -0.0013 | 0.0008 |
| E1 | with_odds | league_ordinal | 793 | 0.0000 | -0.0002 | 0.0002 | 0.0000 | -0.0001 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E1 | with_odds | league_onehot | 793 | -0.0007 | -0.0020 | 0.0006 | -0.0005 | -0.0014 | 0.0004 | -0.0008 | -0.0020 | 0.0005 | -0.0005 | -0.0013 | 0.0003 |
| E2 | base | league_ordinal | 816 | 0.0002 | -0.0002 | 0.0006 | 0.0002 | -0.0001 | 0.0004 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E2 | base | league_onehot | 816 | -0.0004 | -0.0017 | 0.0010 | -0.0002 | -0.0011 | 0.0007 | -0.0006 | -0.0020 | 0.0008 | -0.0004 | -0.0013 | 0.0006 |
| E2 | with_odds | league_ordinal | 816 | -0.0003 | -0.0005 | -0.0000 | -0.0002 | -0.0003 | -0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E2 | with_odds | league_onehot | 816 | -0.0001 | -0.0013 | 0.0011 | 0.0000 | -0.0008 | 0.0008 | 0.0002 | -0.0010 | 0.0013 | 0.0002 | -0.0006 | 0.0009 |
| E3 | base | league_ordinal | 816 | 0.0004 | -0.0000 | 0.0009 | 0.0003 | 0.0000 | 0.0006 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E3 | base | league_onehot | 816 | 0.0007 | -0.0008 | 0.0021 | 0.0004 | -0.0006 | 0.0014 | 0.0002 | -0.0012 | 0.0017 | 0.0001 | -0.0009 | 0.0011 |
| E3 | with_odds | league_ordinal | 816 | 0.0001 | -0.0002 | 0.0003 | 0.0001 | -0.0001 | 0.0002 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| E3 | with_odds | league_onehot | 816 | 0.0007 | -0.0005 | 0.0020 | 0.0006 | -0.0002 | 0.0015 | 0.0006 | -0.0006 | 0.0018 | 0.0006 | -0.0003 | 0.0014 |

## Per-league evidence summary

- **E0 / base / league_ordinal vs none**: better point estimates; uncertainty overlaps 0 (ΔLogLoss=-0.0003, ΔBrier=-0.0002).
- **E0 / base / league_onehot vs ordinal**: better point estimates; uncertainty overlaps 0 (ΔLogLoss=-0.0012, ΔBrier=-0.0008).
- **E0 / with_odds / league_ordinal vs none**: better point estimates; uncertainty overlaps 0 (ΔLogLoss=-0.0000, ΔBrier=-0.0000).
- **E0 / with_odds / league_onehot vs ordinal**: better point estimates; uncertainty overlaps 0 (ΔLogLoss=-0.0001, ΔBrier=-0.0001).
- **E1 / base / league_ordinal vs none**: worse point estimates; uncertainty overlaps 0 (ΔLogLoss=0.0001, ΔBrier=0.0001).
- **E1 / base / league_onehot vs ordinal**: better point estimates; uncertainty overlaps 0 (ΔLogLoss=-0.0004, ΔBrier=-0.0003).
- **E1 / with_odds / league_ordinal vs none**: worse point estimates; uncertainty overlaps 0 (ΔLogLoss=0.0000, ΔBrier=0.0000).
- **E1 / with_odds / league_onehot vs ordinal**: better point estimates; uncertainty overlaps 0 (ΔLogLoss=-0.0008, ΔBrier=-0.0005).
- **E2 / base / league_ordinal vs none**: worse point estimates; uncertainty overlaps 0 (ΔLogLoss=0.0002, ΔBrier=0.0002).
- **E2 / base / league_onehot vs ordinal**: better point estimates; uncertainty overlaps 0 (ΔLogLoss=-0.0006, ΔBrier=-0.0004).
- **E2 / with_odds / league_ordinal vs none**: better on both metrics; paired 95% CIs below 0 (ΔLogLoss=-0.0003, ΔBrier=-0.0002).
- **E2 / with_odds / league_onehot vs ordinal**: worse point estimates; uncertainty overlaps 0 (ΔLogLoss=0.0002, ΔBrier=0.0002).
- **E3 / base / league_ordinal vs none**: worse point estimates; uncertainty overlaps 0 (ΔLogLoss=0.0004, ΔBrier=0.0003).
- **E3 / base / league_onehot vs ordinal**: worse point estimates; uncertainty overlaps 0 (ΔLogLoss=0.0002, ΔBrier=0.0001).
- **E3 / with_odds / league_ordinal vs none**: worse point estimates; uncertainty overlaps 0 (ΔLogLoss=0.0001, ΔBrier=0.0001).
- **E3 / with_odds / league_onehot vs ordinal**: worse point estimates; uncertainty overlaps 0 (ΔLogLoss=0.0006, ΔBrier=0.0006).

## Per-league and season metrics

| Season | League | Variant | N | Accuracy | LogLoss | Brier | X_top2_rate | X_mean_prob | X_actual_rate | X_brier | Delta_LogLoss_vs_Odds | Delta_Brier_vs_Odds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2425 | E0 | Odds | 173 | 0.5780 | 0.9493 | 0.5630 | 0.6069 | 0.2322 | 0.1965 | 0.1599 | 0.0000 | 0.0000 |
| 2425 | E0 | base/league_none | 173 | 0.5607 | 1.0296 | 0.6164 | 0.6243 | 0.2828 | 0.1965 | 0.1664 | 0.0803 | 0.0534 |
| 2425 | E0 | base/league_ordinal | 173 | 0.5607 | 1.0293 | 0.6161 | 0.6243 | 0.2832 | 0.1965 | 0.1665 | 0.0800 | 0.0531 |
| 2425 | E0 | base/league_onehot | 173 | 0.5607 | 1.0248 | 0.6128 | 0.6358 | 0.2842 | 0.1965 | 0.1675 | 0.0755 | 0.0497 |
| 2425 | E0 | with_odds/league_none | 173 | 0.5491 | 0.9977 | 0.5940 | 0.7110 | 0.2815 | 0.1965 | 0.1655 | 0.0484 | 0.0310 |
| 2425 | E0 | with_odds/league_ordinal | 173 | 0.5491 | 0.9977 | 0.5940 | 0.7110 | 0.2817 | 0.1965 | 0.1656 | 0.0483 | 0.0310 |
| 2425 | E0 | with_odds/league_onehot | 173 | 0.5491 | 0.9963 | 0.5934 | 0.7514 | 0.2822 | 0.1965 | 0.1659 | 0.0470 | 0.0304 |
| 2526 | E0 | Odds | 380 | 0.4895 | 1.0185 | 0.6115 | 0.6263 | 0.2444 | 0.2737 | 0.2000 | 0.0000 | 0.0000 |
| 2526 | E0 | base/league_none | 380 | 0.4816 | 1.0362 | 0.6233 | 0.4289 | 0.2597 | 0.2737 | 0.1990 | 0.0177 | 0.0118 |
| 2526 | E0 | base/league_ordinal | 380 | 0.4842 | 1.0360 | 0.6231 | 0.4289 | 0.2603 | 0.2737 | 0.1991 | 0.0175 | 0.0116 |
| 2526 | E0 | base/league_onehot | 380 | 0.4737 | 1.0362 | 0.6235 | 0.4421 | 0.2606 | 0.2737 | 0.1992 | 0.0177 | 0.0120 |
| 2526 | E0 | with_odds/league_none | 380 | 0.4789 | 1.0183 | 0.6114 | 0.6342 | 0.2587 | 0.2737 | 0.1992 | -0.0002 | -0.0002 |
| 2526 | E0 | with_odds/league_ordinal | 380 | 0.4789 | 1.0184 | 0.6114 | 0.6316 | 0.2586 | 0.2737 | 0.1992 | -0.0002 | -0.0001 |
| 2526 | E0 | with_odds/league_onehot | 380 | 0.4763 | 1.0188 | 0.6116 | 0.6289 | 0.2589 | 0.2737 | 0.1991 | 0.0003 | 0.0001 |
| 2425 | E1 | Odds | 241 | 0.4896 | 1.0344 | 0.6212 | 0.6349 | 0.2694 | 0.2365 | 0.1823 | 0.0000 | 0.0000 |
| 2425 | E1 | base/league_none | 241 | 0.4647 | 1.0511 | 0.6332 | 0.5477 | 0.2880 | 0.2365 | 0.1838 | 0.0166 | 0.0120 |
| 2425 | E1 | base/league_ordinal | 241 | 0.4730 | 1.0514 | 0.6334 | 0.5519 | 0.2887 | 0.2365 | 0.1839 | 0.0169 | 0.0122 |
| 2425 | E1 | base/league_onehot | 241 | 0.4772 | 1.0502 | 0.6324 | 0.6100 | 0.2942 | 0.2365 | 0.1846 | 0.0158 | 0.0112 |
| 2425 | E1 | with_odds/league_none | 241 | 0.4855 | 1.0420 | 0.6258 | 0.6639 | 0.2861 | 0.2365 | 0.1832 | 0.0076 | 0.0046 |
| 2425 | E1 | with_odds/league_ordinal | 241 | 0.4855 | 1.0421 | 0.6258 | 0.6598 | 0.2864 | 0.2365 | 0.1831 | 0.0076 | 0.0046 |
| 2425 | E1 | with_odds/league_onehot | 241 | 0.4855 | 1.0415 | 0.6255 | 0.6971 | 0.2885 | 0.2365 | 0.1834 | 0.0071 | 0.0043 |
| 2526 | E1 | Odds | 552 | 0.4620 | 1.0416 | 0.6285 | 0.5761 | 0.2645 | 0.2645 | 0.1955 | 0.0000 | 0.0000 |
| 2526 | E1 | base/league_none | 552 | 0.4275 | 1.0693 | 0.6468 | 0.3732 | 0.2644 | 0.2645 | 0.1958 | 0.0277 | 0.0182 |
| 2526 | E1 | base/league_ordinal | 552 | 0.4257 | 1.0692 | 0.6468 | 0.3859 | 0.2647 | 0.2645 | 0.1957 | 0.0277 | 0.0182 |
| 2526 | E1 | base/league_onehot | 552 | 0.4239 | 1.0692 | 0.6468 | 0.3822 | 0.2654 | 0.2645 | 0.1957 | 0.0276 | 0.0182 |
| 2526 | E1 | with_odds/league_none | 552 | 0.4420 | 1.0530 | 0.6360 | 0.4964 | 0.2653 | 0.2645 | 0.1955 | 0.0115 | 0.0075 |
| 2526 | E1 | with_odds/league_ordinal | 552 | 0.4457 | 1.0530 | 0.6360 | 0.4946 | 0.2653 | 0.2645 | 0.1955 | 0.0115 | 0.0075 |
| 2526 | E1 | with_odds/league_onehot | 552 | 0.4438 | 1.0522 | 0.6354 | 0.4982 | 0.2656 | 0.2645 | 0.1953 | 0.0106 | 0.0069 |
| 2425 | E2 | Odds | 264 | 0.4848 | 1.0209 | 0.6124 | 0.6098 | 0.2642 | 0.2348 | 0.1808 | 0.0000 | 0.0000 |
| 2425 | E2 | base/league_none | 264 | 0.4659 | 1.0415 | 0.6268 | 0.5720 | 0.2851 | 0.2348 | 0.1808 | 0.0206 | 0.0144 |
| 2425 | E2 | base/league_ordinal | 264 | 0.4659 | 1.0419 | 0.6271 | 0.5682 | 0.2849 | 0.2348 | 0.1808 | 0.0210 | 0.0147 |
| 2425 | E2 | base/league_onehot | 264 | 0.4508 | 1.0398 | 0.6257 | 0.5985 | 0.2858 | 0.2348 | 0.1806 | 0.0190 | 0.0133 |
| 2425 | E2 | with_odds/league_none | 264 | 0.4735 | 1.0303 | 0.6183 | 0.6818 | 0.2832 | 0.2348 | 0.1811 | 0.0095 | 0.0059 |
| 2425 | E2 | with_odds/league_ordinal | 264 | 0.4773 | 1.0296 | 0.6178 | 0.6742 | 0.2830 | 0.2348 | 0.1810 | 0.0088 | 0.0055 |
| 2425 | E2 | with_odds/league_onehot | 264 | 0.4735 | 1.0301 | 0.6183 | 0.6894 | 0.2826 | 0.2348 | 0.1811 | 0.0093 | 0.0060 |
| 2526 | E2 | Odds | 552 | 0.5036 | 1.0188 | 0.6097 | 0.4909 | 0.2653 | 0.2518 | 0.1868 | 0.0000 | 0.0000 |
| 2526 | E2 | base/league_none | 552 | 0.4928 | 1.0374 | 0.6230 | 0.3986 | 0.2639 | 0.2518 | 0.1887 | 0.0185 | 0.0134 |
| 2526 | E2 | base/league_ordinal | 552 | 0.4891 | 1.0374 | 0.6231 | 0.3877 | 0.2641 | 0.2518 | 0.1886 | 0.0186 | 0.0134 |
| 2526 | E2 | base/league_onehot | 552 | 0.4946 | 1.0376 | 0.6233 | 0.4004 | 0.2638 | 0.2518 | 0.1886 | 0.0187 | 0.0136 |
| 2526 | E2 | with_odds/league_none | 552 | 0.4928 | 1.0267 | 0.6149 | 0.4547 | 0.2645 | 0.2518 | 0.1886 | 0.0079 | 0.0052 |
| 2526 | E2 | with_odds/league_ordinal | 552 | 0.4946 | 1.0267 | 0.6149 | 0.4475 | 0.2646 | 0.2518 | 0.1886 | 0.0079 | 0.0052 |
| 2526 | E2 | with_odds/league_onehot | 552 | 0.4891 | 1.0267 | 0.6149 | 0.4384 | 0.2642 | 0.2518 | 0.1886 | 0.0078 | 0.0052 |
| 2425 | E3 | Odds | 264 | 0.4205 | 1.0651 | 0.6417 | 0.5871 | 0.2726 | 0.2992 | 0.2082 | 0.0000 | 0.0000 |
| 2425 | E3 | base/league_none | 264 | 0.4053 | 1.0733 | 0.6481 | 0.5417 | 0.2858 | 0.2992 | 0.2086 | 0.0082 | 0.0064 |
| 2425 | E3 | base/league_ordinal | 264 | 0.4053 | 1.0731 | 0.6480 | 0.5455 | 0.2856 | 0.2992 | 0.2086 | 0.0080 | 0.0063 |
| 2425 | E3 | base/league_onehot | 264 | 0.4091 | 1.0731 | 0.6479 | 0.5682 | 0.2869 | 0.2992 | 0.2082 | 0.0080 | 0.0062 |
| 2425 | E3 | with_odds/league_none | 264 | 0.4242 | 1.0760 | 0.6488 | 0.6212 | 0.2848 | 0.2992 | 0.2089 | 0.0109 | 0.0071 |
| 2425 | E3 | with_odds/league_ordinal | 264 | 0.4280 | 1.0761 | 0.6489 | 0.6136 | 0.2846 | 0.2992 | 0.2090 | 0.0110 | 0.0072 |
| 2425 | E3 | with_odds/league_onehot | 264 | 0.4167 | 1.0768 | 0.6495 | 0.6098 | 0.2848 | 0.2992 | 0.2089 | 0.0117 | 0.0079 |
| 2526 | E3 | Odds | 552 | 0.5036 | 1.0218 | 0.6137 | 0.5562 | 0.2667 | 0.2482 | 0.1875 | 0.0000 | 0.0000 |
| 2526 | E3 | base/league_none | 552 | 0.5091 | 1.0342 | 0.6211 | 0.4167 | 0.2625 | 0.2482 | 0.1866 | 0.0124 | 0.0074 |
| 2526 | E3 | base/league_ordinal | 552 | 0.5091 | 1.0349 | 0.6216 | 0.4185 | 0.2628 | 0.2482 | 0.1866 | 0.0131 | 0.0080 |
| 2526 | E3 | base/league_onehot | 552 | 0.5054 | 1.0353 | 0.6218 | 0.4221 | 0.2624 | 0.2482 | 0.1866 | 0.0135 | 0.0081 |
| 2526 | E3 | with_odds/league_none | 552 | 0.5236 | 1.0209 | 0.6122 | 0.5290 | 0.2631 | 0.2482 | 0.1866 | -0.0009 | -0.0015 |
| 2526 | E3 | with_odds/league_ordinal | 552 | 0.5272 | 1.0210 | 0.6123 | 0.5308 | 0.2632 | 0.2482 | 0.1866 | -0.0008 | -0.0014 |
| 2526 | E3 | with_odds/league_onehot | 552 | 0.5217 | 1.0216 | 0.6128 | 0.5272 | 0.2635 | 0.2482 | 0.1866 | -0.0002 | -0.0009 |

## Overall p_X calibration

| Variant | Bin | N | MeanPredictedX | ObservedXRate |
| --- | --- | --- | --- | --- |
| Odds | 0.00-0.20 | 161 | 0.1715 | 0.1925 |
| Odds | 0.20-0.25 | 622 | 0.2328 | 0.2508 |
| Odds | 0.25-0.30 | 2069 | 0.2747 | 0.2557 |
| Odds | 0.30+ | 126 | 0.3069 | 0.3333 |
| base/league_none | 0.00-0.20 | 0 | NA | NA |
| base/league_none | 0.20-0.25 | 315 | 0.2400 | 0.2794 |
| base/league_none | 0.25-0.30 | 2424 | 0.2702 | 0.2504 |
| base/league_none | 0.30+ | 239 | 0.3085 | 0.2636 |
| base/league_ordinal | 0.00-0.20 | 0 | NA | NA |
| base/league_ordinal | 0.20-0.25 | 271 | 0.2399 | 0.2694 |
| base/league_ordinal | 0.25-0.30 | 2444 | 0.2696 | 0.2520 |
| base/league_ordinal | 0.30+ | 263 | 0.3086 | 0.2624 |
| base/league_onehot | 0.00-0.20 | 1 | 0.1953 | 0.0000 |
| base/league_onehot | 0.20-0.25 | 309 | 0.2399 | 0.2557 |
| base/league_onehot | 0.25-0.30 | 2324 | 0.2689 | 0.2530 |
| base/league_onehot | 0.30+ | 344 | 0.3138 | 0.2645 |
| with_odds/league_none | 0.00-0.20 | 0 | NA | NA |
| with_odds/league_none | 0.20-0.25 | 322 | 0.2395 | 0.2484 |
| with_odds/league_none | 0.25-0.30 | 2535 | 0.2721 | 0.2529 |
| with_odds/league_none | 0.30+ | 121 | 0.3043 | 0.3058 |
| with_odds/league_ordinal | 0.00-0.20 | 0 | NA | NA |
| with_odds/league_ordinal | 0.20-0.25 | 321 | 0.2395 | 0.2430 |
| with_odds/league_ordinal | 0.25-0.30 | 2540 | 0.2721 | 0.2539 |
| with_odds/league_ordinal | 0.30+ | 117 | 0.3043 | 0.2991 |
| with_odds/league_onehot | 0.00-0.20 | 0 | NA | NA |
| with_odds/league_onehot | 0.20-0.25 | 317 | 0.2400 | 0.2208 |
| with_odds/league_onehot | 0.25-0.30 | 2503 | 0.2717 | 0.2557 |
| with_odds/league_onehot | 0.30+ | 158 | 0.3059 | 0.3038 |

Per-league p_X calibration is saved in the generated `_X_CALIBRATION.csv` artifact.

## Interpretation guardrails

- LogLoss and Brier are the primary model-selection metrics.
- Representation decisions require the direct paired deltas against `league_none`; comparison with odds is contextual only.
- Do not declare a representation winner when LogLoss/Brier conflict or paired confidence intervals overlap 0.
- `X_top2_rate` counts ties for second place as top-2.
- Small third-decimal differences are expected and must not trigger a production change without paired evidence.
