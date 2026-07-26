# Phase 3 Statistical Analysis Report

## 1. Seed Aggregation (Mean ± Std across Seeds)

| Model Group | N Seeds | Test RMSE | ± | Test QLIKE | ± | Test R² | ± |
|---|---|---|---|---|---|---|---|
| EGARCH(1,1,1) / — / — | 1 | 38.9553 | 0.0000 | 267227271.2896 | 0.0000 | -4413.7509 | 0.0000 |
| ESN-210 / — / — | 2 | 0.3335 | 0.0000 | 0.0581 | 0.0000 | 0.6764 | 0.0000 |
| ESN-500 / — / — | 2 | 0.3666 | 0.0000 | 0.0690 | 0.0000 | 0.6091 | 0.0000 |
| GARCH(1,1) / — / — | 1 | 4.5988 | 0.0000 | 137.0708 | 0.0000 | -60.5255 | 0.0000 |
| HAR-Ridge / — / — | 2 | 0.3148 | 0.0000 | 0.0537 | 0.0000 | 0.7118 | 0.0000 |
| LSTM / — / — | 1 | 0.6565 | 0.0000 | 0.1973 | 0.0000 | -0.2538 | 0.0000 |
| OnionQRC / N=10 / fully_connected | 3 | 0.3542 | 0.0021 | 0.0611 | 0.0006 | 0.6350 | 0.0044 |
| OnionQRC / N=10 / ring | 3 | 0.3726 | 0.0101 | 0.0660 | 0.0027 | 0.5958 | 0.0218 |
| OnionQRC / N=15 / ring | 3 | 0.3485 | 0.0032 | 0.0598 | 0.0008 | 0.6466 | 0.0065 |
| OnionQRC / N=5 / fully_connected | 3 | 0.3810 | 0.0021 | 0.0681 | 0.0006 | 0.5777 | 0.0048 |
| OnionQRC / N=5 / ring | 3 | 0.3780 | 0.0027 | 0.0673 | 0.0007 | 0.5843 | 0.0059 |
| Persistence / — / — | 2 | 0.3320 | 0.0000 | 0.0586 | 0.0000 | 0.6794 | 0.0000 |
| RandomFeatureRidge-210 / — / — | 2 | 0.3607 | 0.0000 | 0.0635 | 0.0000 | 0.6216 | 0.0000 |

## 2. Diebold-Mariano Tests vs. HAR-Ridge (QLIKE, two-sided)

> DM stat < 0 means model has lower QLIKE (better) than HAR-Ridge. p < 0.10 indicates significance.

| Model | Test QLIKE | HAR-Ridge QLIKE | DM Stat | p-value | Better? | Significant? |
|---|---|---|---|---|---|---|
| EGARCH(1,1,1) / — / — | 267227271.2896 | 0.0537 | 2.079 | 0.0376 | No | Yes |
| ESN-210 / — / — | 0.0581 | 0.0537 | 1.886 | 0.0593 | No | Yes |
| ESN-500 / — / — | 0.0690 | 0.0537 | 3.982 | 0.0001 | No | Yes |
| GARCH(1,1) / — / — | 137.0708 | 0.0537 | 7.936 | 0.0000 | No | Yes |
| LSTM / — / — | 0.1973 | 0.0537 | 7.942 | 0.0000 | No | Yes |
| OnionQRC / N=10 / fully_connected | 0.0606 | 0.0537 | 2.242 | 0.0250 | No | Yes |
| OnionQRC / N=10 / ring | 0.0624 | 0.0537 | 2.675 | 0.0075 | No | Yes |
| OnionQRC / N=15 / ring | 0.0591 | 0.0537 | 2.053 | 0.0401 | No | Yes |
| OnionQRC / N=5 / fully_connected | 0.0673 | 0.0537 | 3.030 | 0.0024 | No | Yes |
| OnionQRC / N=5 / ring | 0.0663 | 0.0537 | 2.875 | 0.0040 | No | Yes |
| Persistence / — / — | 0.0586 | 0.0537 | 1.209 | 0.2267 | No | No |
| RandomFeatureRidge-210 / — / — | 0.0635 | 0.0537 | 2.888 | 0.0039 | No | Yes |

## 3. Model Confidence Set (α = 10%)

**Survivors (0 models):**


**Eliminated (0 models):**


## 4. Mincer-Zarnowitz Test Summary

> An unbiased forecast has intercept=0 and slope=1. Joint p-value tests this jointly.

| Model | Intercept | Slope | p(intercept) | p(slope) | Joint p-value | Unbiased? |
|---|---|---|---|---|---|---|
| EGARCH(1,1,1) / — / — | -5.406 | -0.001 | 0.0000 | 0.0000 | 0.0000 | No |
| ESN-210 / — / — | 0.174 | 1.041 | 0.4492 | 0.3388 | 0.0528 | Yes |
| ESN-500 / — / — | 0.052 | 1.025 | 0.8425 | 0.6002 | 0.0015 | No |
| GARCH(1,1) / — / — | -8.031 | -0.266 | 0.0000 | 0.0000 | 0.0000 | No |
| HAR-Ridge / — / — | 0.167 | 1.034 | 0.4397 | 0.3843 | 0.3838 | Yes |
| LSTM / — / — | 2851.208 | 559.508 | 0.0000 | 0.0000 | 0.0000 | No |
| OnionQRC / N=10 / fully_connected | 1.297 | 1.268 | 0.0000 | 0.0000 | 0.0000 | No |
| OnionQRC / N=10 / ring | 1.430 | 1.296 | 0.0000 | 0.0000 | 0.0000 | No |
| OnionQRC / N=15 / ring | 1.164 | 1.240 | 0.0000 | 0.0000 | 0.0000 | No |
| OnionQRC / N=5 / fully_connected | 1.544 | 1.323 | 0.0000 | 0.0000 | 0.0000 | No |
| OnionQRC / N=5 / ring | 1.481 | 1.310 | 0.0000 | 0.0000 | 0.0000 | No |
| Persistence / — / — | -0.844 | 0.843 | 0.0000 | 0.0000 | 0.0000 | No |
| RandomFeatureRidge-210 / — / — | 0.840 | 1.182 | 0.0011 | 0.0002 | 0.0000 | No |
