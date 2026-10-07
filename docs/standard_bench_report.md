# Standard-Bench Evaluation (by granularity)

Reported **per granularity (frequency)** — not per dataset, not per model. Each table aggregates every dataset at one frequency (across all domains). Mean of per-dataset aggregate metrics over rolling windows (step subsampling = 1). Lower is better for MAE/SMAPE/MAPE/RMSE/MASE. Splits: **test** (2021-01-01 .. 2023-12-31), **validation** (2019-01-01 .. 2020-12-31).

## TEST split (2021-01-01 .. 2023-12-31)

### DAY granularity

| model | n | MAE | SMAPE | MAPE | RMSE | MASE |
|---|---:|---:|---:|---:|---:|---:|
| arima | 92 | 2.5988 | 5.565 | 6.27 | 2.9319 | 5.4234 |
| chronos-2-finetuned | 3 | 1.1493 | 5.3038 | 6.1667 | 1.3146 | 4.1783 |
| chronos-bolt-base | 92 | 2.3973 | 5.2203 | 5.6333 | 2.7137 | 4.996 |
| moving_average | 92 | 3.05 | 6.6676 | 8.0206 | 3.3184 | 6.4954 |
| naive | 92 | 2.3412 | 5.0349 | 5.6179 | 2.653 | 4.8584 |
| random_walk | 92 | 2.3412 | 5.0349 | 5.6179 | 2.653 | 4.8584 |
| seasonal_naive | 92 | 3.3767 | 7.3993 | 8.7524 | 3.6389 | 7.2195 |
| timesfm-2.5-200m-pytorch | 92 | 2.4532 | 5.3155 | 6.5674 | 2.7707 | 5.0966 |

_by domain within this frequency_

| domain | model | MAE | SMAPE | MASE |
|---|---|---:|---:|---:|
| bond | arima | 0.1236 | 14.433 | 5.0785 |
| bond | chronos-2-finetuned | 0.10392 | 11.698 | 3.9075 |
| bond | chronos-bolt-base | 0.11025 | 13.567 | 4.631 |
| bond | moving_average | 0.14729 | 17.362 | 6.1497 |
| bond | naive | 0.10404 | 13.042 | 4.3997 |
| bond | random_walk | 0.10404 | 13.042 | 4.3997 |
| bond | seasonal_naive | 0.17104 | 19.338 | 7.1415 |
| bond | timesfm-2.5-200m-pytorch | 0.10849 | 13.793 | 4.6276 |
| equity | arima | 4.6104 | 3.4979 | 6.4792 |
| equity | chronos-2-finetuned | 2.5685 | 3.3385 | 6.837 |
| equity | chronos-bolt-base | 4.2431 | 3.2598 | 5.9844 |
| equity | moving_average | 5.3916 | 4.1322 | 7.6056 |
| equity | naive | 4.1453 | 3.1655 | 5.8852 |
| equity | random_walk | 4.1453 | 3.1655 | 5.8852 |
| equity | seasonal_naive | 5.9711 | 4.5571 | 8.363 |
| equity | timesfm-2.5-200m-pytorch | 4.3381 | 3.3228 | 6.1237 |
| fx | arima | 0.29243 | 0.97769 | 3.1632 |
| fx | chronos-2-finetuned | 0.77539 | 0.87451 | 1.7904 |
| fx | chronos-bolt-base | 0.2988 | 0.93986 | 2.9266 |
| fx | moving_average | 0.38874 | 1.2424 | 4.1004 |
| fx | naive | 0.29171 | 0.90097 | 2.7958 |
| fx | random_walk | 0.29171 | 0.90097 | 2.7958 |
| fx | seasonal_naive | 0.41697 | 1.3723 | 4.4464 |
| fx | timesfm-2.5-200m-pytorch | 0.31992 | 0.97208 | 3.0448 |

### HOUR granularity

| model | n | MAE | SMAPE | MAPE | RMSE | MASE |
|---|---:|---:|---:|---:|---:|---:|
| arima | 9 | 0.04671 | 0.30295 | 0.30294 | 0.054777 | 3.9136 |
| chronos-2-finetuned | 0 | - | - | - | - | - |
| chronos-bolt-base | 9 | 0.044264 | 0.29369 | 0.29376 | 0.051978 | 3.7991 |
| moving_average | 9 | 0.071207 | 0.43703 | 0.43714 | 0.076604 | 5.6668 |
| naive | 9 | 0.04163 | 0.27135 | 0.27138 | 0.049061 | 3.51 |
| random_walk | 9 | 0.04163 | 0.27135 | 0.27138 | 0.049061 | 3.51 |
| seasonal_naive | 9 | 0.078556 | 0.48356 | 0.48371 | 0.083594 | 6.2807 |
| timesfm-2.5-200m-pytorch | 9 | 0.044206 | 0.28665 | 0.2867 | 0.051671 | 3.7083 |

_by domain within this frequency_

| domain | model | MAE | SMAPE | MASE |
|---|---|---:|---:|---:|
| fx | arima | 0.04671 | 0.30295 | 3.9136 |
| fx | chronos-2-finetuned | - | - | - |
| fx | chronos-bolt-base | 0.044264 | 0.29369 | 3.7991 |
| fx | moving_average | 0.071207 | 0.43703 | 5.6668 |
| fx | naive | 0.04163 | 0.27135 | 3.51 |
| fx | random_walk | 0.04163 | 0.27135 | 3.51 |
| fx | seasonal_naive | 0.078556 | 0.48356 | 6.2807 |
| fx | timesfm-2.5-200m-pytorch | 0.044206 | 0.28665 | 3.7083 |

## VALIDATION split (2019-01-01 .. 2020-12-31)

### DAY granularity

| model | n | MAE | SMAPE | MAPE | RMSE | MASE |
|---|---:|---:|---:|---:|---:|---:|
| arima | 92 | 2.1795 | 7.8868 | 11.96 | 2.5389 | 4.0334 |
| chronos-2-finetuned | 3 | 0.66943 | 12.339 | 21.397 | 0.77517 | 2.3695 |
| chronos-bolt-base | 92 | 2.2079 | 7.3919 | 11.776 | 2.538 | 4.0525 |
| moving_average | 92 | 3.1958 | 9.529 | 14.444 | 3.4554 | 5.6446 |
| naive | 92 | 2.0843 | 7.489 | 11.595 | 2.4183 | 3.8816 |
| random_walk | 92 | 2.0843 | 7.489 | 11.595 | 2.4183 | 3.8816 |
| seasonal_naive | 92 | 3.7494 | 10.516 | 14.915 | 3.9902 | 6.493 |
| timesfm-2.5-200m-pytorch | 92 | 2.5172 | 8.2461 | 12.974 | 2.8367 | 4.5358 |

_by domain within this frequency_

| domain | model | MAE | SMAPE | MASE |
|---|---|---:|---:|---:|
| bond | arima | 0.075407 | 22.891 | 3.2682 |
| bond | chronos-2-finetuned | 0.05855 | 33.875 | 2.2017 |
| bond | chronos-bolt-base | 0.071313 | 20.822 | 3.0992 |
| bond | moving_average | 0.09898 | 25.174 | 4.211 |
| bond | naive | 0.071517 | 21.678 | 3.0358 |
| bond | random_walk | 0.071517 | 21.678 | 3.0358 |
| bond | seasonal_naive | 0.11029 | 26.818 | 4.632 |
| bond | timesfm-2.5-200m-pytorch | 0.082361 | 22.73 | 3.579 |
| equity | arima | 3.8909 | 4.1451 | 5.1821 |
| equity | chronos-2-finetuned | 1.1472 | 2.0762 | 3.0538 |
| equity | chronos-bolt-base | 3.9431 | 4.143 | 5.2977 |
| equity | moving_average | 5.7221 | 6.0579 | 7.4366 |
| equity | naive | 3.7203 | 3.9645 | 5.0531 |
| equity | random_walk | 3.7203 | 3.9645 | 5.0531 |
| equity | seasonal_naive | 6.7238 | 7.119 | 8.7188 |
| equity | timesfm-2.5-200m-pytorch | 4.507 | 4.8686 | 5.9404 |
| fx | arima | 0.21518 | 0.73673 | 2.0035 |
| fx | chronos-2-finetuned | 0.8025 | 1.0676 | 1.853 |
| fx | chronos-bolt-base | 0.21987 | 0.74046 | 1.9879 |
| fx | moving_average | 0.28632 | 0.99712 | 2.7416 |
| fx | naive | 0.20861 | 0.69268 | 1.8829 |
| fx | random_walk | 0.20861 | 0.69268 | 1.8829 |
| fx | seasonal_naive | 0.31652 | 1.0778 | 2.9757 |
| fx | timesfm-2.5-200m-pytorch | 0.22082 | 0.75763 | 2.0769 |

### HOUR granularity

| model | n | MAE | SMAPE | MAPE | RMSE | MASE |
|---|---:|---:|---:|---:|---:|---:|
| arima | 9 | 0.027973 | 0.24776 | 0.24774 | 0.03264 | 3.1891 |
| chronos-2-finetuned | 0 | - | - | - | - | - |
| chronos-bolt-base | 9 | 0.026222 | 0.24248 | 0.24251 | 0.030788 | 3.1262 |
| moving_average | 9 | 0.042182 | 0.37427 | 0.37433 | 0.045667 | 4.8147 |
| naive | 9 | 0.024901 | 0.22584 | 0.22583 | 0.029277 | 2.9092 |
| random_walk | 9 | 0.024901 | 0.22584 | 0.22583 | 0.029277 | 2.9092 |
| seasonal_naive | 9 | 0.048127 | 0.42949 | 0.42958 | 0.051469 | 5.5244 |
| timesfm-2.5-200m-pytorch | 9 | 0.02601 | 0.24146 | 0.24146 | 0.030518 | 3.1078 |

_by domain within this frequency_

| domain | model | MAE | SMAPE | MASE |
|---|---|---:|---:|---:|
| fx | arima | 0.027973 | 0.24776 | 3.1891 |
| fx | chronos-2-finetuned | - | - | - |
| fx | chronos-bolt-base | 0.026222 | 0.24248 | 3.1262 |
| fx | moving_average | 0.042182 | 0.37427 | 4.8147 |
| fx | naive | 0.024901 | 0.22584 | 2.9092 |
| fx | random_walk | 0.024901 | 0.22584 | 2.9092 |
| fx | seasonal_naive | 0.048127 | 0.42949 | 5.5244 |
| fx | timesfm-2.5-200m-pytorch | 0.02601 | 0.24146 | 3.1078 |

Total runs: 1420; errors: 0 ([])
