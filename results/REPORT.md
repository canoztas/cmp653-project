# Comprehensive Experimental Report

Generated from 11 experiment CSVs.

## Model Validation (R6 main): alpha sweep at fixed k

Total trials: 720 (6 alphas x 4 k-values x 30 trials)

Mean predicted vs empirical mean by (alpha, k) -- the model claim is mean-vs-mean accuracy:

```
            pred  emp_mean  emp_std  mean_err_abs  mean_err_pct
alpha k                                                        
0.0   10   5.502     5.533    0.730         0.031          0.56
      25   6.852     6.767    0.430         0.085          1.24
      50   6.997     7.000    0.000         0.003          0.04
      100  7.000     7.000    0.000         0.000          0.00
0.5   10   5.300     5.233    1.006         0.067          1.26
      25   6.729     6.733    0.450         0.004          0.06
      50   6.984     6.933    0.254         0.051          0.73
      100  7.000     7.000    0.000         0.000          0.00
1.0   10   4.730     4.700    0.877         0.030          0.63
      25   6.317     6.300    0.794         0.017          0.27
      50   6.880     6.900    0.305         0.020          0.29
      100  6.995     7.000    0.000         0.005          0.07
1.5   10   3.977     3.867    0.860         0.110          2.77
      25   5.570     5.700    0.952         0.130          2.33
      50   6.483     6.400    0.675         0.083          1.28
      100  6.912     6.933    0.254         0.021          0.30
2.0   10   3.246     3.133    0.860         0.113          3.48
      25   4.640     4.700    0.794         0.060          1.29
      50   5.693     5.733    0.691         0.040          0.70
      100  6.503     6.433    0.679         0.070          1.08
3.0   10   2.191     2.067    0.691         0.124          5.66
      25   3.070     3.333    0.922         0.263          8.57
      50   3.849     3.900    0.845         0.051          1.33
      100  4.718     4.667    0.959         0.051          1.08
```

Cells with |pred - empirical_mean| / pred >= 3%: 3 / 24
Worst case mean-vs-mean relative error: 8.57%

## Membership Inference Attack (R4)

```
 eps  sensitivity  n_with  n_without  mia_auc  mia_accuracy  theoretical_max_acc
0.01          1.0      50         49   0.4994        0.4956               0.5025
0.05          1.0      50         49   0.5095        0.5060               0.5125
0.10          1.0      50         49   0.5220        0.5169               0.5250
0.50          1.0      50         49   0.6191        0.6103               0.6225
1.00          1.0      50         49   0.7236        0.6957               0.7311
2.00          1.0      50         49   0.8656        0.8159               0.8808
5.00          1.0      50         49   0.9883        0.9590               0.9933
0.01          1.0     500        499   0.4994        0.4956               0.5025
0.05          1.0     500        499   0.5095        0.5060               0.5125
0.10          1.0     500        499   0.5220        0.5169               0.5250
0.50          1.0     500        499   0.6191        0.6103               0.6225
1.00          1.0     500        499   0.7236        0.6957               0.7311
2.00          1.0     500        499   0.8656        0.8159               0.8808
5.00          1.0     500        499   0.9883        0.9590               0.9933
0.01          1.0    5000       4999   0.4994        0.4956               0.5025
0.05          1.0    5000       4999   0.5095        0.5060               0.5125
0.10          1.0    5000       4999   0.5220        0.5169               0.5250
0.50          1.0    5000       4999   0.6191        0.6103               0.6225
1.00          1.0    5000       4999   0.7236        0.6957               0.7311
2.00          1.0    5000       4999   0.8656        0.8159               0.8808
5.00          1.0    5000       4999   0.9883        0.9590               0.9933
```

## Reconstruction Attack (R4)

```
 eps_per_query  mean_abs_error  p95_abs_error  max_abs_error
          0.01         148.176        404.364       1091.094
          0.05          29.635         80.873        218.219
          0.10          14.818         40.436        109.109
          0.50           2.964          8.087         21.822
          1.00           1.482          4.044         10.911
          2.00           0.741          2.022          5.455
          5.00           0.296          0.809          2.182
```

## Temporal Validation (R3)

```
                             emp    pred  expired  invalidated
experiment   tau  lambda                                      
lambda_sweep 1000 0.00     7.000   6.995    0.000        0.000
                  0.05    15.500  17.487    8.500        9.000
                  0.10    18.800  27.979   11.800       12.800
                  0.20    31.700  48.964   24.700       26.400
tau_sweep    10   0.00    38.300  69.949   31.300        0.000
             25   0.00    22.500  27.979   15.500        0.000
             50   0.00    13.700  13.990    6.700        0.000
             100  0.00     7.000   6.995    0.000        0.000
             1000 0.00     7.000   6.995    0.000        0.000

(Update RNG is now seeded per-trial: each trial draws an independent update
stream, so these means are over genuinely independent realizations. The model
over-predicts the realized budget in every cell with temporal dynamics; the
static cells reduce to the ordinary E[u_k] forecast. It is a conservative
planning estimate, not a proven upper bound.)
```

## Extended Alpha Sweep (alpha up to 10, k=200)

```
        pred    emp    std
alpha                     
0.0    7.000  7.000  0.000
0.5    7.000  7.000  0.000
1.0    7.000  7.000  0.000
1.5    6.996  7.000  0.000
2.0    6.905  6.933  0.254
3.0    5.594  5.467  0.730
5.0    2.814  2.900  0.803
10.0   1.181  1.167  0.379
```

## Epsilon Sweep Validation (eps in {0.1, 0.5, 1.0, 2.0})

```
            pred    emp  err_pct
eps_q k                         
0.1   25   6.317  6.300   10.360
      50   6.880  6.800    3.956
      100  6.995  7.000    0.073
      250  7.000  7.000    0.000
0.5   25   6.317  6.200   13.385
      50   6.880  6.900    2.852
      100  6.995  6.967    0.545
      250  7.000  7.000    0.000
1.0   25   6.317  6.233    9.974
      50   6.880  6.800    3.956
      100  6.995  7.000    0.073
      250  7.000  7.000    0.000
2.0   25   6.317  6.200   11.223
      50   6.880  6.867    3.220
      100  6.995  7.000    0.073
      250  7.000  7.000    0.000
```

## Large-k Sweep (k up to 500)

```
      pred    emp    std
k                       
25   6.317  6.400  0.632
50   6.880  6.933  0.258
100  6.995  7.000  0.000
250  7.000  7.000  0.000
500  7.000  7.000  0.000
```

## Semantic L2 Cache Validation

```
                      eps  exact_hits  sem_hits       err  lat_ms
alpha mode                                                       
0.0   naive_dp     10.000       0.000     0.000  5154.789   1.808
      semantic_dp   1.000       7.300    41.700  5190.079  25.712
      workload_dp   7.000      43.000     0.000     0.950   1.850
0.5   naive_dp     10.000       0.000     0.000  6373.250   1.518
      semantic_dp   1.000       8.300    40.700  4938.959  11.405
      workload_dp   6.967      43.033     0.000     1.019   1.644
1.0   naive_dp     10.000       0.000     0.000  7524.382   1.461
      semantic_dp   1.000      12.233    36.767  3930.507   9.703
      workload_dp   6.900      43.100     0.000     0.774   1.621
2.0   naive_dp     10.000       0.000     0.000  8967.058   1.502
      semantic_dp   1.000      23.167    25.833  1821.941   7.125
      workload_dp   5.600      44.400     0.000     0.971   1.574
```

## Full Benchmark Campaign (6 workloads x 4 eps x 3 modes x 30 trials)

Total cells: 2160 trials

Budget consumption by (workload, mode) at eps_q=1.0:

```
                                budget  cache   err  n_ans
workload           mode                                   
W1_repetitive      naive_dp      100.0    0.0  0.94  100.0
                   temporal_dp     1.0   99.0  0.87  100.0
                   workload_dp     1.0   99.0  1.13  100.0
W2_tpch_priority   naive_dp      100.0    0.0  0.97  100.0
                   temporal_dp     5.0   95.0  0.91  100.0
                   workload_dp     5.0   95.0  0.85  100.0
W2_tpch_returnflag naive_dp      100.0    0.0  0.94  100.0
                   temporal_dp     3.0   97.0  0.88  100.0
                   workload_dp     3.0   97.0  0.86  100.0
W2_zipf            naive_dp      100.0    0.0  0.93  100.0
                   temporal_dp     7.0   93.0  0.92  100.0
                   workload_dp     7.0   93.0  0.95  100.0
W3_uniform         naive_dp      100.0    0.0  0.99  100.0
                   temporal_dp     7.0   93.0  0.85  100.0
                   workload_dp     7.0   93.0  0.94  100.0
W4_drilldown       naive_dp      100.0    0.0  0.95  100.0
                   temporal_dp   100.0    0.0  0.96  100.0
                   workload_dp   100.0    0.0  0.96  100.0
```

## Cross-Scale Validation (SF=1 vs SF=10)

```
                                      pred    emp
scale_factor template_set     alpha              
SF1          priority (m=5)   0.0    5.000  5.000
                              0.5    5.000  5.000
                              1.0    5.000  5.000
                              2.0    4.924  4.900
             returnflag (m=3) 0.0    3.000  3.000
                              0.5    3.000  3.000
                              1.0    3.000  3.000
                              2.0    3.000  3.000
SF10         priority (m=5)   0.0    5.000  5.000
                              0.5    5.000  5.000
                              1.0    5.000  5.000
                              2.0    4.924  4.933
             returnflag (m=3) 0.0    3.000  3.000
                              0.5    3.000  3.000
                              1.0    3.000  3.000
                              2.0    3.000  3.000
```
