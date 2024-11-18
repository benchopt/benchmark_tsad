# Benchmark repository for Time Series Anomaly Detection (TSAD) algorithms

![Build Status](https://github.com/Jad-yehya/benchmark_tsad/workflows/Tests/badge.svg)

## About Benchopt

Benchopt is a package to simplify and make more transparent and reproducible the comparisons of optimization algorithms.


## Objective

This benchmark evaluates and compares Time Series Anomaly Detection (TSAD) algorithms. The goal is to:

- Provide a standardized framework for comparing different TSAD approaches
- Enable reproducible evaluation of anomaly detection performance

## Install

This benchmark can be run using the following command:

```bash
$ pip install -U benchopt
$ git clone https://github.com/Jad-yehya/benchmark_tsad
$ benchopt run benchmark_tsad
```
Options for running the benchmark can be passed as command line arguments.
For example, to run a specific solver and a specific dataset, use the following command:

```bash
$ benchopt run benchmark_tsad -s IsolationForest -d SMAP
```

Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

## Available Solvers

- AR (Autoregressive Linear Model)
- ABOD (Angle-Based Outlier Detection)
- CBLOF (Cluster-Based Local Outlier Factor)
- DIF (Deep Isolation Forest)
- Isolation Forest
- LOF (Local Outlier Factor)
- LSTM (Long Short-Term Memory)
- OCSVM (One-Class SVM)
- VAE (Variational Autoencoder)
- Transformer

## Datasets

- Soil Moisture Active Passive (SMAP)
- Mars Science Laboratory (MSL)
- Pooled Server Metric (PSM)
- Secure Water Treatment (SWaT)
- Water Distribution (WADI.A2_19)
- Simulated dataset

The SMAP, MSL and PSM datasets are automatically fetched when running the benchmark. 
The simulated dataset is generated at running time. 
However, the automatic use of the SWaT and WADI datasets is not possible. In order to use them, you must request access to the owners at the following link :
https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/. 


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
