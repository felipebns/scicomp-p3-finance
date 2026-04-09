# Portfolio Manager with Machine Learning

This project implements a **portfolio manager** based on machine learning for stocks. The core idea is simple:

1. the system collects historical market data;
2. calculates technical and statistical features;
3. uses predictive models to estimate the probability of upside or downside;
4. transforms this prediction into a capital allocation decision;
5. compares the strategy's performance with traditional benchmarks.

The goal is not to predict the exact price of a stock, but to **make better investment decisions than naive strategies**.

---

## Overview

The project was designed to answer the question:

> "Using only historical data, can I create an allocation strategy that beats simple market benchmarks?"

To do this, the system works with financial time series and evaluates the strategy in terms of:

* cumulative return;
* annualized return;
* Sharpe ratio;
* maximum drawdown;
* decision hit rate;
* model classification metrics.

The models, features, and buy/sell rules can be adjusted over time. The project structure was designed to allow gradual evolution without rewriting the entire pipeline.

---

## How the system works

The general project flow is:

### 1. Data collection

The system downloads historical data for the selected stock, typically using columns like:

* `open`
* `high`
* `low`
* `close`
* `adj_close`
* `volume`

### 2. Feature engineering

From raw data, the project creates derived variables, for example:

* daily returns;
* lagged returns;
* moving averages;
* RSI;
* MACD;
* Bollinger Bands;
* momentum indicators;
* other technical features.

These features are calculated only with past data to avoid information leakage from the future.

### 3. Model training

Models receive the features and learn to predict the direction of the next move in the asset.

The exact type of model can vary over time. The architecture was designed to support, for example:

* Logistic Regression;
* Support Vector Classifier;
* Random Forest;
* voting ensembles;

### 4. Signal generation

The model produces a prediction, typically something like:

* `1` = buy / stay long;
* `0` = stay in cash;

### 5. Backtest

The strategy is simulated on out-of-sample data. The backtest calculates how capital would have evolved over time if the strategy had been used for real.

### 6. Comparison with benchmarks

The system compares the model strategy with simple reference strategies.

### 7. Visualization and reporting

At the end, the project generates charts and saves a summary with the main metrics.

---

## Benchmarks used

Comparing a strategy with simple benchmarks is fundamental. Without it, there is no context to know if the model is truly useful.

### 1. Buy & Hold

Buys the asset at the beginning of the period and holds it until the end.

This benchmark is important because it represents a passive approach common in the market. If the active strategy doesn't beat Buy & Hold, it needs to justify why it's still interesting.

### 2. Fixed Income / Cash

Represents money sitting idle earning a constant risk-free rate.

This benchmark shows what would happen if capital were not exposed to the stock market.

It is typically modeled as daily compound interest equivalent to a fixed annual rate.

### 3. Mean Reversion

Simple strategy based on regression to the mean.

A possible rule is to buy when the price is below a moving average and sell when it's above.

It serves as a heuristic benchmark, that is, a simple reference based on an intuitive rule.

### 4. Random Walk / Random Strategy

Random strategy used as a sanity check.

It helps verify whether the model strategy is truly learning something useful or just behaving by chance.

### Important note

Not all benchmarks are equally strong. In general:

* **Buy & Hold** is the main benchmark;
* **Fixed Income** is the conservative reference;
* **Random Walk** serves as a control;
* **Mean Reversion** is an additional simple heuristic.

---

## Classification metrics

Since the problem is formulated as classification, the project also measures predictive metrics of the model.

### Accuracy

Overall proportion of correct predictions.

It's simple and intuitive, but can be misleading in some scenarios, especially if the class is imbalanced.

### Precision

Of all the times the model predicted buy, how many times was it correct?

This metric is useful when false positives are expensive.

### Recall

Of all real buy opportunities, how many did the model capture?

This metric shows how much the model "sees" relevant movements.

### F1 Score

Harmonic mean between precision and recall.

It's useful when you want balance between capturing opportunities and avoiding bad signals.

### AUC

Area under the ROC curve.

Measures the model's ability to separate classes in general, regardless of a fixed threshold.

---

## Financial metrics

In addition to classification metrics, the project evaluates the strategy in financial terms.

### Total Return

Cumulative return at the end of the period.

Shows how much capital grew or fell in total.

### Annualized Return

Annualized return.

Transforms the observed return over the period into an equivalent annual rate.

### Sharpe Ratio

Measures risk-adjusted return.

In simple terms:

> how much return the strategy generates per unit of volatility.

Higher Sharpe typically indicates a more efficient strategy. But a very high Sharpe can be a sign of methodological error if the backtest is contaminated by information leakage.

### Max Drawdown

Largest capital decline from a previous peak.

It's a very important risk metric because it shows the worst suffering of the capital curve during the period.

### Win Rate / Active Hit Rate

Hit rate of decisions or active days, depending on implementation.

This metric indicates how many trades or signals were positive.

---

## Walk-forward validation

The project can use walk-forward validation to evaluate models and choose the best one more robustly.

The idea is:

1. train on a window of the past;
2. test on the next period;
3. advance the window;
4. repeat several times.

This is important because the market changes over time. A model good in one regime can fail in another.

Walk-forward is more reliable than just doing a fixed train/test split.

---

## Project structure

The project organization can follow this logic:

```text
project/
├── output/
├── services/
│   ├── stock.py
│   ├── transform.py
│   ├── pipeline.py
│   ├── backtesting.py
│   ├── algorithms/
│   │   ├── base.py
│   │   ├── logistic_regression.py
│   │   ├── random_forest.py
│   │   ├── svc.py
│   │   └── ensemble.py
├── utils/
│   ├── walk_forward.py
│   └── metrics.py
├── main.py
├── requirements.txt
└── README.md
```

The file names can change, but the general idea is to maintain a clear separation between:

* data collection;
* features;
* models;
* backtest;
* evaluation utilities;
* main execution.

---

## Requirements

You can install everything via `requirements.txt`.

---

## How to set up the environment

### 1. Create a virtual environment

On Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the project

Normally:

```bash
python main.py
```

If your entry point has a different name, adjust the command accordingly.

---

## How to run in practice

The execution flow typically is:

1. choose the ticker;
2. define historical period;
3. download the data;
4. generate features;
5. train the models;
6. evaluate metrics;
7. run backtest;
8. save charts and JSON with results.

Example of conceptual configuration:

* ticker: `AAPL`
* historical window: last 10 to 20 years
* temporal split: train and test without shuffling
* benchmarks: Buy & Hold, Fixed Income, Mean Reversion, Random Walk

---

## Interpreting results

### If the model beats Buy & Hold

That's a good sign, but it's still not sufficient on its own.

### If Sharpe is higher than benchmarks

The strategy may be delivering better return per unit of risk.

### If drawdown is lower

The strategy suffers less in bad periods, which is great for real-world use.

### If the model loses to random walk

It's a strong sign that the system hasn't yet captured real signal.

### If classification is good, but backtest is bad

This may indicate that the allocation strategy needs to be better, even if the classifier is learning something useful.

---

## What can still change

Some project details can be adjusted over time, such as:

* which models will be used;
* which features will be kept;
* what buy/sell rule will be adopted;
* whether the strategy will be long-only or long/cash;
* whether there will be rebalancing;
* whether weights proportional to model confidence will be used.

---

## Generated outputs

The project can save:

* metrics in JSON;
* equity curve charts;
* drawdown charts;
* comparison between models;
* backtest summary by strategy.

---
