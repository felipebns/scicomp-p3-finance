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

The project is organized in **three distinct phases**:

### Phase 1: Walk-Forward Model Selection
The system uses Walk-Forward Validation to robustly select the best model without testing data leakage:
- Trains each model on expanding windows of historical data
- Tests on the subsequent period
- Calculates **per-fold Sharpe ratio** to measure consistency
- Uses robust scoring: `mean_sharpe - (0.5 × std_sharpe)` to favor stable models
- Selects the algorithm with highest robust score

### Phase 2: Pure ML Evaluation
On the complete training set, the selected model is retrained and evaluated using **classification metrics only**:
- **Accuracy** - overall correctness
- **Precision** - false positive rate
- **Recall** - signal capture rate  
- **F1 Score** - balance between precision and recall
- **AUC** - ability to separate classes
- **Information Coefficient (IC)** - correlation between predicted probability and actual returns
- **Spearman Correlation** - rank correlation (robust to outliers)

These metrics measure **pure prediction quality**, independent of any trading strategy.

### Phase 3: Backtesting with Multiple Probability Thresholds
The model produces probability predictions (0-1) that are converted into trading signals using **multiple thresholds**:
- Tests thresholds: 0.50, 0.55, 0.60, 0.65, 0.70
- Each threshold generates positions: `buy = (probability > threshold)`
- Each strategy is backtested independently
- Calculates financial metrics: total return, annualized return, Sharpe ratio, max drawdown, hit rate
- Compares against 4 benchmarks: Buy & Hold, Fixed Income, Mean Reversion, Random Walk

---

## Detailed workflow

### 1. Data collection
Downloads historical data for the selected stock with columns like:
- `open`, `high`, `low`, `close`, `adj_close`, `volume`

### 2. Feature engineering  
Creates derived variables from raw data:
- daily returns, lagged returns, moving averages
- RSI, MACD, Bollinger Bands
- momentum indicators, other technical features
- All computed using past data only (no lookahead bias)

### 3. Walk-Forward Model Selection (Phase 1)
- Growing window approach: train on `[0:t]`, test on `[t:t+w]`
- Each fold uses `copy.deepcopy(algorithm)` to ensure isolation
- Per-fold Sharpe ratios calculated
- Best model selected by robust score

### 4. Retraining on complete training set (Phase 2)
- Selected model trained on all historical data
- Evaluated with **classification metrics only**
- No financial/backtest metrics at this stage

### 5. Probability prediction and strategy generation (Phase 3)
- Model generates probability predictions `p(1)` for test set
- Multiple trading strategies created by varying threshold
- Each threshold: `signal = (probability > threshold)`

### 6. Backtesting with multiple strategies
- Each threshold-based strategy is simulated
- Benchmarks also run for comparison
- Financial metrics calculated for each

### 7. Visualization and reporting
Charts and JSON summary with detailed results

---

## Benchmarks used

Comparing strategies with simple benchmarks is fundamental. Without context, there's no way to know if the model is truly useful.

### 1. Buy & Hold
Buys the asset at the beginning of the period and holds it until the end.

This represents a passive approach common in the market. Any active strategy must justify why it should be used instead.

### 2. Fixed Income / Cash
Constant risk-free rate (e.g., 5% annual).

This benchmark shows what happens if capital earns interest without market exposure. Useful for comparison with low-volatility allocation decisions.

### 3. Mean Reversion
Simple strategy based on regression to the mean.

Example rule: buy when price < moving average, sell when price > moving average. Serves as a heuristic reference based on intuitive logic.

### 4. Random Walk / Monte Carlo Simulation
Random strategy used as a sanity check.

Tests 50 Monte Carlo runs of random signals to establish a baseline. If the model doesn't beat random walk, it hasn't learned real signal.

### Strategy Selection
Not all benchmarks are equally important:
- **Buy & Hold** is the primary comparison point
- **Fixed Income** is the conservative reference
- **Random Walk** is the control/sanity check
- **Mean Reversion** is an additional simple heuristic

---

## Pure ML Metrics vs Financial Metrics

This project separates **two evaluation layers** that must remain independent:

### Pure ML Metrics (Phase 2)
Measured on the training set after model selection. These metrics evaluate **prediction quality alone**:
- **Accuracy**: Proportion of correct predictions
- **Precision**: Of predicted buys, how many were right?
- **Recall**: Of actual buy opportunities, how many were captured?
- **F1**: Harmonic mean (balance between precision and recall)
- **AUC**: Ability to separate classes regardless of threshold
- **IC (Information Coefficient)**: Correlation between predicted probability and actual returns
- **Spearman Correlation**: Rank correlation (robust to outliers)

**Important**: These metrics are independent of any trading strategy or threshold choice.

### Financial Metrics (Phase 3)
Measured during backtesting on the test set. These depend on threshold choice and strategy parameters:
- **Total Return**: Cumulative gain/loss over period
- **Annualized Return**: Total return annualized
- **Sharpe Ratio**: Risk-adjusted return (return per unit volatility)
- **Max Drawdown**: Largest capital loss from peak
- **Active Hit Rate**: Percentage of active decisions that were profitable

**Why separate them?**
- Same model with different thresholds → different financial metrics but same ML metrics
- Ensures flexibility for testing multiple strategies
- Prevents overfitting financial metrics during model training
- More methodologically correct and reproducible

---

## Information Coefficient (IC)

A key ML metric for trading models.

**Definition**: Correlation between model's predicted probability and actual returns.

**Interpretation**:
- IC = 0: Model predictions have no correlation with returns (no signal)
- IC > 0: Positive correlation (model's confidence aligns with actual returns)
- IC < 0: Negative correlation (model's predictions are inverse to actual returns)

**Why IC matters**:
- Measures prediction quality independent of threshold
- Robust even if classification accuracy is modest
- More relevant for trading than traditional classification metrics
- Can identify models with signal even if they predict minority class poorly

---

## Classification metrics

Since the problem is formulated as classification, the project also measures predictive metrics of the model.

### Accuracy

Overall proportion of correct predictions.

It's simple and intuitive, but can be misleading if the class is imbalanced.

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

In addition to classification metrics, the project evaluates strategy performance in financial terms **during backtesting only**.

### Total Return

Cumulative return at the end of the period.

Shows how much capital grew or fell in total.

### Annualized Return

Annualized return.

Transforms the observed return over the period into an equivalent annual rate.

### Sharpe Ratio

Measures risk-adjusted return.

Formula: `(return - risk_free_rate) / volatility`

Higher Sharpe indicates better return per unit of risk. **Important**: A very high Sharpe can indicate methodological error if there's information leakage.

### Max Drawdown

Largest capital decline from a previous peak.

Very important risk metric showing worst-case drawdown during the period.

## Walk-Forward Validation

Walk-Forward Validation is used in **Phase 1** to robustly select the best model without data leakage.

### How it works:
1. Train on window `[0:t]`, test on `[t:t+w]`
2. Calculate metrics for this fold
3. Slide window forward and repeat
4. Average metrics across all folds

### Why it matters:
- Markets change over time; models good in one period may fail in another
- Growing windows ensure no training-test overlap (no lookahead bias)
- Per-fold metrics reveal whether a model is robust or lucky
- Much more reliable than single fixed train/test split

### Implementation details:
- Each fold uses `copy.deepcopy(algorithm)` for complete isolation
- Per-fold Sharpe ratios calculated to measure consistency
- Robust scoring: `score = mean_sharpe - (0.5 × std_sharpe)`
  - Rewards models with high average performance
  - Penalizes models with high variance across folds
  - Selects the most stable, generalizable algorithm

## Probability Thresholds and Strategy Optimization

In Phase 3, instead of using a single binary prediction (buy/don't buy), the system tests **multiple probability thresholds**:

- **Threshold 0.50**: Buy when probability > 50% (most aggressive)
- **Threshold 0.55**: Buy when probability > 55%
- **Threshold 0.60**: Buy when probability > 60%
- **Threshold 0.65**: Buy when probability > 65%
- **Threshold 0.70**: Buy when probability > 70% (most conservative)

### Why multiple thresholds?
- Different thresholds produce different risk/return profiles
- Higher thresholds → fewer trades, less frequent signals, potentially more selective
- Lower thresholds → more trades, more frequent signals, higher activity
- Allows identification of optimal threshold without retraining model

### How to interpret results:
- Compare Sharpe ratios across thresholds to find optimal entry selectivity
- Check if higher thresholds reduce max drawdown (more selective = lower risk)
- Identify which threshold has highest hit rate
- Balance between return and risk across different probability levels

---

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

The three-phase pipeline execution:

**Phase 1**: Walk-Forward Validation selects best model by robust score
**Phase 2**: Selected model retrained on complete training set; pure ML metrics calculated  
**Phase 3**: Probability thresholds tested; financial metrics evaluated; results visualized

Example system flow:
1. Download historical SPY data (2006-2026)
2. Split into Train (80%) and Test (20%) without shuffling
3. Phase 1: WFV tests Logistic Regression, SVC, Random Forest, Ensemble
4. Phase 2: Best model evaluated with accuracy, precision, recall, F1, AUC, IC, Spearman
5. Phase 3: Probabilities converted to signals using 5 thresholds; backtesting with benchmarks
6. Visualizations and JSON summary saved to `output/`

---

## Interpreting Results

### Phase 1: Model Selection
- Check which model won and its robust score
- Inspect fold-level Sharpe ratios; high variance suggests regime-dependent model
- Compare mean Sharpe across models

### Phase 2: Pure ML Evaluation  
- **Accuracy 50-55%**: Models struggle with signal (market noise dominant)
- **AUC 0.50-0.55**: Slightly better than random but weak
- **IC near zero**: Predictions have no correlation with actual returns
- **IC > 0.01**: Good sign; model is capturing real signal
- **Spearman ≥ 0.02**: Rank correlation suggests predictive power

### Phase 3: Backtesting & Threshold Optimization
- **Compare thresholds**: Identify sweet spot between selectivity and opportunity capture
- **Threshold 0.70 with high Sharpe**: Model is selective; only strong signals trigger trades
- **Threshold 0.50 with high returns**: Aggressively trading even weak signals
- **Max Drawdown < 5%**: Strategy has strong risk control
- **Max Drawdown > 20%**: Strategy endures significant drawdowns despite ML predictions

### Beating Benchmarks
- **Better than Buy & Hold**: Strategy adapts to market changes
- **Better than Fixed Income**: Strategy generates real returns above risk-free rate
- **Worse than Buy & Hold**: Market-timing attempted by model is unsuccessful
- **Worse than Random Walk**: No signal is being captured; fundamental model failure

### When to Investigate
- **High IC but low financial returns**: Features are correlated with returns but allocation is poor
- **Low IC but positive returns**: Possible luck or special market condition (not robust)
- **Classification metrics good but Sharpe low**: Model predicts correctly but volatility is high

---

## What Can Still Change

Some project details can be adjusted over time:

* **Model selection**: Test different algorithms or ensemble methods
* **Feature engineering**: Add/remove technical indicators or lagged features
* **Thresholds**: Adjust probability threshold range or granularity
* **Walk-forward parameters**: Change window size, overlap, or number of folds
* **Position sizing**: Use confidence-weighted positions instead of fixed sizes
* **Rebalancing**: Add periodic rebalancing rules
* **Benchmarks**: Add sector-specific or risk-parity benchmarks
* **Risk-free rate**: Adjust annual rate assumption for Sharpe calculations
* **Transaction costs**: Model realistic spread and commission fees

## Generated outputs

The project saves the following in the `output/` directory:

**Charts**:
- `backtest_comparison.png` - All strategies' equity curves + metrics bar chart
- `threshold_comparison.png` - Focus on ML threshold strategies only
- `drawdown_analysis.png` - Drawdown analysis for all strategies

**Data**:
- `backtest_summary.json` - Detailed metrics for each strategy
- `classification_metrics.json` - Phase 2 pure ML metrics

---
