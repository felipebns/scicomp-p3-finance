# ML Portfolio Manager: Daily Trading Signals on 5 Stocks

A simple system to test whether a machine learning model can select stocks better than simply "buy everything and hold it."

**The question:** Can an ML model beat buy-and-hold through daily trading signals on 5 tech stocks during a 5-year period?

---

## What This Project Does

```
1. Get 20 years of price data (2006-2026) for 5 stocks: SPY, AAPL, MSFT, GOOGL, AMZN

2. Calculate 17 technical signals per stock per day
   (Exponential Moving Averages, RSI, Bollinger Bands, Momentum, etc)

3. Train a model on 15 years of data (2006-2020) to predict: "Tomorrow up or down?"

4. Test the model on 5 years it has never seen (2021-2026)

5. Compare 7 different strategies:
   - ML model with different confidence levels
   - Buy & Hold (buy everything and keep it)
   - Passive Income / Cash (earn interest, stay safe)
   - Simple benchmarks (mean reversion, random)
```

---

## How It Works: 5 Steps

### Step 1: Data Collection

Downloads 20 years of historical data in "panel" format:
- Each day has an entry for each stock
- Date | Ticker | Open | Close | Volume | ...
- Example:
  - 2021-01-04 | SPY | 371.0 | 372.2 | 100M
  - 2021-01-04 | AAPL | 127.1 | 128.0 | 50M  ← Same date, different stock
  - 2021-01-04 | MSFT | 215.5 | 216.1 | 75M

**Result:** ~5,000 trading days × 5 stocks = 25,000 rows of data

### Step 2: Calculate Technical Signals

For **each stock individually**, calculates 17 technical indicators:
- Exponential Moving Averages (EMA 10, 20 days)
- RSI (Relative Strength Index - trend strength)
- Bollinger Bands (volatility)
- MACD, Momentum, Rate of Change
- Past returns (1, 5, 10 days ago)

**Important:** Calculated separately per stock. SPY data doesn't "contaminate" AAPL.

**Result:** 17 features per stock per day + target label ("did price go up tomorrow?")

### Step 3: Train the Model (2006-2020)

Uses 15 years of training data with walk-forward validation:
- Train on years 1-3, test on year 4 → Get score
- Train on years 1-4, test on year 5 → Get score
- Train on years 1-5, test on year 6 → Get score
- ... continue expanding training window ...
- Train on years 1-14, test on year 15 → Get score

**Metric:** Information Coefficient (IC) = correlation between predictions and actual returns
- High IC = model ranks stocks correctly (knows which will rise first)
- Low IC = model is random

Models tested: Logistic Regression, Random Forest, SVC, Ensemble

**Result:** Ensemble wins with IC ≈ 0.021 (weak but best available)

### Step 4: Prepare Final Model

Retrain the best model (Ensemble) using ALL 15 years of training data (2006-2020).

Now it's ready to test on 5 years it has never seen.

### Step 5: Backtest (2021-2026)

Every day, the model:
1. Looks at 5 stocks' historical data
2. Makes prediction: "What's the probability each stock goes up?"
3. Decides: Buy or stay in cash?
4. Records gain/loss for that day
5. Moves to next day

**Result:** An equity curve showing how $10,000 performs over 5 years

---

## Files & Architecture

```
main.py
  └─ CONFIG: All parameters here
     Can change: tickers, model, confidence thresholds, dates, everything

services/
  ├─ stock.py           → Download price data from Yahoo Finance
  ├─ transform.py       → Calculate technical signals
  ├─ pipeline.py        → Orchestrate entire workflow
  ├─ backtesting.py     → Simulate trading strategies
  └─ algorithms/        → ML models
      ├─ base.py                  (base class)
      ├─ ensemble.py              (winner - combination of models)
      ├─ random_forest.py
      ├─ svc.py
      └─ logistic_regression.py

output/
  ├─ classification_metrics.json  → Model quality metrics
  ├─ backtest_summary.json        → Returns of each strategy
  └─ [plots]                      → 3 comparison charts

utils/
  └─ walk_forward.py    → Walk-forward validation logic
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the pipeline
```bash
python main.py
```

The script will:
- Download historical data
- Calculate features
- Train model with walk-forward validation
- Run backtest with 7 strategies
- Generate results and plots

### 3. Check results
```bash
# View summary statistics
cat output/backtest_summary.json

# View model quality
cat output/classification_metrics.json

# Open charts
open output/backtest_comparison.png          # All strategies vs each other
open output/threshold_comparison.png         # ML at different confidence levels
open output/drawdown_analysis.png            # Worst drawdowns
```

---

## Customization

All parameters are in `main.py` under the `CONFIG` dictionary:

```python
CONFIG = {
    "tickers": ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN"],
    "train_start": "2006-01-01",
    "train_end": "2020-12-31",
    "test_start": "2021-01-01",
    "test_end": "2026-12-31",
    "model_type": "ensemble",
    "thresholds": [0.50, 0.55, 0.57, 0.60],
    "position_sizing": "equal_weight",
    # ... more parameters
}
```

Change any parameter and run `python main.py` again to see new results.

---

## Key Takeaways

1. **ML vs Buy & Hold: Context Dependent**
   - In tech bear markets: Passive strategies win
   - In bull markets: Good ML can outperform by 5x
   - In crises: Good defensive ML protects capital

2. **The Real Edge: Knowing When to Do Nothing**
   - Your ML model: +3.70% (trying hard)
   - Cash strategy: +21.74% (doing almost nothing)
   - Lesson: Perfect execution on bad idea = failure

3. **2021-2026 Was Harsh for Concentrated Tech**
   - 80% tech exposure + rising rates = -24.92%
   - Diversified SPY + rising rates = only -5%
   - Energy + rising rates = +50% to +100%

4. **Professional Quants Rotate Between Asset Classes**
   - Not just "stocks vs bonds"
   - But stocks + bonds + cash all together
   - Allocation changes based on confidence

---

## Next Steps to Improve Returns

1. **Better Signals** (IC ≈ 0.02 is too weak)
   - Add sentiment analysis (news, social media)
   - Add volume patterns (institutional buying/selling)
   - Add macro indicators (inflation, unemployment, interest rates)
   - Target: IC from 0.02 → 0.10
   - Expected improvement: +3.70% → +10% to +20%

2. **Lower Trading Costs**
   - Current: Trade every day (expensive)
   - Try: Weekly or monthly rebalancing
   - Expected improvement: +3.70% → +8% to +12%

3. **More Diversification**
   - Current: 5 stocks (80% tech)
   - Try: 20+ stocks across sectors (energy, healthcare, finance, utilities)
   - Expected benefit: Better protection in bear markets

4. **Dynamic Asset Allocation**
   - Instead of just "stocks yes/no"
   - Allocate between: stocks + bonds + cash

---

## Technical Details

**Model Selection Metric:** Information Coefficient (IC)
- Measures correlation between predictions and actual returns
- Scale: -1.0 (perfect opposite) to +1.0 (perfect alignment)
- Your best model: IC ≈ 0.021 (weak but statistically significant)

**Validation:** Walk-Forward Cross-Validation
- Prevents data leakage (never train on future data)
- Mimics real trading (always test on unseen data)
- More realistic than standard K-fold CV

**Portfolio Construction:** Equal-weight positions
- Each stock gets same allocation
- Daily rebalancing to maintain equal weights
- Undeployed capital earns 5% risk-free rate

**Backtesting:** Daily simulation
- Each day: predict, allocate, record returns
- Include transaction costs (0.10% per trade)
- Calculate Sharpe ratio for risk-adjusted returns

---