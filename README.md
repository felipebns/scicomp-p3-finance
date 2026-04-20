# ML Portfolio Manager

Tests if machine learning can beat buy-and-hold through daily trading signals on 5 stocks (SPY, AAPL, MSFT, GOOGL, AMZN).

## Quick Start

```bash
python -m venv venv
pip install -r requirements.txt
python main.py
```


---

## The Full Pipeline (5 Phases)

### **Phase 0: Data Preparation**
1. Download 20 years of OHLCV data (2006-2026)
2. Engineer 26 technical features per stock:
   - **Price-based**: `return_close`, `return_adj`, `price_divergence`
   - **Momentum**: RSI, MACD, Rate of Change
   - **Volatility**: ATR, Historical Volatility
   - **Volume**: Volume ratio, OBV signal
   - **Mean Reversion**: Bollinger Bands width/position
   - **Lagged Returns**: Past 3 days of returns
3. Create binary target: `close[today+1] > close[today]?`
4. Split data: 80% train (2006-2020), 20% test (2021-2026)

### **Phase 1: Model Selection via Walk-Forward Validation**

**What is Walk-Forward Validation?**

Rolling window validation that prevents data leakage:

```
Train Window: Fixed 750 trading days
Test Window: Fixed 250 trading days

Fold 1: Train[dates 0-750]      → Test[dates 750-1000]
Fold 2: Train[dates 250-1000]   → Test[dates 1000-1250]
Fold 3: Train[dates 500-1250]   → Test[dates 1250-1500]
...
```

Each model trains on **exactly 750 days** (fair comparison!), then tests on 250 unseen days.

**Selection Metric: Information Coefficient (IC)**
- Measures: "Does model correctly rank stocks?"
- Range: -1.0 (opposite ranking) to +1.0 (perfect ranking)
- Formula: Correlation(model_prob, actual_returns)
- Your model: IC ≈ 0.021 (weak but consistent)

**Best Model Selected**: Algorithm with highest mean IC across all folds

### **Phase 2: Full Training**
Train the best model on **entire** 750-day training set

### **Phase 3: Backtesting**
Run model on test set (2021-2026) with:
- 4 ML strategies × 11 probability thresholds = **44 combinations tested**
- Transaction costs: 0.05% per trade
- Slippage: 0.05% market impact

### **Phase 4: Visualization**
Generate equity curves, model comparison, strategy rankings

---

## Understanding the Strategies

All strategies take the model's probability (0-1) and the threshold, then modify the buy signal.

### **Why Do Strategies Need a Threshold?**

The model outputs a **probability** (0-1):
- 0.50 = "Meh, could go either way"
- 0.75 = "Pretty sure it goes up"
- 0.95 = "Very confident it goes up"

The threshold is a **minimum confidence gate**:
- At threshold 0.50: Buy if probability > 50% (loose)
- At threshold 0.60: Buy if probability > 60% (strict)

**Strategies adjust this threshold based on market conditions.**

---

### **Strategy 1: Threshold Only (Baseline)**
```
if probability > threshold:
    BUY
else:
    SELL/CASH
```
No modifications. Pure ML signal.

---

### **Strategy 2: Mean Reversion**

**Idea**: Prices oscillate around their average. Buy when oversold.

```
sma_50 = 50-day average price
distance = (current_price - sma_50) / sma_50

# If price is BELOW average (oversold):
#   → We're more confident → Lower threshold requirement
# If price is ABOVE average (overbought):
#   → We're less confident → Raise threshold requirement

adjusted_threshold = threshold + (distance × 2)
```

**Example**:
- Price 5% below SMA → adjusted_threshold = 0.50 - 0.10 = **0.40**
  - *Easier to buy* (prices tend to recover)
- Price 5% above SMA → adjusted_threshold = 0.50 + 0.10 = **0.60**
  - *Harder to buy* (prices may pull back)

**Why it worked best (80.3% return)**: The model + mean reversion found optimal sweet spots.

---

### **Strategy 3: Momentum**

**Idea**: Follow the trend. Recent winners tend to continue.

```
momentum = price_change_last_N_days

# If price is RISING (positive momentum):
#   → Trust the model more → Lower threshold
# If price is FALLING (negative momentum):
#   → Require more model confidence → Raise threshold

adjusted_threshold = threshold - (momentum × 5)
```

**Example**:
- Price up 5% recently → adjusted_threshold = 0.50 - 0.25 = **0.25** (easy buy)
- Price down 5% recently → adjusted_threshold = 0.50 + 0.25 = **0.75** (hard buy)

---

### **Strategy 4: Volatility Weighted**

**Idea**: In turbulent markets, take smaller positions. In calm markets, go full.

```
volatility = price_standard_deviation

# Scale position size by volatility
position_size = model_probability × (1 - volatility/max_volatility)

# If volatility is LOW (calm market):
#   → position_size ≈ model_probability (full)
# If volatility is HIGH (risky market):
#   → position_size ≈ 0.5 × model_probability (half)
```

**Example**:
- Calm day, vol = 5% → position = 1.0x (full buy if prob > 0.50)
- Risky day, vol = 20% → position = 0.5x (half buy if prob > 0.50)

Risk management: "Don't make big bets in storms."

---

## Configuration

Edit `config/config.py`:

```python
CONFIG = {
    "tickers": ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN"],
    "start_date": "2006-01-01",
    "wfv_train_window": 750,        # Trading days per fold
    "wfv_test_window": 250,         # Test days per fold
    "probability_thresholds": [0.50, 0.51, ..., 0.60],  # Test all thresholds
    "position_selection": "top_5",  # Select top N stocks per day
    "allocation_mode": "full_deployment",
    "purchase_threshold": 0.50,     # Global minimum confidence
    "transaction_cost": 0.0005,     # 0.05% per trade
    "slippage": 0.0005,             # 0.05% market impact
}
```

---

## Understanding the Data

### **Price Data Used**

**Training Target** (what model learns to predict):
```python
target = (close[tomorrow] > close[today]) ? 1 : 0
```
Uses **CLOSE** (what traders see live on Bloomberg)

**Training Features** (what model sees as input):
```python
return_close = close.pct_change()      # Market price perspective
return_adj = adj_close.pct_change()    # Real investor return (with dividends)
price_divergence = (close - adj_close) / adj_close  # Splits/dividend indicator
```
Model sees **BOTH** and learns which is more predictive

**Backtesting** (how returns are calculated):
```python
daily_return = position × return_adj  # Real investor returns
```
Uses **ADJ_CLOSE** (accounts for splits, dividends)

### **Why This Setup?**
- ✅ Model learns what traders see (CLOSE)
- ✅ Model gets both perspectives (return_close + return_adj)
- ✅ Backtest shows real P&L (return_adj)
- ✅ No overfitting to unrealistic price movements

---

## Results Interpretation

Example from backtest_summary.json:

```
ML mean_reversion 0.50:   Return +80.3%   Sharpe 1.04   Drawdown -16.2%
Buy & Hold (SPY):         Return +76.5%   Sharpe 0.71   Drawdown -33.5%
Fixed Income 5%:          Return +21.8%   Sharpe 0.00   Drawdown 0%
```

**Metrics explained**:
- **Total Return**: Total gain over backtest period (2021-2026)
- **Annualized Return**: % gain per year on average
- **Sharpe Ratio**: Risk-adjusted return (higher = better)
  - \> 1.0: Good
  - \> 2.0: Very good
  - Negative: Strategy lost money on risk-adjusted basis
- **Max Drawdown**: Worst peak-to-trough loss
  - -16%: Lost 16% from peak before recovering
  - -33%: Much worse, more stressful

**Is your model good?**
- ✅ **Beats buy-and-hold**: You won the game
- ✅ **Positive Sharpe > 1.0**: Risk-adjusted winner
- ⚠️ **Beats cash but lower than BnH**: Skill present but not enough
- ❌ **Lower than BnH + negative Sharpe**: Model adds no value

---

## Technical Details

### **Walk-Forward vs Backtesting**

**Walk-Forward (Model Selection)**:
- Trains on sliding 750-day windows
- Tests on 250-day windows
- Prevents looking into future
- Metric: Information Coefficient
- Purpose: Pick best algorithm

**Backtesting (Strategy Testing)**:
- Trains on entire 80% (2006-2020)
- Tests on entire 20% (2021-2026)
- Metric: Sharpe ratio, return, drawdown
- Purpose: Evaluate real-world performance

### **Rolling Window Explanation**

```
Why ROLLING (not GROWING)?
GROWING:     Train[0:750], Train[0:1000], Train[0:1250]
             Problem: Later models have MORE data → unfair advantage

ROLLING:     Train[0:750], Train[250:1000], Train[500:1250]
             Solution: Every model trains on EXACTLY 750 days → fair!
```

---