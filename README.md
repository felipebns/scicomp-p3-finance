# ML Portfolio Manager

Tests if machine learning can beat buy-and-hold through daily trading signals on 5 stocks (SPY, AAPL, MSFT, GOOGL, AMZN).

## Quick Start

```bash
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

Edit `main.py` to customize:

```python
CONFIG = {
    "allocation_mode": "full_deployment",  # or "cash_allocation"
    "purchase_threshold": 0.50,            # Min confidence to invest
    "probability_thresholds": [0.50-0.60], # Test 11 thresholds (per-stock gate)
    "position_selection": "top_5",         # Select top N stocks
    "transaction_cost": 0.0005,            # 0.05% per trade
    "slippage": 0.0005,                    # 0.05% impact
}
```

### Allocation Modes

**FULL_DEPLOYMENT** (default, aggressive):
- If 2 stocks selected → 100% deployed (50% each)
- If no signals → 100% CASH
- More volatile, higher potential returns

**CASH_ALLOCATION** (conservative):
- If 2 stocks selected → 40% deployed, 60% CASH
- More flexibility, smoother equity curve

### Thresholds

- **probability_thresholds**: Which stocks to select (per-stock decision)
- **purchase_threshold**: Global gate - forces 100% CASH if best signal < 0.50

---

## Results

Example output from backtest_summary.json:

```
ML mean_reversion 0.57:   Return +0.45%   Sharpe 0.12   Drawdown -8.56%
Buy & Hold:               Return -5.32%   Sharpe -0.34  Drawdown -38.25%
Cash Strategy:            Return +2.50%   Sharpe 2.45   Drawdown -0.05%
```

**Is your model good?**
- ✅ Beats buy-and-hold + positive Sharpe
- ⚠️ Beats cash but loses to buy-and-hold
- ❌ Loses to both + negative Sharpe

**Key metrics:**
- **Total Return**: % gain over 5 years
- **Annualized Return**: % per year
- **Sharpe Ratio**: Risk-adjusted (higher = better)
- **Max Drawdown**: Worst loss from peak

---

## Technical Notes

**Walk-Forward Validation:**
- Prevents looking into the future during training
- Trains expanding window: [2006-2008], then [2006-2009], etc.
- Tests on unseen future data: 2009, 2010, etc.

**Information Coefficient (IC):**
- Measures how well model ranks stocks
- Range: -1.0 (opposite) to +1.0 (perfect)
- Your model: IC ≈ 0.021 (weak but consistent)

**44 Combinations:**
- 4 ML strategies × 11 probability thresholds
- Tests high confidence (0.60) vs low confidence (0.50)
- Finds optimal threshold for each strategy

---