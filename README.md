# ML Portfolio Manager

Tests if machine learning can beat buy-and-hold through daily trading signals on 5 stocks (SPY, AAPL, MSFT, GOOGL, AMZN).

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Results: `output/backtest_summary.json` and `output/ml_strategies_by_threshold.png`

## What It Does (6 Steps)

1. **Download data** (2006-2026, 20 years)
2. **Calculate 17 technical indicators** per stock per day
3. **Train ML models** with walk-forward validation (2006-2020)
4. **Backtest on unseen data** (2021-2026)
5. **Test 44 ML combinations** (4 strategies × 11 thresholds)
6. **Generate results** + visualizations + JSON

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