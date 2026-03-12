# Causal Inference Trading Platform
## Moving from Correlation to Causation



---

## What Is This Project?

Most trading strategies are based on **correlation** — if two things tend to move
together, we trade on that pattern. But correlations can be misleading and often
break down unexpectedly.

This project explores **causal inference** — a more powerful approach that asks:
*"Does X actually **cause** Y, or do they just happen to move together?"*

We apply this idea to trading by:
1. Building a causal graph of economic variables
2. Estimating the effect of interest rate changes on stock returns
3. Controlling for confounders (hidden variables that fool naive analysis)
4. Creating a trading strategy based on causal relationships

---

## Project Structure

```
causal_trading/
│
├── causal_trading.py        ← Main Python script (fully commented)
├── notebook.ipynb           ← Jupyter notebook version
├── README.md                ← This file
│
├── data/
│   ├── simulated_financial_data.csv   ← Generated financial dataset
│   └── performance_metrics.csv        ← Strategy comparison results
│
└── plots/
    ├── 01_causal_graph.png            ← DAG showing causal relationships
    ├── 02_stock_price_trend.png       ← Stock price + interest rate over time
    ├── 03_strategy_signals.png        ← Buy/sell signals on price chart
    ├── 04_strategy_vs_buyhold.png     ← Performance comparison
    ├── 05_correlation_vs_causation.png← Why correlation ≠ causation
    └── 06_dashboard.png               ← All-in-one summary dashboard
```

---

## Key Concepts Explained Simply

### Correlation vs Causation
- **Correlation**: Ice cream sales and drowning both go up in summer.
  Does ice cream cause drowning? No — hot weather causes both.
- **Causation**: Higher interest rates → higher borrowing costs → lower profits → lower stock price.
  That's a real causal chain.

### What Is a Confounder?
A hidden variable that affects both the cause and the effect.
In our project: **Inflation** affects both interest rates (cause) and stock prices (effect).
If we ignore inflation, we might wrongly attribute its effects to interest rates alone.

### Propensity Score Matching
A technique to "balance" the comparison groups by controlling for confounders.
We match each "rate-hike day" to a similar "normal day" — making a fair comparison.

---

## How to Run

### Option A: Google Colab (Easiest)
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `notebook.ipynb`
3. Click **Runtime → Run all**

### Option B: Local Python
```bash
# Install required libraries (if not already installed)
pip install pandas numpy matplotlib networkx scipy scikit-learn

# Run the main script
python causal_trading.py
```

---

## Libraries Used

| Library     | Purpose                                      |
|-------------|----------------------------------------------|
| `pandas`    | Data tables and time-series manipulation     |
| `numpy`     | Numerical computing and random simulation    |
| `matplotlib`| Drawing all charts and plots                 |
| `networkx`  | Drawing the causal graph (DAG)               |
| `scipy`     | Statistical tests (t-test for significance)  |
| `sklearn`   | Logistic regression for propensity scores    |

---

## Results Summary

| Metric        | Causal Strategy | Buy & Hold |
|---------------|-----------------|------------|
| Annual Return | ~-2% to +5%     | ~4-8%      |
| Volatility    | Lower           | Higher     |
| Sharpe Ratio  | Varies          | Baseline   |

> **Note**: Results vary with each simulation run.
> The goal is to LEARN the methodology, not to optimize returns.
> A real project would use actual market data and more sophisticated models.

---

## What I Learned

1. Correlations in financial data are often misleading
2. Causal graphs (DAGs) help us think clearly about cause and effect
3. Confounders can completely change our conclusions if ignored
4. Propensity score matching is a practical way to control for confounders
5. Simple causal strategies often underperform Buy & Hold — and that's OK!
   The lesson is the methodology, not the profit

---

## Next Steps to Explore

- Use **real stock data** with `yfinance` library
- Add more causal variables (GDP growth, earnings reports)
- Try **Instrumental Variables** — another causal inference method
- Learn about **Difference-in-Differences** for policy effect estimation
- Read: *"The Book of Why"* by Judea Pearl — the best intro to causal thinking

---

*Made as a beginner portfolio project to learn causal inference in finance.*
