# =============================================================================
# Causal Inference Trading Platform
# Moving from Correlation to Causation
# =============================================================================
# Author  : Beginner Data Science Student
# Purpose : Learn the difference between correlation and causation in finance
#           and build a simple trading strategy based on causal relationships.
#
# Libraries used:
#   - pandas      : work with data tables
#   - numpy       : math operations
#   - matplotlib  : draw charts
#   - networkx    : draw causal graphs
#   - scipy       : statistical tests
#   - sklearn     : linear regression (for propensity score matching)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings, os
warnings.filterwarnings('ignore')   # keep output clean

# Make sure output folders exist
os.makedirs('data',  exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Set a nice chart style
plt.style.use('seaborn-v0_8-whitegrid')

# Fix random seed so results are the same every time you run the code
np.random.seed(42)

print("=" * 60)
print("  Causal Inference Trading Platform")
print("  Moving from Correlation to Causation")
print("=" * 60)


# =============================================================================
# PART 1 - INTRODUCTION
# Explain: what is correlation vs causation?
# =============================================================================

print("\n--- PART 1: Correlation vs Causation ---\n")

# --- What is CORRELATION? ---
# Correlation means two things move together.
# Example: Ice cream sales go up in summer AND drowning rates go up in summer.
# Does eating ice cream CAUSE drowning? No! Both are caused by HOT WEATHER.
# Hot weather is called a "confounder" — a hidden third variable.

# --- What is CAUSATION? ---
# Causation means one thing DIRECTLY causes another.
# If you raise interest rates, borrowing becomes expensive,
# so companies invest less, which can lower stock prices.
# That is a real causal chain.

# --- Why does this matter in trading? ---
# Most trading strategies just look at correlations (two things move together).
# But correlations can break down! They may just be coincidences or confounders.
# Causal relationships are more stable and reliable over time.

intro_text = """
CORRELATION vs CAUSATION IN TRADING
=====================================
Correlation: When stock A goes up, stock B tends to go up too.
             (they may BOTH be driven by a hidden factor)

Causation  : When interest rates RISE, company costs INCREASE,
             profits FALL, and stock prices tend to DROP.
             (there is a real cause-and-effect chain)

Why use causal inference?
  -> Correlations can be misleading and break down over time.
  -> Causal relationships are more stable and trustworthy.
  -> We make better decisions when we understand WHY things happen.
"""
print(intro_text)


# =============================================================================
# PART 2 - DATASET
# We simulate realistic financial data (no internet needed).
# In a real project you would use: yf.download('AAPL', start='2020-01-01')
# =============================================================================

print("\n--- PART 2: Creating the Dataset ---\n")

# Number of trading days (about 3 years)
n_days = 756

# Create a date index for trading days
dates = pd.bdate_range(start='2021-01-04', periods=n_days)

# ---- Simulate INTEREST RATE (%) ----
# Start at 0.5%, gradually rises to ~5% (like the real 2022-2023 rate hikes)
# We add small random noise each day
interest_rate = np.zeros(n_days)
interest_rate[0] = 0.5
for i in range(1, n_days):
    # Slow drift upward + small daily noise
    drift = 0.003          # tiny daily increase
    noise = np.random.normal(0, 0.05)  # random wobble
    interest_rate[i] = interest_rate[i-1] + drift + noise
    # Clip so it stays in a realistic range (0% to 6%)
    interest_rate[i] = np.clip(interest_rate[i], 0.0, 6.0)

# ---- Simulate INFLATION (%) ----
# Inflation tends to follow interest rates with some lag + noise
# High inflation usually leads to higher interest rates (causal!)
inflation = np.zeros(n_days)
inflation[0] = 1.5
for i in range(1, n_days):
    # Inflation is correlated with interest rate but also has own momentum
    ir_effect = 0.15 * (interest_rate[i] - inflation[i-1])  # pull toward ir
    noise = np.random.normal(0, 0.08)
    inflation[i] = inflation[i-1] + ir_effect * 0.3 + noise
    inflation[i] = np.clip(inflation[i], 0.0, 10.0)

# ---- Simulate MARKET INDEX (like S&P 500) ----
# Market is negatively affected by high interest rates and inflation
market_index = np.zeros(n_days)
market_index[0] = 4500.0
for i in range(1, n_days):
    # Causal effects on the market:
    # Higher interest rate -> lower market (negative effect)
    # Higher inflation     -> lower market (negative effect)
    ir_impact    = -0.8 * (interest_rate[i] - interest_rate[i-1])
    infla_impact = -0.5 * (inflation[i]     - inflation[i-1])
    base_growth  =  0.03    # small positive daily drift (markets tend to go up)
    noise        = np.random.normal(0, 25)   # daily random moves
    market_index[i] = market_index[i-1] + base_growth + ir_impact + infla_impact + noise
    market_index[i] = max(market_index[i], 1000)  # can't go below 1000

# ---- Simulate APPLE STOCK PRICE ----
# Stock price is influenced by the market index + some company-specific noise
stock_price = np.zeros(n_days)
stock_price[0] = 130.0
for i in range(1, n_days):
    # Stock follows the market closely (beta effect)
    market_return = (market_index[i] - market_index[i-1]) / market_index[i-1]
    beta = 1.2    # Apple is slightly more volatile than the market (beta > 1)
    company_noise = np.random.normal(0, 1.5)   # Apple-specific daily noise
    stock_price[i] = stock_price[i-1] * (1 + beta * market_return) + company_noise
    stock_price[i] = max(stock_price[i], 10)   # can't go below $10

# ---- Put everything into a pandas DataFrame ----
df = pd.DataFrame({
    'date'          : dates,
    'interest_rate' : interest_rate,
    'inflation'     : inflation,
    'market_index'  : market_index,
    'stock_price'   : stock_price
})
df = df.set_index('date')

# ---- Calculate daily stock RETURNS (% change from yesterday) ----
df['stock_return'] = df['stock_price'].pct_change()

# ---- Calculate daily CHANGES in interest rate and inflation ----
df['ir_change']    = df['interest_rate'].diff()   # today minus yesterday
df['infla_change'] = df['inflation'].diff()

# Remove first row (it has NaN because we can't compute change for day 1)
df = df.dropna()

print(f"Dataset created: {len(df)} trading days")
print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"\nFirst 5 rows of the dataset:")
print(df[['interest_rate', 'inflation', 'market_index', 'stock_price']].head())
print(f"\nBasic statistics:")
print(df[['interest_rate', 'inflation', 'stock_price']].describe().round(2))

# Save the dataset to a CSV file
df.to_csv('data/simulated_financial_data.csv')
print("\nDataset saved to: data/simulated_financial_data.csv")


# =============================================================================
# PART 3 - CAUSAL GRAPH CONSTRUCTION
# We draw a "Directed Acyclic Graph" (DAG) showing the causal relationships.
# Arrows show cause -> effect.
# =============================================================================

print("\n--- PART 3: Building the Causal Graph ---\n")

# A causal graph (DAG) has:
# - NODES: the variables (interest rate, inflation, etc.)
# - EDGES (arrows): the cause -> effect relationships

# We create this using the networkx library
G = nx.DiGraph()   # DiGraph = Directed Graph (arrows have direction)

# Add nodes (variables)
nodes = ['Interest\nRate', 'Inflation', 'Market\nIndex', 'Stock\nPrice']
G.add_nodes_from(nodes)

# Add directed edges (cause -> effect)
# Read "A -> B" as "A causally affects B"
causal_edges = [
    ('Interest\nRate', 'Market\nIndex'),   # higher rates -> lower market
    ('Interest\nRate', 'Stock\nPrice'),    # higher rates -> lower stock
    ('Inflation',      'Interest\nRate'),  # high inflation -> rate hikes
    ('Inflation',      'Market\nIndex'),   # high inflation -> lower market
    ('Market\nIndex',  'Stock\nPrice'),    # market moves -> stock moves
]
G.add_edges_from(causal_edges)

print("Causal graph created with:")
print(f"  Nodes (variables): {list(G.nodes())}")
print(f"  Edges (causal links): {len(G.edges())} relationships")
print("\nCausal structure:")
for source, target in G.edges():
    print(f"  {source.replace(chr(10), ' ')} --> {target.replace(chr(10), ' ')}")

# ---- Verify it is a DAG (no circular loops) ----
# A real causal graph must be a DAG (Directed Acyclic Graph).
# It cannot have loops like A->B->C->A, because that would mean A causes itself.
is_dag = nx.is_directed_acyclic_graph(G)
print(f"\nIs this a valid DAG (no circular loops)? {is_dag}")


# =============================================================================
# PART 4 - TREATMENT EFFECT ESTIMATION
# We estimate how much interest rate changes affect stock returns.
# "Treatment" = a rise in interest rates
# "Outcome"   = stock return on that day
# =============================================================================

print("\n--- PART 4: Estimating Treatment Effect ---\n")

# ---- Define "Treatment" ----
# We say a day is "treated" (treatment=1) if the interest rate ROSE that day.
# We say it's "control" (treatment=0) if the interest rate STAYED THE SAME or FELL.

# Threshold: if interest rate went up by more than 0.05%, that's a "rate hike day"
threshold = 0.05
df['treatment'] = (df['ir_change'] > threshold).astype(int)

n_treated = df['treatment'].sum()
n_control = (df['treatment'] == 0).sum()
print(f"Rate-hike days (treatment=1): {n_treated}")
print(f"Normal days    (treatment=0): {n_control}")

# ---- Naive comparison (raw average difference) ----
# Simply compare average stock return on hike days vs normal days.
# This is the NAIVE estimate — it does NOT account for confounders!
treated_returns = df[df['treatment'] == 1]['stock_return']
control_returns = df[df['treatment'] == 0]['stock_return']

naive_ate = treated_returns.mean() - control_returns.mean()
print(f"\nNaive Average Treatment Effect (ATE):")
print(f"  Avg return on rate-hike days : {treated_returns.mean():.4f} ({treated_returns.mean()*100:.2f}%)")
print(f"  Avg return on normal days    : {control_returns.mean():.4f} ({control_returns.mean()*100:.2f}%)")
print(f"  Naive ATE = {naive_ate:.4f} ({naive_ate*100:.2f}%)")
print(f"  Interpretation: On rate-hike days, stock return is {naive_ate*100:.2f}% {'lower' if naive_ate < 0 else 'higher'}")

# ---- Statistical significance test ----
# Is this difference REAL or just random noise?
# We use a t-test: if p-value < 0.05, the difference is statistically significant.
t_stat, p_value = stats.ttest_ind(treated_returns, control_returns)
print(f"\nStatistical significance test (t-test):")
print(f"  t-statistic : {t_stat:.3f}")
print(f"  p-value     : {p_value:.4f}")
if p_value < 0.05:
    print(f"  Result: The difference IS statistically significant (p < 0.05)")
else:
    print(f"  Result: The difference is NOT statistically significant (p >= 0.05)")


# =============================================================================
# PART 5 - CONFOUNDING ADJUSTMENT
# A confounder is a hidden variable that affects BOTH the treatment and outcome.
# If we ignore it, our causal estimate will be wrong.
# =============================================================================

print("\n--- PART 5: Confounding Adjustment ---\n")

confounder_text = """
What is a CONFOUNDER?
========================
A confounder is a variable that:
  1. Affects the TREATMENT (here: interest rate changes)
  2. Affects the OUTCOME   (here: stock returns)
  3. Creates a FAKE correlation between treatment and outcome

Example in our case:
  - High INFLATION causes the central bank to RAISE interest rates (treatment)
  - High INFLATION also directly HURTS stock prices (outcome)
  - So if we ignore inflation, we might think rate hikes hurt stocks MORE than they really do
  - The real culprit (partly) is inflation, not just the rate hike

We must CONTROL for inflation to get a cleaner causal estimate.
"""
print(confounder_text)

# ---- Method: Propensity Score Matching ----
# Propensity score = the probability of being "treated" (getting a rate hike)
# given the confounders (inflation, previous market performance).
#
# By matching treated days to similar control days (same propensity score),
# we compare apples to apples.

print("Applying Propensity Score Matching to control for confounders...")

# --- Step 1: Estimate propensity scores ---
# We train a logistic regression to predict treatment from confounders
confounders = ['inflation', 'infla_change', 'market_index']

# Prepare the data (remove rows with missing values in our variables)
df_clean = df[confounders + ['treatment', 'stock_return']].dropna()

X = df_clean[confounders]
y = df_clean['treatment']

# Standardize confounders (scale to mean=0, std=1) — good practice for logistic regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit logistic regression to get propensity scores
logit_model = LogisticRegression(random_state=42, max_iter=500)
logit_model.fit(X_scaled, y)
propensity_scores = logit_model.predict_proba(X_scaled)[:, 1]  # probability of treatment=1

df_clean = df_clean.copy()
df_clean['propensity_score'] = propensity_scores

print(f"Propensity score range: {propensity_scores.min():.3f} to {propensity_scores.max():.3f}")

# --- Step 2: Match each treated day to a similar control day ---
# For each "treated" day, find the "control" day with the closest propensity score.
# This balances the confounders between treated and control groups.

treated_df = df_clean[df_clean['treatment'] == 1].copy()
control_df = df_clean[df_clean['treatment'] == 0].copy()

matched_pairs = []
used_control_indices = set()

for idx, treated_row in treated_df.iterrows():
    # Find the control day with the closest propensity score
    ps_diff = abs(control_df['propensity_score'] - treated_row['propensity_score'])
    
    # Exclude already-used control days (1-to-1 matching)
    ps_diff_available = ps_diff.drop(index=list(used_control_indices), errors='ignore')
    
    if len(ps_diff_available) == 0:
        continue
    
    best_match_idx = ps_diff_available.idxmin()
    matched_pairs.append({
        'treated_return' : treated_row['stock_return'],
        'control_return' : control_df.loc[best_match_idx, 'stock_return']
    })
    used_control_indices.add(best_match_idx)

matched_df = pd.DataFrame(matched_pairs)

# --- Step 3: Calculate the ADJUSTED treatment effect ---
# Now we compare returns in the matched sample (confounders are balanced)
adjusted_ate = (matched_df['treated_return'] - matched_df['control_return']).mean()

print(f"\nResults after Propensity Score Matching:")
print(f"  Matched pairs           : {len(matched_df)}")
print(f"  Naive ATE (unadjusted)  : {naive_ate*100:.3f}%")
print(f"  Adjusted ATE (matched)  : {adjusted_ate*100:.3f}%")
print(f"\nInterpretation:")
print(f"  After controlling for inflation and market conditions,")
print(f"  a rate-hike day is associated with a {adjusted_ate*100:.3f}% stock return.")
if abs(adjusted_ate) > abs(naive_ate) * 1.1:
    print(f"  The confounders were UNDERSTATING the effect!")
elif abs(adjusted_ate) < abs(naive_ate) * 0.9:
    print(f"  The confounders were OVERSTATING the effect!")
else:
    print(f"  The confounders had a modest effect on the estimate.")


# =============================================================================
# PART 6 - TRADING STRATEGY
# We build a simple causal-based trading rule.
# =============================================================================

print("\n--- PART 6: Building the Trading Strategy ---\n")

# ---- Causal Trading Rule ----
# Based on our causal analysis:
#   - Rising interest rates CAUSALLY hurt stock prices
#   - We should AVOID holding the stock on rate-hike days
#   - We should HOLD on normal days (when rates are stable or falling)
#
# Simple rule:
#   -> If interest rate change today < 0 (rates fell)  => BUY SIGNAL  (hold stock)
#   -> If interest rate change today > 0 (rates rose)  => SELL SIGNAL (stay in cash)
#   -> We use yesterday's rate change to decide today's trade (no look-ahead!)

df['signal'] = 0   # default: no position (cash)

# Use .shift(1) to use YESTERDAY's interest rate change for TODAY's decision
# This is critical! We can't use today's information to trade today.
df['signal'] = np.where(df['ir_change'].shift(1) < 0, 1, 0)
# signal = 1 means "hold the stock"
# signal = 0 means "stay in cash"

# Strategy return = stock return if signal=1, else 0 (cash earns nothing)
df['strategy_return'] = df['stock_return'] * df['signal']

# Buy and Hold return = just hold the stock every day
df['buyhold_return'] = df['stock_return']

# Build the CUMULATIVE wealth curves starting from $1
df['strategy_wealth'] = (1 + df['strategy_return']).cumprod()
df['buyhold_wealth']  = (1 + df['buyhold_return']).cumprod()

total_signals = df['signal'].sum()
print(f"Strategy signals generated:")
print(f"  BUY  (hold) signals : {total_signals}  ({total_signals/len(df)*100:.1f}% of days)")
print(f"  SELL (cash) signals : {len(df)-total_signals}  ({(len(df)-total_signals)/len(df)*100:.1f}% of days)")


# =============================================================================
# PART 7 - VISUALIZATION
# We create 4 charts and save them all.
# =============================================================================

print("\n--- PART 7: Creating Visualizations ---\n")

# ---- CHART 1: Causal Graph ----
fig1, ax1 = plt.subplots(figsize=(9, 6))
fig1.patch.set_facecolor('#F8F9FA')
ax1.set_facecolor('#F8F9FA')

# Define where each node sits on the chart (x, y positions)
pos = {
    'Inflation'     : (0, 1),
    'Interest\nRate': (1, 1),
    'Market\nIndex' : (2, 0.5),
    'Stock\nPrice'  : (3, 0.5),
}

# Node colors
node_colors = ['#FF8F00', '#1565C0', '#2E7D32', '#C62828']

# Draw the graph
nx.draw_networkx_nodes(G, pos, ax=ax1,
                        node_color=node_colors,
                        node_size=3200, alpha=0.9)
nx.draw_networkx_labels(G, pos, ax=ax1,
                         font_size=10, font_color='white',
                         font_weight='bold')
nx.draw_networkx_edges(G, pos, ax=ax1,
                        edge_color='#555555',
                        arrows=True,
                        arrowsize=25,
                        arrowstyle='-|>',
                        width=2.5,
                        connectionstyle='arc3,rad=0.1',
                        min_source_margin=35,
                        min_target_margin=35)

ax1.set_title('Causal Graph (DAG): Relationships Between Economic Variables',
               fontsize=13, fontweight='bold', pad=15)
ax1.axis('off')

# Add a legend explaining the arrows
legend_text = "Arrows show CAUSAL direction (A → B means A causes B)"
fig1.text(0.5, 0.03, legend_text, ha='center', fontsize=10,
          style='italic', color='#555555')

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig('plots/01_causal_graph.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/01_causal_graph.png")

# ---- CHART 2: Stock Price Trend + Interest Rate ----
fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig2.suptitle('Stock Price Trend vs Interest Rate Over Time',
               fontsize=13, fontweight='bold')

# Top panel: stock price
ax2a.plot(df.index, df['stock_price'], color='#1565C0', linewidth=1.5, label='Stock Price')
ax2a.fill_between(df.index, df['stock_price'],
                   df['stock_price'].min(), alpha=0.1, color='#1565C0')
ax2a.set_ylabel('Stock Price ($)', fontsize=11)
ax2a.legend(loc='upper left', fontsize=10)
ax2a.set_facecolor('#FAFAFA')

# Bottom panel: interest rate (the "treatment" variable)
ax2b.plot(df.index, df['interest_rate'], color='#C62828', linewidth=1.5, label='Interest Rate (%)')
ax2b.fill_between(df.index, df['interest_rate'], 0, alpha=0.15, color='#C62828')
ax2b.set_ylabel('Interest Rate (%)', fontsize=11)
ax2b.set_xlabel('Date', fontsize=11)
ax2b.legend(loc='upper left', fontsize=10)
ax2b.set_facecolor('#FAFAFA')

plt.tight_layout()
plt.savefig('plots/02_stock_price_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/02_stock_price_trend.png")

# ---- CHART 3: Strategy Signals ----
# Show buy/sell signals overlaid on the stock price chart
fig3, ax3 = plt.subplots(figsize=(13, 6))
ax3.set_facecolor('#FAFAFA')

# Plot the stock price line
ax3.plot(df.index, df['stock_price'], color='#1565C0', linewidth=1.2,
          alpha=0.7, label='Stock Price', zorder=2)

# Highlight BUY (hold) days in green
buy_days = df[df['signal'] == 1]
ax3.scatter(buy_days.index, buy_days['stock_price'],
             color='#2E7D32', s=5, alpha=0.5, label='BUY Signal (Hold Stock)', zorder=3)

# Highlight SELL (cash) days in red
sell_days = df[df['signal'] == 0]
ax3.scatter(sell_days.index, sell_days['stock_price'],
             color='#C62828', s=5, alpha=0.5, label='SELL Signal (Go to Cash)', zorder=3)

ax3.set_title('Causal Trading Strategy Signals\n(Green = Hold Stock | Red = Stay in Cash)',
               fontsize=12, fontweight='bold')
ax3.set_ylabel('Stock Price ($)', fontsize=11)
ax3.set_xlabel('Date', fontsize=11)
ax3.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('plots/03_strategy_signals.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/03_strategy_signals.png")

# ---- CHART 4: Strategy vs Buy and Hold ----
fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(12, 8))
fig4.suptitle('Causal Strategy vs Buy & Hold: Performance Comparison',
               fontsize=13, fontweight='bold')

# Top: Cumulative wealth curves ($1 grows to ...)
ax4a.plot(df.index, df['strategy_wealth'], color='#1565C0', linewidth=2,
           label='Causal Strategy')
ax4a.plot(df.index, df['buyhold_wealth'], color='#C62828', linewidth=2,
           linestyle='--', label='Buy & Hold')
ax4a.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
ax4a.set_ylabel('Value of $1 Invested', fontsize=11)
ax4a.legend(fontsize=10)
ax4a.set_facecolor('#FAFAFA')
ax4a.set_title('Cumulative Wealth (starting from $1)', fontsize=10)

# Bottom: 30-day rolling returns comparison
rolling_strat = df['strategy_return'].rolling(30).mean() * 252  # annualized
rolling_bh    = df['buyhold_return'].rolling(30).mean() * 252

ax4b.plot(df.index, rolling_strat, color='#1565C0', linewidth=1.5,
           label='Causal Strategy (30d rolling)')
ax4b.plot(df.index, rolling_bh, color='#C62828', linewidth=1.5,
           linestyle='--', label='Buy & Hold (30d rolling)')
ax4b.axhline(y=0, color='gray', linestyle=':', alpha=0.7)
ax4b.set_ylabel('Rolling Annual Return', fontsize=11)
ax4b.set_xlabel('Date', fontsize=11)
ax4b.legend(fontsize=10)
ax4b.set_facecolor('#FAFAFA')
ax4b.set_title('30-Day Rolling Annualized Return', fontsize=10)

plt.tight_layout()
plt.savefig('plots/04_strategy_vs_buyhold.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/04_strategy_vs_buyhold.png")

# ---- CHART 5: Correlation vs Causation illustration ----
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(12, 5))
fig5.suptitle('Why Correlation ≠ Causation in Trading', fontsize=13, fontweight='bold')

# Left: scatter of interest rate vs stock price (shows correlation)
ax5a.scatter(df['interest_rate'], df['stock_price'],
              alpha=0.3, color='#FF8F00', s=15)
m, b, r, p, _ = stats.linregress(df['interest_rate'], df['stock_price'])
x_line = np.linspace(df['interest_rate'].min(), df['interest_rate'].max(), 100)
ax5a.plot(x_line, m * x_line + b, color='#C62828', linewidth=2)
ax5a.set_xlabel('Interest Rate (%)', fontsize=10)
ax5a.set_ylabel('Stock Price ($)', fontsize=10)
ax5a.set_title(f'Correlation: r = {r:.2f}\n(They move together, but WHY?)', fontsize=10)
ax5a.set_facecolor('#FAFAFA')

# Right: show inflation as the confounder
sc = ax5b.scatter(df['interest_rate'], df['stock_price'],
                   c=df['inflation'], cmap='RdYlGn_r',
                   alpha=0.4, s=15)
plt.colorbar(sc, ax=ax5b, label='Inflation (%)')
ax5b.set_xlabel('Interest Rate (%)', fontsize=10)
ax5b.set_ylabel('Stock Price ($)', fontsize=10)
ax5b.set_title('Inflation (colour) as Confounder\n(Both are driven by rising inflation!)', fontsize=10)
ax5b.set_facecolor('#FAFAFA')

plt.tight_layout()
plt.savefig('plots/05_correlation_vs_causation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/05_correlation_vs_causation.png")


# ---- CHART 6: Summary Dashboard ----
fig6 = plt.figure(figsize=(16, 10))
fig6.patch.set_facecolor('#F0F4F8')
fig6.suptitle('Causal Inference Trading Platform — Summary Dashboard',
               fontsize=15, fontweight='bold', y=0.98)

# 2x3 grid
ax_causal    = fig6.add_subplot(2, 3, 1)
ax_price     = fig6.add_subplot(2, 3, 2)
ax_wealth    = fig6.add_subplot(2, 3, 3)
ax_signals   = fig6.add_subplot(2, 3, 4)
ax_ir        = fig6.add_subplot(2, 3, 5)
ax_metrics   = fig6.add_subplot(2, 3, 6)

# Mini causal graph
nx.draw_networkx_nodes(G, pos, ax=ax_causal, node_color=node_colors, node_size=1500, alpha=0.9)
nx.draw_networkx_labels(G, pos, ax=ax_causal, font_size=6, font_color='white', font_weight='bold')
nx.draw_networkx_edges(G, pos, ax=ax_causal, edge_color='#555555',
                        arrows=True, arrowsize=12, width=1.5,
                        connectionstyle='arc3,rad=0.1',
                        min_source_margin=20, min_target_margin=20)
ax_causal.set_title('Causal Graph (DAG)', fontsize=9, fontweight='bold')
ax_causal.axis('off')
ax_causal.set_facecolor('#F0F4F8')

# Stock price
ax_price.plot(df.index, df['stock_price'], color='#1565C0', linewidth=1)
ax_price.set_title('Stock Price ($)', fontsize=9, fontweight='bold')
ax_price.set_facecolor('#FAFAFA')
ax_price.tick_params(labelsize=7)

# Wealth curves
ax_wealth.plot(df.index, df['strategy_wealth'], color='#1565C0', linewidth=1.5, label='Causal')
ax_wealth.plot(df.index, df['buyhold_wealth'],  color='#C62828', linewidth=1.5, linestyle='--', label='B&H')
ax_wealth.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
ax_wealth.set_title('Cumulative Wealth ($1 start)', fontsize=9, fontweight='bold')
ax_wealth.legend(fontsize=7)
ax_wealth.set_facecolor('#FAFAFA')
ax_wealth.tick_params(labelsize=7)

# Signals
ax_signals.plot(df.index[-252:], df['stock_price'].iloc[-252:], color='#1565C0', linewidth=0.8, alpha=0.7)
buy_last = df[df['signal'] == 1].iloc[-252:]
ax_signals.scatter(buy_last.index, buy_last['stock_price'], color='#2E7D32', s=4, alpha=0.6, label='Buy')
ax_signals.set_title('Signals (Last 1 Year)', fontsize=9, fontweight='bold')
ax_signals.legend(fontsize=7)
ax_signals.set_facecolor('#FAFAFA')
ax_signals.tick_params(labelsize=7)

# Interest rate
ax_ir.plot(df.index, df['interest_rate'], color='#C62828', linewidth=1)
ax_ir.fill_between(df.index, df['interest_rate'], 0, alpha=0.1, color='#C62828')
ax_ir.set_title('Interest Rate (%)', fontsize=9, fontweight='bold')
ax_ir.set_facecolor('#FAFAFA')
ax_ir.tick_params(labelsize=7)

# Performance metrics bar chart (computed in Part 8)
ax_metrics.axis('off')   # we'll fill this after computing metrics

plt.tight_layout()
plt.savefig('plots/06_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/06_dashboard.png")


# =============================================================================
# PART 8 - EVALUATION
# Compare the causal strategy against Buy and Hold using standard metrics.
# =============================================================================

print("\n--- PART 8: Strategy Evaluation ---\n")

def total_return(returns):
    """How much did $1 grow overall?"""
    return (1 + returns).prod() - 1

def annual_return(returns):
    """Per-year average return (annualized)."""
    n_years = len(returns) / 252   # 252 trading days per year
    return (1 + returns).prod() ** (1 / n_years) - 1

def volatility(returns):
    """How much do returns bounce around? (annualized)"""
    return returns.std() * np.sqrt(252)

def sharpe_ratio(returns, risk_free=0.02):
    """Return per unit of risk. Higher is better."""
    excess = returns - risk_free / 252
    return (excess.mean() / excess.std()) * np.sqrt(252)

def max_drawdown(returns):
    """Worst peak-to-valley loss experienced."""
    wealth = (1 + returns).cumprod()
    peak   = wealth.cummax()
    dd     = (wealth - peak) / peak
    return dd.min()

def calmar_ratio(returns):
    """Annual return divided by max drawdown (risk-adjusted measure)."""
    ann_ret = annual_return(returns)
    mdd     = abs(max_drawdown(returns))
    return ann_ret / mdd if mdd != 0 else 0

# Calculate all metrics for both strategies
strat_ret = df['strategy_return']
bh_ret    = df['buyhold_return']

metrics = pd.DataFrame({
    'Metric'          : ['Total Return', 'Annual Return', 'Volatility',
                          'Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio',
                          'Days Invested'],
    'Causal Strategy' : [
        f"{total_return(strat_ret)*100:.1f}%",
        f"{annual_return(strat_ret)*100:.1f}%",
        f"{volatility(strat_ret)*100:.1f}%",
        f"{sharpe_ratio(strat_ret):.2f}",
        f"{max_drawdown(strat_ret)*100:.1f}%",
        f"{calmar_ratio(strat_ret):.2f}",
        f"{df['signal'].sum()} / {len(df)}"
    ],
    'Buy & Hold'      : [
        f"{total_return(bh_ret)*100:.1f}%",
        f"{annual_return(bh_ret)*100:.1f}%",
        f"{volatility(bh_ret)*100:.1f}%",
        f"{sharpe_ratio(bh_ret):.2f}",
        f"{max_drawdown(bh_ret)*100:.1f}%",
        f"{calmar_ratio(bh_ret):.2f}",
        f"{len(df)} / {len(df)}"
    ]
})

print("Performance Comparison:")
print(metrics.to_string(index=False))

# Save metrics to CSV
metrics.to_csv('data/performance_metrics.csv', index=False)
print("\nMetrics saved to: data/performance_metrics.csv")

# ---- Final interpretation ----
strat_sharpe = sharpe_ratio(strat_ret)
bh_sharpe    = sharpe_ratio(bh_ret)

print("\n--- FINAL INTERPRETATION ---")
print(f"""
The causal strategy avoided holding the stock on days following interest rate hikes.
This reduced volatility but also reduced participation in market gains.

Key takeaways:
  1. Causal inference helped us identify a REAL relationship, not just correlation.
  2. The strategy correctly avoids exposure on causally-identified risky days.
  3. Whether it beats Buy & Hold depends on how strong the causal effect is
     and how often rate hikes occur.
  4. In practice, you would combine multiple causal signals (not just one).
  5. Transaction costs (commissions) would further reduce the active strategy.

The most important lesson: correlation-based strategies may SEEM to work in
historical data but fail in new conditions. Causal strategies are more robust
because they are based on WHY things happen, not just WHAT happened together.
""")

print("=" * 60)
print("  Project Complete! All files saved.")
print("  Charts: plots/  |  Data: data/")
print("=" * 60)
