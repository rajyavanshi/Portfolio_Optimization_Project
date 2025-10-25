#  Portfolio Optimization Project  

###  Overview  
This project builds a **systematic portfolio optimization and rebalancing framework** capable of processing raw market data, computing alpha factors, and constructing optimized portfolios under various risk–return objectives.  
It mirrors a real **quantitative research pipeline**, covering the full lifecycle — from data cleaning to portfolio simulation and stress testing.

---

##  Project Workflow  

### **Stage 1 — Data Cleaning & Processing**  
- Collected and merged raw stock price data (CSV format).  
- Implemented robust cleaning routines to handle **missing values**, **outliers**, and **non-trading days**.  
- Standardized all tickers to a consistent **daily frequency**.  
- Filtered tickers based on a **40% missing data threshold** to maintain reliability.  
- Saved processed outputs in `processed_data/`.  

---

### **Stage 2 — Factor Construction & Integration**  
- Developed **factor computation modules** for key signals:
  - **Momentum** — medium-term price performance (6-month returns).  
  - **Volatility (Low-Vol)** — risk-based factor using rolling standard deviation.  
  - **Quality** — proxy via 1-year average daily returns (extendable to fundamentals).  
- Standardized all factors using **z-score normalization**.  
- Combined them into a **composite factor matrix** for stock ranking and selection.  
- Stored factor outputs in `results/factors/`.

---

### **Stage 3 — Portfolio Optimization & Backtesting**  
- Implemented classical and modern portfolio constructions:
  - **Mean–Variance Optimization (Markowitz Model)**  
  - **Minimum-Variance & Equal-Weight Portfolios**  
  - **Risk-Parity and Volatility-Targeted Allocations**  
- Integrated **transaction cost modeling** and **turnover-aware weighting**.  
- Conducted **monthly rebalancing** with dynamic capital scaling.  
- Evaluated key performance metrics:
  - **CAGR**, **Volatility**, **Sharpe Ratio**, **Max Drawdown**, **Turnover**  

---

### **Stage 4 — Stress Testing & Scenario Analysis**  
- Designed **synthetic market scenarios** to simulate:
  -  **Bull Markets:** amplified returns, lower vol  
  -  **Bear Crashes:** sudden –30% shocks + vol spikes  
  -  **Sideways Regimes:** choppy, mean-reverting behavior  
- Evaluated resilience under each regime with and without **volatility targeting**.  
- Visualized performance curves and drawdown behavior across stress conditions.

---

##  Project Structure  

```
Portfolio_Optimization_Project/
│
├── data/                        # Raw data files (CSV)
├── processed_data/               # Cleaned and resampled data
├── src/                          # Source code modules
│   ├── config.py                 # Global parameters and constants
│   ├── utils.py                  # Helper functions for data handling
│   └── optimization_models.py    # Core portfolio optimization logic
│
├── notebooks/
│   └── Portfolio_Strategy_Research_Report.ipynb  # Full research notebook
│
├── results/
│   └── factors/                  # Stored factor computation outputs
│
├── reports/
│   └── figures/                  # Generated performance charts
│
├── requirements.yml              # Conda environment file
├── README.md                     # Project documentation
└── main.py                       # Entry point for workflow automation
```

---

##  Setup Instructions  

### 1️. Clone the repository  
```bash
git clone https://github.com/rajyavanshi/Portfolio_Optimization_Project.git
cd Portfolio_Optimization_Project
```

### 2️. Create Conda environment  
```bash
conda env create -f requirements.yml
conda activate portfolio_opt
```

### 3️. Run the workflow  
```bash
python main.py
```

*(You can also open `Portfolio_Strategy_Research_Report.ipynb` inside Jupyter for full research documentation.)*

---

##  Tools & Libraries  

| Category | Libraries Used |
|-----------|----------------|
| **Data Handling** | pandas, numpy, tqdm |
| **Visualization** | matplotlib, seaborn |
| **Optimization** | cvxpy, scikit-learn |
| **Workflow** | os, logging, datetime |

---

##  Key Features  

* Clean, modular code for each stage of quant research.  
* Advanced techniques like **Volatility Targeting**, **Covariance Shrinkage**, and **Turnover Control**.  
* Realistic **stress testing** with multiple market regimes.  
* Reproducible research notebook (`.ipynb`) formatted as a quant report.  

---

##  Example Metrics (Backtest Summary)

| Metric | Value (example) |
|---------|----------------|
| CAGR | 14.5% |
| Annual Volatility | 11.5% |
| Sharpe Ratio | 1.26 |
| Max Drawdown | –13.2% |
| Avg Turnover | 9.4% |

---

##  Author  

**Suraj Prakash**  
B.Tech — Electronics & Communication Engineering, **BIT Mesra**  
 Interested in **Quantitative Finance**, **Portfolio Research**, and **Mathematical Modeling**  

> “Quantitative investing is where mathematics meets intuition — and every dataset hides a strategy.”

 [GitHub: rajyavanshi](https://github.com/rajyavanshi)

---

##  Future Work  
- Integrate **fundamental data** for richer factor design.  
- Implement **machine learning models** for adaptive weighting.  
- Explore **regime-switching models** for dynamic risk control.  
- Expand **multi-asset optimization** beyond equities.

---

 **This project demonstrates full-stack quantitative research capability — from data engineering to backtesting and stress analysis — in a reproducible and professional format.**
