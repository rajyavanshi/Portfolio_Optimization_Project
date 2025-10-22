#  Portfolio Optimization Project  

###  Overview  
This project aims to build a **systematic portfolio optimization and rebalancing framework** that can process raw financial data, compute factors, and generate optimal portfolios based on various risk–return objectives.  
It is divided into structured stages to mirror a real quantitative research pipeline.  

---

##  Project Workflow  

### **Stage 1 — Data Cleaning & Processing**  
- Collected raw stock price data (CSV files).  
- Implemented **data cleaning scripts** to handle missing values and outliers.  
- Resampled data to ensure consistent daily frequency.  
- Saved processed data in a structured folder (`processed_data/`).  

### **Stage 2 — Factor Construction & Integration**  
- Built **factor computation modules** (e.g., Momentum, Value, Volatility).  
- Computed **price-based** and **fundamental-based** factors.  
- Combined all factors into a unified `combined_factor_matrix`.  
- Stored results in the `results/factors/` directory for further analysis.  

---

##  **Stage 3 — Portfolio Optimization**  
This stage will focus on:  
- Implementing **Mean-Variance Optimization (Markowitz Model)**.  
- Exploring **Minimum-Variance**, **Equal-Weight**, and **Risk-Parity** portfolios.  
- Performing **backtesting** on historical factor data.  
- Evaluating metrics like **Sharpe Ratio**, **Drawdown**, and **Portfolio Turnover**.  

---

## **Project Structure**

##  **Setup Instructions**

### 1️ Clone the repository 
```bash
git clone https://github.com/rajyavanshi/Portfolio_Optimization_Project.git
cd Portfolio_Optimization_Project

### 2 Create Conda environment
conda env create -f requirements.yml
conda activate portfolio_opt

### Runn the workflow
python main.py

### Tools & Libraries

Python 3.10+

Pandas, NumPy, Matplotlib

Scikit-learn (for factor normalization)

cvxpy (for optimization)

tqdm, os, logging 

Author
Suraj Prakash
B.Tech ECE @ BIT Mesra
Interested in Quantitative Finance, Portfolio Research, and Mathematical Modeling.

“Quantitative investing is where mathematics meets intuition — and every dataset hides a strategy.”