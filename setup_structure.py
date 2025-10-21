import os


base_dir = r"D:\Portfolio Optimzation project"

# folder structure
folders = [
    "raw_data",
    "processed_data",
    "scripts",
    "notebooks",
    "configs",
    "logs",
    "results",
    "reports"
]

# create folders
for folder in folders:
    path = os.path.join(base_dir, folder)
    os.makedirs(path, exist_ok=True)
    print(f"Created folder: {path}")

# create script files
script_files = [
    "__init__.py",
    "data_loader.py",
    "data_cleaner.py",
    "factor_builder.py",
    "ml_model.py",
    "portfolio_optimizer.py",
    "backtester.py",
    "stress_tester.py",
    "visualizer.py",
    "utils.py"
]

for f in script_files:
    open(os.path.join(base_dir, "scripts", f), 'a').close()

# create notebook placeholders
notebooks = [
    "01_data_preprocessing.ipynb",
    "02_factor_construction.ipynb",
    "03_ml_prediction.ipynb",
    "04_portfolio_allocation.ipynb",
    "05_backtest_results.ipynb",
    "06_stress_testing.ipynb",
    "07_final_summary.ipynb"
]

for n in notebooks:
    open(os.path.join(base_dir, "notebooks", n), 'a').close()

# create config placeholders
configs = [
    "data_config.yaml",
    "model_config.yaml",
    "backtest_config.yaml",
    "stress_config.yaml"
]

for c in configs:
    open(os.path.join(base_dir, "configs", c), 'a').close()

# create main.py and readme
open(os.path.join(base_dir, "main.py"), 'a').close()
open(os.path.join(base_dir, "README.md"), 'a').close()

print("\n Project structure successfully created!")
