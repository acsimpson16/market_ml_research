# Market ML Research

# market_ml_research

![CI](https://github.com/acsimpson16/market_ml_research/actions/workflows/ci.yml/badge.svg)

A Python-based machine learning project exploring whether historical market data and engineered financial features can help predict short-term stock returns.

market_ml_research/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .vscode/
├── data/
├── notebooks/
├── results/
├── src/
├── tests/
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt

## Overview

`market_ml_research` is a Python-based machine learning project exploring whether historical market data and engineered financial features can help predict short-term stock returns.

The project uses historical daily data from 25 US-listed companies across five sectors, alongside market benchmarks and volatility indicators. It explores data collection, cleaning, exploratory data analysis, feature engineering and machine learning model evaluation.

The main aim is to build a reproducible end-to-end data science workflow while investigating the challenges of predicting financial market movements.

## Research Question

**Does training machine learning models on stocks within the same sector improve out-of-sample five-day return prediction compared with training on stocks across different sectors?**

The project investigates:

- Whether historical price and volume features contain useful information about future returns.
- Whether market benchmarks provide additional predictive information.
- Whether sector-specific models perform differently from a pooled model trained across multiple sectors.
- How model performance compares with simple baseline predictions.

## Dataset

The project uses historical daily market data from Yahoo Finance, collected using the `yfinance` Python library.

The dataset contains 25 US-listed companies across five sectors:

| Sector | Companies |
|---|---|
| Technology | AAPL, MSFT, NVDA, META, GOOGL |
| Financials | JPM, GS, BAC, MS, BLK |
| Healthcare | JNJ, UNH, ABBV, PFE, MRK |
| Energy | XOM, CVX, COP, SLB, EOG |
| Consumer | AMZN, COST, WMT, HD, MCD |

Additional market indicators include:

- **SPY:** S&P 500 ETF
- **QQQ:** Nasdaq-100 ETF
- **VIX:** CBOE Volatility Index

The historical stock data currently covers approximately ten years of daily observations.

## Methodology

The project follows a structured machine learning workflow.

### 1. Data collection and processing

Historical market data is downloaded and processed using Python and pandas.

The data pipeline includes:

- Downloading historical price and volume data.
- Reshaping and cleaning downloaded data.
- Adding sector classifications.
- Checking for missing values and duplicate observations.
- Saving raw and processed datasets in Parquet format.

### 2. Exploratory data analysis

Exploratory analysis examines stock price movements, returns, volatility and trading volume.

The analysis uses Python, pandas and Plotly to investigate patterns across individual companies and sectors.

### 3. Feature engineering

Technical and market-related features are generated from historical price and volume data.

Current features include:

| Feature | Description |
|---|---|
| Daily_Return | Daily percentage change in closing price |
| Return_5D | Five-day historical return |
| Return_10D | Ten-day historical return |
| Return_20D | Twenty-day historical return |
| Return_60D | Sixty-day historical return |
| Volatility_20D | Rolling 20-day standard deviation of daily returns |
| Price_to_SMA_20D | Closing price relative to its 20-day moving average |
| Price_to_SMA_50D | Closing price relative to its 50-day moving average |
| Relative_Volume_20D | Daily volume relative to its 20-day average |
| SPY_Return | Daily return of the S&P 500 ETF |
| QQQ_Return | Daily return of the Nasdaq-100 ETF |

### 4. Target variable

The prediction target is the future five-day return:

`Future_Return_5D`

This represents the percentage change between the current closing price and the closing price five trading sessions later.

### 5. Model development

The project explores different regression approaches to predict future returns.

Models investigated include:

- Baseline predictions
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

Hyperparameter tuning is also explored to assess whether model configuration improves predictive performance.

### 6. Model evaluation

A chronological train/test split is used to reflect the time-dependent nature of financial data.

A gap is included between the training and testing periods to reduce overlap between the five-day target windows.

Models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

Model performance is also compared with simple baseline predictions.

### 7. Sector-specific modelling

The project investigates whether models trained on companies within the same sector perform differently from a model trained across all 25 companies.

Technology and Energy are the initial sectors selected for this comparison.

The same features, prediction horizon, chronological split and evaluation metrics are retained to make the comparison meaningful.

## Repository Structure

market_ml_research/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .vscode/
├── data/
├── notebooks/
├── results/
├── src/
├── tests/
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt

### Source code

Reusable project functionality is organised in the `src/market_ml/` Python package.

| Module | Purpose |
|---|---|
| `config.py` | Project configuration and data paths |
| `data_loader.py` | Downloading, processing and validating market data |
| `EDA.py` | Exploratory data analysis functionality |
| `feature_engineering.py` | Creating model features and target variables |
| `models.py` | Model training and evaluation functionality |
| `__init__.py` | Python package initialisation |

### Notebooks

The `notebooks/` directory contains the project's analysis and modelling workflows:

- `Data_prep.ipynb` – data preparation.
- `Exploratory_data_analysis.ipynb` – exploratory analysis.
- `Feature_engineering.ipynb` – feature creation.
- `Modelling_categorical.ipynb` – categorical modelling experiments.
- `Modelling_regression.ipynb` – regression modelling experiments.


## Tests

The `tests/` directory contains automated tests for the project's main components:

- `test_EDA.py`
- `test_data_loader.py`
- `test_feature_engineering.py`
- `test_models.py`

The tests cover data loading and processing, data validation, exploratory data analysis, feature engineering, target generation and model functionality.

The project uses **pytest** for automated testing.

To run the test suite locally, execute the following command from the project root directory:

```bash
pytest -v
```

## Continuous Integration

The project uses **GitHub Actions** to automate testing and help maintain code quality.

The CI workflow is defined in `.github/workflows/ci.yml` and runs automatically when:

- Code is pushed to the `main` branch.
- A pull request is opened against the `main` branch.

The workflow:

1. Checks out the repository.
2. Sets up the Python environment.
3. Installs project dependencies.
4. Installs the project as an editable package.
5. Runs the automated test suite using pytest.

This ensures that changes are tested in a consistent environment and helps identify errors before they are integrated into the main branch.

## Technologies

The project currently uses, or is designed around, the following tools:

- **Python** – primary programming language
- **pandas** – data manipulation and processing
- **NumPy** – numerical operations
- **yfinance** – historical market data
- **scikit-learn** – machine learning and model evaluation
- **XGBoost** – gradient boosting
- **Plotly** – exploratory data visualisation
- **Parquet** – efficient data storage
- **pytest** – automated testing

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/acsimpson16/market_ml_research.git
cd market_ml_research
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On macOS or Linux, activate it using:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Explore the project

Open the notebooks in the `notebooks/` directory to explore the data preparation, exploratory analysis, feature engineering and modelling workflows.

## Current Status

**Work in progress**

| Component | Status |
|---|---|
| Historical data collection | Implemented |
| Data processing and validation | Implemented |
| Exploratory data analysis | In progress |
| Feature engineering | Implemented |
| Baseline and regression models | Implemented |
| Hyperparameter tuning | Explored |
| Sector-specific modelling | In progress |
| Model interpretation | Planned |
| Automated data updates | Planned |
| Interactive dashboard | Planned |

## Limitations

Financial market returns are inherently difficult to predict. Historical relationships may not persist, and model performance can vary across companies, sectors and market conditions.

The project therefore focuses on out-of-sample evaluation and comparison against simple baselines rather than assuming that a model with greater complexity will necessarily perform better.

The results are for research and educational purposes only and should not be interpreted as investment advice.

## Future Development

Potential extensions include:

- Further sector-specific model comparisons.
- Model interpretation using SHAP.
- Additional feature engineering and market indicators.
- More robust time-series cross-validation.
- Automated data updates and database storage.
- An interactive dashboard for exploring model results.
- Improved testing, documentation and reproducibility.

---

**Author:** Aoife Simpson  
**GitHub:** [acsimpson16](https://github.com/acsimpson16)
