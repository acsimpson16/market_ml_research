# Market ML Research

![CI](https://github.com/acsimpson16/market_ml_research/actions/workflows/ci.yml/badge.svg)

A Python-based machine learning project investigating whether historical market data and engineered financial features can help predict short-term stock price direction.

## Overview

`market_ml_research` is an end-to-end data science and machine learning project using historical daily financial market data.

The current modelling work focuses on **predicting whether an individual stock will rise or fall over the next five trading sessions**, using Apple (AAPL) as an initial proof of concept.

The wider project uses historical data from 25 US-listed companies across five sectors, alongside market benchmarks. Once the single-stock workflow is established, the analysis will expand to multiple companies and investigate differences in predictive performance across sectors.

The project aims to demonstrate a reproducible workflow covering:

- Financial data collection and processing.
- Exploratory data analysis and visualisation.
- Feature engineering.
- Classification and time-series modelling.
- Model evaluation and interpretation.
- Trading signal generation and backtesting.
- SQL-based data management and interactive dashboards.

## Research Question

**Can historical price, volume and market-related features help classify whether a stock's closing price will be higher or lower five trading sessions into the future?**

The project investigates:

- Whether historical price and volume patterns contain useful information about future price direction.
- Whether market benchmarks provide additional predictive information.
- How different machine learning models perform against simple baseline predictions.
- Whether feature selection improves model performance.
- Whether model predictions can be translated into meaningful trading signals.

## Dataset

Historical daily market data is collected from Yahoo Finance using the `yfinance` Python library.

The wider dataset contains 25 US-listed companies across five sectors:

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

The historical stock data covers approximately ten years of daily observations.

**Current modelling scope:** AAPL is being used for the initial classification proof of concept. The wider dataset will support subsequent multi-stock and sector-level experiments.

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
| Volatility_5D | Rolling five-day return volatility |
| Volatility_10D | Rolling ten-day return volatility |
| Volatility_20D | Rolling twenty-day return volatility |
| SMA_20D | Twenty-day simple moving average |
| SMA_50D | Fifty-day simple moving average |
| Price_to_SMA_20D | Closing price relative to its 20-day moving average |
| Price_to_SMA_50D | Closing price relative to its 50-day moving average |
| Volume_Average_20D | Twenty-day average trading volume |
| Relative_Volume_20D | Daily volume relative to its 20-day average |
| SPY_Return | Daily return of the S&P 500 ETF |
| QQQ_Return | Daily return of the Nasdaq-100 ETF |

### 4. Target variable

The primary modelling target is the direction of the future five-day return.

`Future_Return_5D` represents the percentage change between the current closing price and the closing price five trading sessions later.

The classification target is:

`Target_5D_Direction`

- **1:** Positive five-day future return.
- **0:** Zero or negative five-day future return.
- **Missing:** Future return cannot be calculated because insufficient future observations are available.

The numerical future return is retained separately for subsequent performance analysis and backtesting, but excluded from the model's input features to prevent data leakage.

### 5. Model development

The initial classification experiments use:

- Majority-class baseline.
- Logistic Regression.
- Random Forest Classifier.
- XGBoost Classifier.

Feature selection is explored using permutation importance, measuring the effect of shuffling individual features on validation ROC-AUC.

A reduced feature set has been investigated to assess whether removing less informative features improves model performance.

The selected-feature XGBoost experiment currently uses:

- Return_20D
- Return_60D
- SMA_50D
- Price_to_SMA_50D
- Price_to_SMA_20D
- Volume_Average_20D
- Volume
- SPY_Return

The earlier regression experiments are retained in `Modelling_regression.ipynb` as exploratory work.

### 6. Model evaluation

Chronological training, validation and test splits are used to reflect the time-dependent nature of financial data.

A gap is included between data partitions to reduce overlap between five-day target windows.

Classification models are evaluated using:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrices

Model performance is compared against a simple majority-class baseline.

#### Preliminary validation results

The following results are from the current AAPL classification experiments. They are validation results, not final out-of-sample test results.

| Model | Accuracy | Balanced accuracy | Precision | Recall | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.530 | 0.500 | 0.530 | 1.000 | 0.500 | 0.530 |
| Logistic Regression | 0.494 | 0.492 | 0.523 | 0.526 | 0.518 | 0.554 |
| Random Forest | 0.506 | 0.517 | 0.557 | 0.333 | 0.516 | 0.580 |
| XGBoost | 0.544 | 0.563 | 0.696 | 0.250 | 0.555 | 0.625 |
| XGBoost Reduced | 0.506 | 0.520 | 0.570 | 0.276 | 0.496 | 0.576 |
| **XGBoost Reduced Selected** | **0.583** | **0.600** | **0.747** | **0.323** | **0.633** | **0.693** |

The selected-feature XGBoost model currently provides the strongest validation performance across the evaluated metrics.

Further threshold exploration has been performed using the validation set. A threshold of 0.50 is provisionally retained because it achieved the highest balanced accuracy among the thresholds tested.

The final model and threshold will be evaluated on the untouched test set before drawing conclusions about out-of-sample performance.

## Repository Structure

```text
market_ml_research/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── Data_prep.ipynb
│   ├── Exploratory_data_analysis.ipynb
│   ├── Feature_engineering.ipynb
│   ├── Modelling_categorical.ipynb
│   └── Modelling_regression.ipynb
│
├── results/
│   └── README.md
│
├── src/
│   └── market_ml/
│       ├── __init__.py
│       ├── config.py
│       ├── data_loader.py
│       ├── EDA.py
│       ├── feature_engineering.py
│       └── models.py
│
├── tests/
│   ├── test_EDA.py
│   ├── test_data_loader.py
│   ├── test_feature_engineering.py
│   └── test_models.py
│
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```

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
- `Modelling_categorical.ipynb` – classification modelling experiments.
- `Modelling_regression.ipynb` – earlier regression modelling experiments.

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
- **GitHub Actions** – continuous integration

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
| Exploratory data analysis | Implemented / ongoing |
| Feature engineering | Implemented |
| Target generation for five-day direction | Implemented |
| Classification baseline and initial models | Implemented |
| XGBoost feature selection | Explored |
| Validation threshold exploration | Explored |
| Final test-set evaluation | Next step |
| ARIMA time-series modelling | Planned |
| LSTM modelling | Planned |
| Trading signal generation | Planned |
| Backtesting | Planned |
| SQL database integration | Planned |
| Interactive dashboard | Planned |
| Multi-stock and sector-level modelling | Planned |

## Limitations

Financial market returns are inherently difficult to predict. Historical relationships may not persist, and model performance can vary across companies, sectors and market conditions.

Classification performance does not necessarily translate into profitable trading strategies. Transaction costs, slippage, execution timing and risk must be considered when evaluating any resulting trading signals.

The project therefore focuses on chronological out-of-sample evaluation and comparison against simple baselines rather than assuming that a more complex model will necessarily perform better.

The results are for research and educational purposes only and should not be interpreted as investment advice.

## Future Development

Planned extensions include:

- Final evaluation using the untouched test set.
- Further time-series modelling using ARIMA.
- LSTM modelling as a deep learning proof of concept.
- Model interpretation and further feature analysis.
- Trading signal generation using predicted probabilities.
- Backtesting with transaction costs and execution assumptions.
- SQL database integration for storing market data and model outputs.
- An interactive dashboard for exploring predictions and strategy performance.
- Expansion to multiple stocks and sector-specific comparisons.
- Automated data updates and improved reproducibility.

---

**Author:** Aoife Simpson  
**GitHub:** [acsimpson16](https://github.com/acsimpson16)
