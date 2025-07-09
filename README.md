# Binance Counterparty PnL Analyzer

This script analyzes Binance aggregated trade data (`aggTrades`) to estimate how profitable the counterparty (taker) is — assuming **you are always the maker**.

## What It Does

- Loads multiple Binance aggTrades CSV files.
- Assumes you are the maker in all trades.
- Calculates unrealized profit/loss (PnL) over time.
- Estimates trade flow toxicity using regression.
- Plots PnL and rolling metrics.
- Saves a combined data file and PnL chart.

## How to Use

1. Run `pull_data.main(START_DATE, END_DATE, SYMBOL)` to download trade data.
2. Initialize and run the analyzer:

```
analyzer = BinanceCounterPartyPNL(csv_pattern="SYMBOL-aggTrades-*.csv")
analyzer.run_analysis(save_plot=True, save_combined_data=True)
```

## Requirements

Python 3.7+

pandas, numpy, matplotlib, scikit-learn

Install with:
```pip install pandas numpy matplotlib scikit-learn
```

Combined trade data CSV
PnL and toxicity plots
Console summary of results
Note: Make sure CSV files follow Binance's aggTrades format and are named like SYMBOL-aggTrades-YYYY-MM-DD.csv
