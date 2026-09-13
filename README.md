# Uber Ride Analysis 

An end-to-end data science project on Uber ride-booking data: cleaning, exploratory
analysis, visualization, and a predictive model for daily ride demand.

## 1. Project Structure

```
Uber_Ride_Data_Analysis/
├── data/
│   └── Uber_Drive_Raw.csv                # raw dataset
├── notebooks/
│   ├── 00_Uber_Ride_Data_Analysis_Full_code.ipynb   # ★ complete, self-contained pipeline
│   ├── 01_importing_all_the_necessary_libraries.ipynb
│   ├── 02_suppress_warnings.ipynb
│   ├── 03_data_loading.ipynb
│   ├── 04_initial_data_exploration.ipynb
│   ├── 05_data_preprocessing.ipynb
│   ├── 06_exploratory_data_analysis_data_visualization.ipynb
│   └── 07_predictive_modeling_demand_forecast.ipynb
├── src/
│   ├── preprocessing.py                  # shared cleaning / feature-engineering pipeline
│   └── utils.py                          # small plotting helpers
├── requirements.txt
└── README.md
```

`00_Uber_Ride_Data_Analysis_Full_code.ipynb` is the single, complete pipeline —
run this one top-to-bottom for the whole project (loading → cleaning → EDA →
visualization → demand forecasting). Notebooks `01`–`07` hold the same work
split by topic for easier review; `07` is also runnable standalone, since it
rebuilds the cleaned data itself via `src/preprocessing.py`.

**Setup:** `pip install -r requirements.txt`, then open the notebooks from
inside the `notebooks/` folder (the data path is relative: `../data/Uber_Drive_Raw.csv`).

## 2. Dataset

- **Source:** `data/Uber_Drive_Raw.csv` — a synthetic Uber ride-booking log, 1,035 rows × 21 columns.
- **Time span:** January 1, 2025 – June 30, 2025.
- **Key fields:** `Date`, `Time`, `Booking ID`, `Booking Status`, `Customer ID`,
  `Vehicle Type`, `Pickup/Drop Location`, `waiting time`, `Avg CTAT`,
  cancellation/incompletion flags and reasons, `Booking Value`, `Ride Distance`,
  `Driver Ratings`, `Customer Rating`, `Payment Method`.

### Preprocessing steps (`src/preprocessing.py`, notebook `05`)
- Removed 34 exact duplicate rows.
- Filled cancellation/incompletion "reason" columns with `"Not Applicable"` where a ride wasn't cancelled/incomplete.
- Coerced numeric columns (`waiting time`, `Avg CTAT`, `Booking Value`, `Ride Distance`, `Driver Ratings`, `Customer Rating`) and imputed missing values with the column median.
- Imputed missing categorical fields (`Vehicle Type`, `Pickup/Drop Location`, `Payment Method`, `Customer ID`) with the column mode.
- Parsed `Date`/`Time` into proper datetime types and engineered: `Year`, `Month`, `Month_Name`, `Day`, `Day_of_week`, `Is_Weekend`, `Hour`, `Day-Night` (Morning/Afternoon/Evening/Night), and `Peak_Hour` (7–10 AM & 5–8 PM).
- Checked numeric columns (`Ride Distance`, `Booking Value`, `Avg CTAT`) for outliers via boxplots.

## 3. Key Insights

- **Booking outcomes:** 753 completed (73%), 116 cancelled by driver (11%), 94 cancelled by customer (9%), 72 incomplete (7%).
- **Fleet mix:** ride volume is fairly even across vehicle types — UberXL (221), Uber Green (218), UberX (207), Uber Comfort (195), Uber Black (179).
- **Payments:** Credit Card (187) and Cash (183) are the most common methods, with UPI, PayPal, Debit Card and Wallet all in a similar 148–172 range — no single method dominates.
- **Pricing:** average booking value is **$40.73**, average ride distance is **18.1 km**.
- **Demand by hour:** ride volume is fairly flat across the day with modest bumps around 9–10 AM and 5 PM–9 PM, consistent with the defined peak-hour windows (7–10 AM, 5–8 PM), but there is no single dominant rush period in this dataset.
- **Cancellations by time of day:** the booking-status vs. day/night heatmap (notebook `06`) shows how cancellation and completion rates shift between Morning/Afternoon/Evening/Night — see the row-normalized heatmap for the clearest read.

## 4. Predictive Modeling — Ride Demand Forecasting

Rides are aggregated to a **daily** grain (ride count per day), with calendar
features (day of week, month, weekend flag, day of month) used to predict the
next day's ride count.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 1.88 | 2.33 | 0.025 |
| Random Forest (200 trees, depth 5) | 1.99 | 2.49 | −0.11 |

**Interpretation:** both models perform weakly (R² near/below zero), meaning
calendar features alone barely explain day-to-day ride-count variation in this
dataset. This is expected given the small sample (181 days) and the synthetic,
fairly noise-dominated nature of the data — daily ride counts don't show a
strong weekly/monthly seasonal pattern to learn from. The pipeline (train/test
split, multiple models, MAE/RMSE/R² evaluation, actual-vs-predicted plot,
feature importance) is production-ready; with a longer history or additional
signals (weather, local events, marketing spend) it would likely generalize
better.

## 5. Visualizations Included

Histograms, pickup/drop location bar charts, peak-hour bar charts, pricing
trend plots, a correlation heatmap, cancellation-reason breakdowns, payment
method distribution, a booking-status pie chart, and booking-status × time-of-day
heatmaps (raw counts and row-normalized), plus the demand-forecasting
scatter and feature-importance plots.

## 6. Challenges & Improvements

- **Data limitations:** six months of data with no weather, traffic, or event
  signals limits how predictive a demand model can be; the two models here are
  a working baseline, not tuned to their ceiling.
- **Next steps:** add lag/rolling features (yesterday's ride count, 7-day
  average), bring in external features (weather, holidays), and try
  time-series-specific models (e.g., Prophet, SARIMA, or gradient boosting
  with lag features) once more historical data is available.
- **Business applications:** the day/hour and location breakdowns support
  driver-supply scheduling (allocate more drivers to peak windows), the
  cancellation analysis can guide interventions to reduce driver-side
  cancellations (the largest cancellation category), and the demand-forecasting
  pipeline is a template for staffing/pricing decisions as more data accumulates.
