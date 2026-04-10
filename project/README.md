# Premier League Match Predictor

## Overview
This project is an end-to-end Machine Learning pipeline that predicts the outcomes (Home Win, Away Win, or Draw) of English Premier League soccer matches. 

The goal of this project is to apply foundational machine learning techniques—specifically Data Cleaning, Feature Engineering (Lag Features), and Classification modeling—to a historically rich, real-world sports dataset.

## The Dataset
The model evaluates **30 years** of historical Premier League data (`premier-league-matches.csv`), charting matches from the 1992/1993 season all the way to 2023. Available here: https://www.kaggle.com/datasets/evangower/premier-league-matches-19922022

**Key Original Features:**
* `Season_End_Year`
* `Wk` (Matchweek)
* `Home` & `Away` (Team Names)
* `HomeGoals` & `AwayGoals`
* `FTR` (Full Time Result: H, A, D)

## Feature Engineering
Sports forecasting is unique because the models cannot look at the goals scored *in the current match* to predict the outcome (which would cause Data Leakage). 

To solve this, advanced **Lag Features** were engineered using Pandas `.groupby()`, `.shift()`, and `.rolling()` windows. This transforms the data into "Time-Series" format. 

**Engineered Features Include:**
* `home_goals_avg`: The moving average of goals scored by the Home team over their previous 5 home matches prior to kickoff.
* `result_encoded`: Categorical target labels mapped to Numerical values (`LabelEncoder`).

## Exploratory Data Analysis (EDA)
The notebook includes detailed visualizations built with Matplotlib and Seaborn to explore the distribution of the data:
1. **Target Distribution Bar Charts:** Proving the statistical prevalence of the "Home Team Advantage".
2. **Goals Over Time Lines:** Tracking the average goals scored per season across 30 years to check for data stationarity.
3. **Correlation Heatmaps:** Proving the predictive strength of our engineered lag features against the final match result.

## Technologies Used
* **Python 3.10+**
* **Pandas & NumPy:** For complex Data Manipulation and Feature Engineering.
* **Scikit-Learn:** For Label Encoding, Data Splitting, Model Training, and Evaluation metrics.
* **Matplotlib & Seaborn:** For Exploratory Data Analysis and visual insights.

## How to Use
1. Clone this repository and ensure the `premier-league-matches.csv` is in the root directory.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Open `model.ipynb` in Jupyter Notebook, VS Code, or Google Colab.
4. Run the cells chronologically to execute the EDA, engineer the features, train the model, and view the prediction scores!

## Future Improvements
* Creating additional lag features based on Win/Loss streaks and head-to-head records.
* Integrating real-time web scraping to predict the upcoming weekend's fixtures dynamically.