# Comparative Study of Influence Analysis Techniques: First Order Analysis vs. TracIn

This repository contains a comparative study of two machine learning model explanation techniques based on Influence Analysis: First Order Analysis and TracIn. The study evaluates their stability and consistency using the Titanic dataset.

## Repository Structure
├── FragilityStudy_notebook.ipynb # Jupyter notebook with the complete study\
├── Score_Analyzer.py # Functions for influence score distribution analysis\
├── Comparison_Metrics.py # Functions for comparing ranked lists\
├── titanic.csv # Titanic dataset used for the study\
└── requirements.txt # Python dependencies

## Study Overview

The study is divided into 5 main steps:

1. **Data Preparation**
   - Loading and preprocessing the Titanic dataset
   - Feature engineering and train-test split

2. **Model Creation**
   - Training machine learning models for influence analysis
   - Model evaluation

3. **Influence Scores Distribution Analysis**
   - Comparative analysis of score distributions between methods
   - Visualization of key distribution characteristics

4. **Consistency Analysis**
   - Evaluation of how consistently each method ranks influential samples
   - Comparison metrics between different runs

5. **Stability Analysis**
   - Assessment of method robustness to data perturbations
   - Measurement of ranking stability across variations

## Key Files

### `Score_Analyzer.py`
Contains functions to analyze and plot information about influence score distributions:

- **Main Distribution Characteristics**
  - Histogram + KDE
  - Box Plot
  - Log-Scale Histogram

- **Tail Analysis (Key for Influence Studies)**
  - Rank-Order Plot (Log-Log)
  - Cumulative Distribution
  - Q-Q Plot

- **Influence Dropoff Analysis**
  - Survival Function
  - Score Dropoff Rate
  - Percentile Plot
  - Tail Proportion

### `Comparison_Metrics.py`
Contains functions to compare two ranked lists using:
- Intersection Count
- Kendall Tau
- Spearman Correlation
- NDCG

## Requirements

To reproduce the experiment, install the required packages using:

```bash
pip install -r requirements.txt

