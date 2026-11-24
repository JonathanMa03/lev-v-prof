
# Modeling the Leverage–Profitability Relationship

A statistical learning investigation of the relationship between financial leverage and corporate profitability among publicly traded U.S. manufacturing firms.

Author: ![Jonathan Ma](jonathanma03.github.io)

## Overview

The relationship between financial leverage and corporate profitability is a longstanding question in corporate finance. Debt financing may provide benefits through tax advantages and managerial discipline, while excessive leverage may introduce financial distress, agency costs, and other inefficiencies. These competing mechanisms suggest that the relationship between leverage and profitability may be nonlinear.

Dalci (2018) documents an inverted U-shaped relationship among publicly listed manufacturing firms in China using panel-data econometric methods. This project revisits the underlying research question in the context of U.S. manufacturing firms, emphasizing statistical learning approaches to functional-form estimation, model complexity, and generalization. We aim to investigate whether the data support a linear, quadratic, or more flexible relationship.

## Research Questions

The project is organized around three questions:

1. **Functional form:** What is the estimated relationship between financial leverage and profitability among publicly traded U.S. manufacturing firms? Is there evidence of nonlinearity?

2. **Model complexity:** Do flexible statistical learning methods, such as regression splines and generalized additive models, reveal patterns that simpler parametric models fail to capture?

3. **Generalization and stability:** Does additional model complexity improve out-of-sample performance, and how stable are the estimated relationships across samples and model specifications?

The primary objective is to characterize the conditional relationship between leverage and profitability. Predictive performance is used to evaluate model generalization and the usefulness of additional complexity.

The analysis is observational and does not seek to establish a causal effect of leverage on profitability.

## Data

The project will construct a firm-level dataset from publicly available U.S. financial information.

### Primary Data Sources

- **SEC EDGAR:** Financial statement data reported in annual filings, including assets, liabilities, debt, net income, equity, and revenue.
- **Yahoo Finance (`yfinance`):** Supplementary company information, financial statement data, and validation of selected accounting variables.

SEC filings will serve as the primary source for accounting fundamentals. Yahoo Finance may supplement missing information where definitions and reporting periods can be verified.

### Target Population

Publicly traded U.S. manufacturing firms, provisionally identified using Standard Industrial Classification (SIC) codes 2000–3999.

The initial analysis will focus on a single fiscal-year cross-section, with the possibility of retaining multiple years for supplementary analyses.

### Candidate Variables

**Profitability measures:**

- Return on Assets (ROA)
- Return on Equity (ROE)

**Leverage measures:**

- Total debt relative to total assets
- Short-term debt relative to total assets

**Potential controls:**

- Firm size, measured by log total assets
- Revenue growth
- Manufacturing industry classification

Final variable definitions, sample restrictions, and accounting conventions will be documented after the data collection and validation stage.

## Methodology

The analysis will proceed from simple parametric models to more flexible statistical learning methods.

### 1. Exploratory Data Analysis

- Examine distributions of profitability, leverage, and firm characteristics.
- Identify missing values and influential observations.
- Visualize the leverage–profitability relationship.
- Investigate variation across manufacturing industries.

### 2. Parametric Regression

Estimate linear and quadratic regression models as benchmarks.

The quadratic specification provides a direct comparison with the inverted U-shaped relationship proposed in the motivating literature.

### 3. Nonlinear Statistical Learning

Investigate more flexible functional forms using:

- Polynomial regression
- Regression splines
- Smoothing splines
- Generalized additive models (GAMs)

Particular attention will be given to basis expansions, regularization, smoothness, and effective degrees of freedom.

### 4. Model Evaluation

Compare specifications using appropriate statistical and predictive criteria, potentially including:

- Cross-validation
- Out-of-sample prediction error
- Model complexity and interpretability
- Bootstrap uncertainty estimates
- Stability of estimated nonlinear relationships

The purpose is not necessarily to identify the most complex or predictively accurate model, but to determine whether additional flexibility is supported by the data.

### 5. Robustness Analysis

Potential robustness checks include:

- Alternative definitions of financial leverage
- Alternative profitability measures
- Sensitivity to influential observations
- Industry-specific heterogeneity
- Sensitivity to data-source coverage and accounting definitions

The final set of analyses will depend on data availability and initial findings.

## Repository Structure

```text
lev-v-prof/
├── code/
│   └──
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── .gitignore
├── LICENSE
├── README.md
├── requirements_py.txt
└── requirements_r.txt
```

### Directory Descriptions

- `code/`: Python scripts for data engineering and R scripts for statistical analysis, visualization, and reporting.
- `data/raw/`: Original data retrieved from external sources, preserved before transformation.
- `data/processed/`: Cleaned and analysis-ready datasets.
- `requirements_py.txt`: Python dependencies for data retrieval, processing, and validation.
- `requirements_r.txt`: R package dependencies for modeling and reporting.

Additional directories may be introduced as the project develops.

## Technology Stack

### Python — Data Engineering

Python will be used for:

- Retrieving SEC financial statement data
- Collecting supplementary Yahoo Finance information
- Matching company identifiers
- Harmonizing reporting periods and accounting variables
- Cleaning, validating, and exporting datasets

### R — Statistical Analysis

R will be used for:

- Exploratory analysis and visualization
- Regression modeling
- Nonlinear statistical learning
- Cross-validation and resampling
- Statistical inference and robustness analysis
- Reproducible reporting through Quarto

The data engineering and statistical analysis stages will remain separate to support reproducibility.

## Project Status

**Status:** Initial development — data collection and feasibility assessment.

The immediate priority is to establish a reproducible data pipeline and evaluate the availability and consistency of the financial variables required for the analysis.

Model specifications and the final sample will be determined after this initial assessment.

## References

Dalci, I. (2018). Impact of financial leverage on profitability of listed manufacturing firms in China. *Pacific Accounting Review, 30*(4), 410–432.

https://doi.org/10.1108/PAR-01-2018-0008

## License

See [LICENSE](LICENSE) for licensing information.
