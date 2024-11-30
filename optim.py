import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize
import statsmodels.api as sm


@st.cache_data
def portfolio_variance(weights, covariance_matrix):
    return weights.T @ covariance_matrix @ weights


@st.cache_data
def optimal_portfolio_markowitz(cov_matrix: np.ndarray) -> np.ndarray:
    n_assets = len(cov_matrix)
    initial_weights = np.ones(n_assets) / n_assets
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))

    result = minimize(
        portfolio_variance,
        initial_weights,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        st.error(f"Optimization failed: {result.message}")

    return result.x


@st.cache_data
def perform_regression_analysis(aligned_data, weekly_returns):
    X = aligned_data["Sentiment"].values.reshape(-1, 1)
    X = sm.add_constant(X)

    results_list = []
    for commodity in weekly_returns.columns:
        y = aligned_data[commodity].values
        model = sm.OLS(y, X).fit()
        results_list.append(
            {
                "Commodity": commodity,
                "Beta": model.params[1],
                "Alpha": model.params[0],
                "R-squared": model.rsquared,
                "P-value": model.pvalues[1],
            }
        )

    return pd.DataFrame(results_list)
