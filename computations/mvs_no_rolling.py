import numpy as np
import pandas as pd
import streamlit as st

from optim import optimal_portfolio_markowitz, portfolio_variance


def compute_minimum_variance_strategy(risk_free_rate, risk_aversion):
    data_handler = st.session_state["data_handler"]

    commodities_data_df_return = data_handler.get_commodities_returns()
    annualized_returns = data_handler.get_annualized_returns()
    covariance_matrix = data_handler.get_covariance_matrix()

    # Step 1: Compute optimal risky portfolio weights
    optimal_weights = optimal_portfolio_markowitz(covariance_matrix)

    # Step 2: Compute risky portfolio return and variance
    portfolio_return = np.dot(optimal_weights, annualized_returns)
    portfolio_variance_value = portfolio_variance(optimal_weights, covariance_matrix)

    # Step 3: Proportion allocated to risky portfolio
    w_risky = (portfolio_return - risk_free_rate) / (
        risk_aversion * portfolio_variance_value
    )
    w_risky = max(0, min(1, w_risky))  # Ensure the proportion is between 0 and 1

    # Step 4: Compute combined weights
    combined_portfolio_weights = optimal_weights * w_risky  # Risky asset weights
    risk_free_weight = 1 - w_risky  # Risk-free asset weight

    # Asset names with Risk-Free Asset
    asset_names = list(commodities_data_df_return.columns)
    asset_names_with_rf = asset_names + ["Risk-Free Asset"]

    # Combine final weights, including the risk-free weight
    final_combined_weights = np.append(combined_portfolio_weights, risk_free_weight)

    # Ensure non-negative weights and normalize
    final_combined_weights = np.clip(final_combined_weights, 0, None)
    final_combined_weights /= final_combined_weights.sum()

    # Compute combined portfolio metrics
    combined_portfolio_variance = w_risky**2 * portfolio_variance_value
    combined_portfolio_return = (w_risky * portfolio_return) + (
        risk_free_weight * risk_free_rate
    )

    weights_df = pd.DataFrame(
        {"Asset": asset_names_with_rf, "Weight": final_combined_weights}
    ).sort_values(by="Weight", ascending=False)

    return (
        final_combined_weights,
        asset_names_with_rf,
        w_risky,
        risk_free_weight,
        combined_portfolio_return,
        np.sqrt(combined_portfolio_variance),
        weights_df,
    )
