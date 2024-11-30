import numpy as np
import pandas as pd
import streamlit as st

from optim import optimal_portfolio_markowitz, portfolio_variance


def compute_minimum_variance_strategy(risk_free_rate, risk_aversion):
    data_handler = st.session_state["data_handler"]
    current_results = st.session_state["current_results"]

    last_risk_free_rate = current_results.get_result("mvs_nor_last_risk_free_rate")
    last_risk_aversion = current_results.get_result("mvs_nor_last_risk_aversion")
    if last_risk_free_rate == risk_free_rate and last_risk_aversion == risk_aversion:
        return (
            current_results.get_result("final_combined_weights"),
            current_results.get_result("asset_names_with_rf"),
            current_results.get_result("w_risky"),
            current_results.get_result("risk_free_weight"),
            current_results.get_result("portfolio_return"),
            current_results.get_result("portfolio_volatility"),
            current_results.get_result("weights_df"),
        )

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

    # Save final weights to session state for later use
    current_results.set_result("final_combined_weights", final_combined_weights)
    st.session_state["final_combined_weights"] = final_combined_weights
    current_results.set_result("asset_names_with_rf", asset_names_with_rf)
    current_results.set_result("w_risky", w_risky)
    current_results.set_result("risk_free_weight", risk_free_weight)
    current_results.set_result("portfolio_return", combined_portfolio_return)
    current_results.set_result(
        "portfolio_volatility",
        np.sqrt(combined_portfolio_variance),
    )
    current_results.set_result("weights_df", weights_df)

    current_results.set_result("mvs_nor_last_risk_free_rate", last_risk_free_rate)
    current_results.set_result("mvs_nor_last_risk_aversion", last_risk_aversion)

    return (
        final_combined_weights,
        asset_names_with_rf,
        w_risky,
        risk_free_weight,
        combined_portfolio_return,
        np.sqrt(combined_portfolio_variance),
        weights_df,
    )
