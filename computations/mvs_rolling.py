import numpy as np
import pandas as pd
from scipy.optimize import minimize
import streamlit as st
from tqdm import tqdm

from optim import portfolio_variance


def compute_performance_with_rolling_window(risk_free_rate, risk_aversion):
    print("Launched")
    data_handler = st.session_state["data_handler"]

    commodities_data_df_return = data_handler.get_commodities_returns()
    expected_returns = data_handler.get_annualized_returns()

    final_combined_weights = st.session_state["final_combined_weights"]

    rolling_window = 20
    initial_weights = final_combined_weights[:-1]

    optimal_weights_list = []

    print("rolling window 1")
    for i in tqdm(range(rolling_window, len(commodities_data_df_return))):
        # Get the rolling returns window
        window_returns = commodities_data_df_return.iloc[i - rolling_window : i]

        # Calculate the covariance matrix for the rolling window
        covariance_matrix = window_returns.cov().values * 252

        # Use the previous weights as the starting point for optimization
        n_assets = len(covariance_matrix)
        constraints = {
            "type": "eq",
            "fun": lambda x: np.sum(x) - 1,
        }  # Sum of weights = 1
        bounds = [(0, 1)] * n_assets  # Long-only constraints

        # Optimize portfolio weights
        result = minimize(
            portfolio_variance,
            initial_weights,
            args=(covariance_matrix,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        # Store the optimized weights for this window
        optimal_weights = result.x
        optimal_weights_list.append(optimal_weights)

        # Update initial_weights for the next iteration
        initial_weights = optimal_weights
    print("Done")

    # Convert list to DataFrame for easier handling
    optimal_weights_df = pd.DataFrame(
        optimal_weights_list,
        index=commodities_data_df_return.index[rolling_window:],
        columns=commodities_data_df_return.columns,
    )

    portfolio_with_risk_free = []

    # Loop through the existing optimal weights DataFrame
    print("itertuples")
    for optimal_weights_tuple in tqdm(optimal_weights_df.itertuples(index=False)):
        optimal_weights = np.array(optimal_weights_tuple)

        # Calculate portfolio return and variance using the optimal weights
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_variance_value = np.dot(
            optimal_weights.T, np.dot(covariance_matrix, optimal_weights)
        )

        # Calculate w_risky
        w_risky = max(
            0,
            min(
                1,
                (portfolio_return - risk_free_rate)
                / (risk_aversion * portfolio_variance_value),
            ),
        )

        # Calculate combined weights for risky assets and risk-free weight
        combined_portfolio_weights = optimal_weights * w_risky
        risk_free_weight = 1 - w_risky

        # Append combined weights with the risk-free weight
        portfolio_with_risk_free.append(
            np.append(combined_portfolio_weights, risk_free_weight)
        )
    print("done")

    # Create a new DataFrame for the portfolio with risk-free weights
    columns = list(optimal_weights_df.columns) + ["Risk-Free Asset"]
    portfolio_with_risk_free_df = pd.DataFrame(
        portfolio_with_risk_free, index=optimal_weights_df.index, columns=columns
    )

    # Align daily returns and weights
    aligned_daily_returns = commodities_data_df_return.loc[
        portfolio_with_risk_free_df.index
    ]  # Align indices
    weights = portfolio_with_risk_free_df.iloc[:, :-1]  # Exclude risk-free asset column
    risk_free_weight = portfolio_with_risk_free_df["Risk-Free Asset"]
    risk_free_returns = risk_free_rate / 252  # Daily risk-free return

    # Calculate daily portfolio returns
    portfolio_daily_returns = (weights.values * aligned_daily_returns.values).sum(
        axis=1
    ) + risk_free_weight.values * risk_free_returns

    # Calculate cumulative return
    cumulative_returns_yearly = (1 + portfolio_daily_returns).cumprod()

    # Create a DataFrame for daily returns with dates
    portfolio_daily_returns_df = pd.DataFrame(
        {
            "Date": portfolio_with_risk_free_df.index,
            "Daily Return": portfolio_daily_returns,
        }
    )
    portfolio_daily_returns_df["Year"] = portfolio_daily_returns_df["Date"].dt.year

    # Group by year and calculate annualized return per year
    annualized_returns_yearly = portfolio_daily_returns_df.groupby("Year").apply(
        lambda x: (1 + x["Daily Return"].mean()) ** 252 - 1
    )
    
    print("ready to return")

    return (
        portfolio_with_risk_free_df,
        cumulative_returns_yearly,
        annualized_returns_yearly,
    )
