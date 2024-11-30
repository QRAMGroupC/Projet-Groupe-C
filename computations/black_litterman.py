import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize
from tqdm import tqdm
from tools.optim import perform_regression_analysis


def gamma_matrix(tau, covariance_matrix):
    return tau * covariance_matrix


def compute_gamma(implied_phi):
    return 1 / implied_phi


def QP_cov(w, cov_matrix, mu_bar, gamma):
    return -1 * (mu_bar.T @ w - gamma * 0.5 * w.T @ cov_matrix @ w)


def compute_black_litterman_portfolio(risk_free_rate, risk_aversion):
    data_handler = st.session_state["data_handler"]
    daily_return = data_handler.get_commodities_returns()
    expected_returns = data_handler.get_annualized_returns()
    sentiment_df = data_handler.get_sentiment_data()

    rolling_window_size = 4

    daily_return = daily_return.dropna()
    weekly_returns = (1 + daily_return).resample("W-THU").prod() - 1
    aligned_data = pd.concat([weekly_returns, sentiment_df], axis=1, join="inner")

    regression_results = perform_regression_analysis(aligned_data, weekly_returns)

    commodities = ["gold", "oil", "gas", "copper", "cobalt", "wheat", "sugar"]
    P = np.eye(len(commodities))

    p_values = regression_results["P-value"].values / 100
    omega = np.diag(p_values)
    tau = 0.05

    alphas = regression_results.set_index("Commodity")["Alpha"]
    betas = regression_results.set_index("Commodity")["Beta"]

    Q = sentiment_df["Sentiment"].values[:, None] * betas.values + alphas.values
    Q = pd.DataFrame(
        Q, index=sentiment_df.index, columns=regression_results["Commodity"]
    )

    equal_weights = np.ones(len(commodities)) / len(commodities)
    covariance_matrix = weekly_returns.cov()
    weeks_per_year = 52

    weekly_portfolio_return = weekly_returns.mean() @ equal_weights
    portfolio_annualized_return = (1 + weekly_portfolio_return) ** weeks_per_year - 1

    weekly_portfolio_volatility = np.sqrt(
        equal_weights @ covariance_matrix @ equal_weights
    )
    portfolio_annualized_volatility = weekly_portfolio_volatility * np.sqrt(
        weeks_per_year
    )

    sharpe_ratio = (
        portfolio_annualized_return - risk_free_rate
    ) / portfolio_annualized_volatility

    numerator = sharpe_ratio * (covariance_matrix @ equal_weights)
    implied_mu = (
        risk_free_rate + numerator / weekly_portfolio_volatility
    ) / weeks_per_year

    implied_mu_df = pd.DataFrame(
        [implied_mu], index=["Implied Mu"], columns=weekly_returns.columns
    )

    mu_bar_df = pd.DataFrame(index=Q.index, columns=weekly_returns.columns)

    for date in tqdm(Q.index):
        Q_week = Q.loc[date].values
        implied_mu_week = implied_mu_df.loc["Implied Mu"].values
        gamma = gamma_matrix(tau, covariance_matrix)
        adjustment_term = Q_week - (P @ implied_mu_week)
        mu_bar_week = (
            implied_mu_week
            + (gamma @ P.T) @ np.linalg.inv(P @ gamma @ P.T + omega) @ adjustment_term
        )
        mu_bar_df.loc[date] = mu_bar_week

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    num_assets = mu_bar_df.shape[1]
    bounds = [(0, 1) for _ in range(num_assets)]
    optimized_weights_dict = {}
    
    mu_bar_values = mu_bar_df.values
    mu_bar_dates = mu_bar_df.index
    
    annual_cov_matrix = covariance_matrix * 52

    for i in tqdm(range(len(mu_bar_df))):
        mu_bar = mu_bar_values[i]
        date = mu_bar_dates[i]
        implied_phi = risk_aversion
        gamma = compute_gamma(implied_phi)
        x0 = np.full(num_assets, 1 / num_assets)
        res = minimize(
            QP_cov,
            x0,
            args=(annual_cov_matrix, mu_bar, gamma),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"disp": False},
        )

        if res.success:
            optimized_weights_dict[date] = res.x
        else:
            print(f"Optimization failed for {date}: {res.message}")

    optimized_weights_df = pd.DataFrame(optimized_weights_dict).T

    optimized_weights_df.columns = mu_bar_df.columns
    optimized_weights_df.index.name = "Date"

    portfolio_with_risk_free = []

    for optimal_weights_tuple in tqdm(optimized_weights_df.itertuples(index=False)):
        optimal_weights = np.array(optimal_weights_tuple)
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_variance_value_bl = np.dot(
            optimal_weights.T, np.dot(covariance_matrix * 52, optimal_weights)
        )
        w_risky = max(
            0,
            min(
                1,
                (portfolio_return - risk_free_rate)
                / (risk_aversion * portfolio_variance_value_bl),
            ),
        )
        combined_portfolio_weights = optimal_weights * w_risky
        risk_free_weight = 1 - w_risky
        portfolio_with_risk_free.append(
            np.append(combined_portfolio_weights, risk_free_weight)
        )

    columns = list(optimized_weights_df.columns) + ["Risk-Free Asset"]
    portfolio_with_risk_free_df_bl = pd.DataFrame(
        portfolio_with_risk_free, index=optimized_weights_df.index, columns=columns
    )

    portfolio_returns = (
        portfolio_with_risk_free_df_bl.iloc[:, :-1] * aligned_data
    ).sum(axis=1)
    risk_free_contribution = portfolio_with_risk_free_df_bl["Risk-Free Asset"] * (
        risk_free_rate / 52
    )
    total_portfolio_returns = portfolio_returns + risk_free_contribution
    cumulative_returns = (1 + total_portfolio_returns).cumprod()
    # Ensure the indices of both DataFrames are in datetime format
    weekly_returns.index = pd.to_datetime(weekly_returns.index)
    portfolio_with_risk_free_df_bl.index = pd.to_datetime(
        portfolio_with_risk_free_df_bl.index
    )

    # Remove the "Risk-Free Asset" column
    optimized_weights_no_rf = portfolio_with_risk_free_df_bl.iloc[:, :-1]

    # Align the weights DataFrame with the weekly_returns index
    aligned_weights = optimized_weights_no_rf.reindex(weekly_returns.index)

    # Check for missing values after alignment
    if aligned_weights.isnull().any().any():
        print("Warning: Missing data after alignment. Filling missing values with 0.")
        aligned_weights = aligned_weights.fillna(0)

    # Calculate portfolio weekly returns
    portfolio_weekly_returns = (weekly_returns * aligned_weights).sum(axis=1)

    # Initialize a dictionary to store annualized returns for each year
    annualized_returns_by_year = {}

    # Calculate annualized return for each year
    for year in portfolio_weekly_returns.index.year.unique():
        # Filter data for the current year
        yearly_returns = portfolio_weekly_returns[
            portfolio_weekly_returns.index.year == year
        ]

        # Calculate the annualized return
        annualized_return = (1 + yearly_returns).prod() ** (
            52 / len(yearly_returns)
        ) - 1
        annualized_returns_by_year[year] = annualized_return

    # Convert results into a DataFrame for better presentation
    annualized_returns_df = pd.DataFrame.from_dict(
        annualized_returns_by_year, orient="index", columns=["Annualized Return"]
    )

    return portfolio_with_risk_free_df_bl, cumulative_returns, annualized_returns_df
