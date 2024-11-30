import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize


def compute_efficient_frontier(risk_free_return):
    data_handler = st.session_state["data_handler"]
    current_results = st.session_state["current_results"]
    
    last_risk_free_rate = current_results.get_result("ef_last_risk_free_rate")
    if last_risk_free_rate == risk_free_return:
        return (
            current_results.get_result("chart_data"),
            current_results.get_result("tangency_weights"),
            current_results.get_result("mu_tangency"),
            current_results.get_result("vol_tangency"),
        )
        
    commodities_data_df_return = data_handler.get_commodities_returns()
    annual_mu = data_handler.get_annualized_returns()
    cov_matrix = data_handler.get_covariance_matrix()

    # Calculate Efficient Frontier
    mu_efficient_frontier = []
    vol_efficient_frontier = []
    gammas = np.linspace(0.01, 3, 50)

    for gam in gammas:
        def objective(weights):
            return 0.5 * np.dot(weights.T, np.dot(cov_matrix, weights)) - gam * np.dot(weights, annual_mu)


        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(annual_mu)))

        res = minimize(objective, np.ones(len(annual_mu)) / len(annual_mu), method='SLSQP', bounds=bounds, constraints=cons)

        if res.success:
            weights = res.x
            mu_portfolio = np.dot(weights, annual_mu)
            vol_portfolio = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            mu_efficient_frontier.append(mu_portfolio)
            vol_efficient_frontier.append(vol_portfolio)

    # Tangency Portfolio Calculation
    weights_tangency = (np.linalg.inv(cov_matrix) @ (annual_mu - risk_free_return)) / (
            np.ones(cov_matrix.shape[0]) @ np.linalg.inv(cov_matrix) @ (annual_mu - risk_free_return)
    )
    weights_tangency = np.clip(weights_tangency, 0, 1)
    weights_tangency /= weights_tangency.sum()

    mu_tangency = np.dot(weights_tangency, annual_mu)
    vol_tangency = np.sqrt(np.dot(weights_tangency.T, np.dot(cov_matrix, weights_tangency)))

    chart_data = pd.DataFrame({
        "vol": vol_efficient_frontier,
        "mu": mu_efficient_frontier,
        "color": "#0000FF",  # Blue for Efficient Frontier
        "size": 8,
    })
    
    chart_data = pd.concat([
        chart_data,
        pd.DataFrame({
            "vol": [0],
            "mu": [risk_free_return],
            "color": ["#FF0000"],  # Red for Risk-Free
            "size": [10],
        })
    ], ignore_index=True)
    
    chart_data = pd.concat([
        chart_data,
        pd.DataFrame({
            "vol": [vol_tangency],
            "mu": [mu_tangency],
            "color": ["#FFA500"],  # Orange for Tangency Portfolio
            "size": [10],
        })
    ], ignore_index=True)
    
    tangency_weights_df = pd.DataFrame({
        "Asset": commodities_data_df_return.columns,
        "Weight": weights_tangency
    }).sort_values(by="Weight", ascending=False)

    current_results.set_result("chart_data", chart_data)
    current_results.set_result("tangency_weights", tangency_weights_df)
    current_results.set_result("mu_tangency", mu_tangency)
    current_results.set_result("vol_tangency", vol_tangency)
    current_results.set_result("ef_last_risk_free_rate", risk_free_return)
    
    return chart_data, tangency_weights_df, mu_tangency, vol_tangency
