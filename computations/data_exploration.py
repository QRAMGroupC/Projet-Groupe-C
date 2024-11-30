import numpy as np
import streamlit as st


def compute_exploration_data():
    data_handler = st.session_state["data_handler"]
    current_results = st.session_state["current_results"]

    if current_results.get_result("volatility_annualized") is None:
        commodities_data_df_return = data_handler.get_commodities_returns()
        volatility_annualized = commodities_data_df_return.std() * np.sqrt(252)
        current_results.set_result("volatility_annualized", volatility_annualized)

    if current_results.get_result("corr_matrix") is None:
        commodities_data_df_return = data_handler.get_commodities_returns()
        corr_matrix = commodities_data_df_return.corr()
        current_results.set_result("corr_matrix", corr_matrix)
