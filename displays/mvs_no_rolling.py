import matplotlib.pyplot as plt
import streamlit as st
from computations.mvs_no_rolling import compute_minimum_variance_strategy


def display_minimum_variance_strategy():
    st.header("Optimal Portfolio Weights with Adjustable Risk-Free Rate")
    st.info(
        "Use the sliders below to adjust the risk-free rate and risk aversion, and observe how the portfolio weights and metrics update dynamically."
    )

    slider_risk_free_value = st.session_state.get(
        "mvs_no_rolling_risk_free_rate", 0.04
    )
    slider_risk_aversion_value = st.session_state.get(
        "mvs_no_rolling_risk_aversion", 3.0
    )

    risk_free_rate = st.slider(
        "Adjust Risk-Free Rate",
        0.0,
        0.09,
        slider_risk_free_value,
        0.01,
    )
    risk_aversion = st.slider(
        "Select Risk Aversion",
        1.00,
        5.0,
        slider_risk_aversion_value,
        1.0,
    )

    if st.button("Confirm and Compute"):
        # Update session state with slider values
        st.session_state["mvs_no_rolling_risk_free_rate"] = risk_free_rate
        st.session_state["mvs_no_rolling_risk_aversion"] = risk_aversion

        with st.spinner("Computing rolling window performance..."):
            # Perform computation
            (
                final_combined_weights,
                asset_names_with_rf,
                w_risky,
                risk_free_weight,
                combined_portfolio_return,
                combined_portfolio_volatility,
                weights_df,
            ) = compute_minimum_variance_strategy(risk_free_rate, risk_aversion)

            # Store the results in session state
            st.session_state["final_combined_weights"] = final_combined_weights
            st.session_state["asset_names_with_rf"] = asset_names_with_rf
            st.session_state["w_risky"] = w_risky
            st.session_state["risk_free_weight"] = risk_free_weight
            st.session_state["combined_portfolio_return"] = combined_portfolio_return
            st.session_state["combined_portfolio_volatility"] = (
                combined_portfolio_volatility
            )
            st.session_state["weights_df"] = weights_df
        st.success("Computation completed!")

    if "final_combined_weights" not in st.session_state:
        st.warning(
            "Please confirm the slider values to compute the portfolio performance."
        )
        return

    final_combined_weights = st.session_state["final_combined_weights"]
    asset_names_with_rf = st.session_state["asset_names_with_rf"]
    w_risky = st.session_state["w_risky"]
    risk_free_weight = st.session_state["risk_free_weight"]
    combined_portfolio_return = st.session_state["combined_portfolio_return"]
    combined_portfolio_volatility = st.session_state["combined_portfolio_volatility"]
    weights_df = st.session_state["weights_df"]

    # Define colors for the assets
    color_map = {
        "gold": "#FFDDC1",  # Soft peach
        "oil": "#FFABAB",  # Light coral
        "gas": "#FFC3A0",  # Pastel orange
        "copper": "#D5AAFF",  # Lavender
        "aluminium": "#85E3FF",  # Light sky blue
        "wheat": "#FFFFB5",  # Soft yellow
        "sugar": "#FF9CEE",  # Light pink
        "Risk-Free Asset": "#B9FBC0",  # Mint green
    }

    # Filter weights below 0.1% and prepare data for the donut chart
    filtered_weights = [
        (name, weight)
        for name, weight in zip(asset_names_with_rf, final_combined_weights)
        if weight >= 0.001
    ]
    filtered_names, filtered_values = (
        zip(*filtered_weights) if filtered_weights else ([], [])
    )
    filtered_colors = [color_map[name] for name in filtered_names]

    # Display weights and metrics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Optimal Portfolio Weights")
        st.dataframe(weights_df.set_index("Asset"))

    with col2:
        st.subheader("Portfolio Metrics")
        st.markdown(f"- **Proportion Allocated to Risky:** {w_risky:.2%}")
        st.markdown(f"- **Proportion Allocated to Risk-Free:** {risk_free_weight:.2%}")
        st.markdown(f"- **Expected Portfolio Return:** {combined_portfolio_return:.2%}")
        st.markdown(f"- **Portfolio Volatility:** {combined_portfolio_volatility:.2%}")

    # Donut chart for portfolio allocation
    st.subheader("Portfolio Allocation")

    fig, ax = plt.subplots(figsize=(6, 6))
    _ = ax.pie(
        filtered_values,
        labels=filtered_names,
        autopct=lambda p: f"{p:.1f}%" if p > 0 else "",
        startangle=90,
        colors=filtered_colors,
        wedgeprops=dict(width=0.3),  # Makes it a donut chart
        textprops=dict(color="black"),  # Text color for better readability
    )
    ax.axis("equal")  # Equal aspect ratio ensures the donut is circular
    ax.set_title("Portfolio Allocation", fontsize=16)
    st.pyplot(fig)
