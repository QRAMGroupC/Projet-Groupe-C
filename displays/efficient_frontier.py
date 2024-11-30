import streamlit as st
from computations.efficient_frontier import compute_efficient_frontier


def display_efficient_frontier():
    # Adjustable Risk-Free Rate
    current_results = st.session_state["current_results"]

    if st.session_state["last_page"] != "efficient_frontier":
        slider_value = current_results.get_result("ef_last_risk_free_rate", 0.04)
        st.session_state["last_page"] = "efficient_frontier"
    else: 
        slider_value = 0.04

    st.header("Efficient Frontier with Adjustable Risk-Free Rate")
    st.info(
        "This page calculates the efficient frontier, treating all commodities as risky assets. You can adjust the risk-free rate dynamically."
    )

    risk_free_return = st.slider(
        "Adjust Risk-Free Rate (Risk-Free Asset)",
        0.0,
        0.07,
        slider_value,
        0.01,
    )
    st.session_state["slider_value"] = risk_free_return
    st.write(f"Selected Risk-Free Rate: {risk_free_return:.2%}")

    chart_data, tangency_weights_df, mu_tangency, vol_tangency = (
        compute_efficient_frontier(risk_free_return)
    )

    # Display the Efficient Frontier
    st.write("### Efficient Frontier")
    st.scatter_chart(chart_data, x="vol", y="mu", color="color", size="size")

    st.write("Legend:")
    st.markdown(
        "- Blue points: Efficient Frontier portfolios (optimized for different risk aversions)"
    )
    st.markdown("- Red point: Risk-Free Rate")
    st.markdown("- Orange point: Tangency Portfolio (maximum Sharpe ratio)")

    # Display Tangency Portfolio Weights
    st.subheader("Tangency Portfolio Weights")
    st.write(tangency_weights_df)

    st.write(f"Expected Return of Tangency Portfolio: {mu_tangency:.2%}")
    st.write(f"Volatility of Tangency Portfolio: {vol_tangency:.2%}")
