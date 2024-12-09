import matplotlib.pyplot as plt
import streamlit as st

from tools.data_management import get_sp500_data


def display_comparison():
    st.title("Comparison between Min Var and Black-litterman strategy")
    try:
        annualized_returns_yearly = st.session_state["annualized_returns_yearly"]
        annualized_returns_df = st.session_state["annualized_returns_df"]
    except KeyError as e:
        st.success(
            "Make sure you have completed the 3 previous steps before analyzing the performance."
        )

    if not (
        st.session_state.get("mvs_no_rolling_risk_free_rate")
        == st.session_state.get("mvs_rolling_risk_free_rate")
        and st.session_state.get("mvs_no_rolling_risk_free_rate")
        == st.session_state.get("bl_risk_free_rate")
        and st.session_state.get("mvs_no_rolling_risk_aversion")
        == st.session_state.get("mvs_rolling_risk_aversion")
        and st.session_state.get("mvs_no_rolling_risk_aversion")
        == st.session_state.get("bl_risk_aversion")
    ):
        st.warning(
            "Please make sure to use the same risk free rate and risk aversion for all 3 strategies for fair comparison."
        )
        return

    # === Graph 1: Annualized Returns Comparison ===
    st.subheader("1. Comparison of the Annualized Portfolio Returns by Year")

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    annualized_returns_yearly.plot(
        kind="bar", ax=ax1, color="blue", alpha=0.7, label="Portfolio"
    )
    ax1.set_title("Annualized Returns by Year of Min Var")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Annualized Return")
    ax1.legend()
    ax1.grid(axis="y")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    annualized_returns_df.plot(
        kind="bar", ax=ax2, color="green", alpha=0.7, label="Portfolio"
    )
    ax2.set_title("Annualized Returns by Year of Black-Litterman")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Annualized Return")
    ax2.legend()
    ax2.grid(axis="y")
    st.pyplot(fig2)

    # Graph 2
    st.subheader("2.Min Var vs Black-litterman Cumulative Returns")
    cumulative_returns = st.session_state["cumulative_returns"]
    cumulative_returns_yearly = st.session_state["cumulative_returns_yearly"]
    portfolio_with_risk_free_df = st.session_state["portfolio_with_risk_free_df"]
    sp500_data = get_sp500_data(
        portfolio_with_risk_free_df.index[0],
        portfolio_with_risk_free_df.index[-1],
    )
    # Create a single plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        sp500_data.index,
        sp500_data["Cumulative_Return"],
        label="S&P 500 ",
        color="orange",
    )
    ax.plot(
        portfolio_with_risk_free_df.index,
        cumulative_returns_yearly,
        label="Min Var portfolio",
        color="blue",
    )
    ax.plot(cumulative_returns, label="Black-litterman portfolio", color="green")
    # Set titles and labels
    ax.set_title("Black-litterman portfolio vs S&P 500")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Returns")
    ax.legend()
    ax.grid()
    st.pyplot(fig)
