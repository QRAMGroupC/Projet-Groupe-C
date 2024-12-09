import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


def display_data_exploration():
    st.session_state['last_page'] = "data_exploration"
    data_handler = st.session_state["data_handler"]
    commodities_data_df_return = data_handler.get_commodities_returns()

    current_results = st.session_state["current_results"]
    expected_returns_annualized = data_handler.get_annualized_returns()
    volatility_annualized = current_results.get_result("volatility_annualized")
    corr_matrix = current_results.get_result("corr_matrix")

    st.header("🔍 Data Analysis and Exploration")

    st.info(""" 
    This section provides an in-depth analysis of commodity performance.  
    We examine the expected returns, volatility, and correlations to uncover key insights  
    about each commodity and how they interact, laying the groundwork for portfolio optimization. 
    """)

    # Expected Annualized Returns
    st.subheader("Expected Annualized Returns (mu)")

    st.write(""" 
    Expected annualized returns give a measure of how much return each commodity is expected  
    to deliver annually, based on historical data. This helps identify strong performers over time. 
    """)

    # Colors matching the donut chart
    st.bar_chart(expected_returns_annualized)

    # Annualized Volatility
    st.subheader("Annualized Volatility (sigma)")

    st.write(""" 
    Volatility represents the level of risk or uncertainty in a commodity's returns.  
    Higher volatility indicates greater price fluctuations, often associated with higher risk. 
    """)

    # Customized bar chart for volatility
    st.bar_chart(volatility_annualized)
    # Summary Table
    st.subheader("Summary: Returns and Volatility")
    st.write(""" 
    Below is a summary of the expected annualized returns and volatility for each commodity.  
    This table helps in assessing the trade-off between potential returns and associated risks. 
    """)

    summary_df = pd.DataFrame(
        {
            "Commodity": commodities_data_df_return.columns,
            "Expected Annualized Return (mu)": expected_returns_annualized.values,
            "Annualized Volatility (sigma)": volatility_annualized.values,
        }
    )
    summary_df.set_index("Commodity", inplace=True)
    st.dataframe(
        summary_df.style.format(
            {
                "Expected Annualized Return (mu)": "{:.2%}",
                "Annualized Volatility (sigma)": "{:.2%}",
            }
        )
    )

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    st.write(""" 
    The correlation matrix shows the relationships between commodities, revealing how their returns move  
    relative to each other. A value closer to 1 indicates strong positive correlation, while a value closer  
    to -1 suggests a strong negative correlation. Diversification is key when managing correlated assets. 
    """)

    corr_matrix = commodities_data_df_return.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax,
        cbar_kws={"label": "Correlation Coefficient"},
    )
    ax.set_title("Correlation Matrix of Commodities")
    st.pyplot(fig)
