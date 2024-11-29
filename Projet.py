import pandas as pd
import streamlit as st
import time
import plotly.graph_objects as go
import numpy as np
import yfinance as yf
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint

# Ajouter une nouvelle option pour l'introduction
import streamlit as st
from streamlit_option_menu import option_menu

# Customize the sidebar layout
with st.sidebar:
    # Adding an option menu with icons for better navigation
    choice = option_menu(
        "Portfolio Optimization",  # Sidebar title
        ["Introduction", "Data Exploration", "Efficient Frontier","Risk Aversion Questionnaire",
         "Minimum Variance Startegy without Rolling Window", "Performance of Minimun Variance Strategy with Rolling Window", "Performance of Black-litterman",
         "Comparison : Minimum Variance vs Blacklitterman", "Methodology"],  # Options list
        icons=['house', 'bar-chart', 'graph-up', 'book','calculator', 'search', 'search', 'graph-up', 'book'],
        # Icons for each option
        menu_icon="cast",  # Sidebar menu icon
        default_index=0,  # Default selected index
        orientation="vertical",  # Vertical menu style
        styles={
            "container": {
                "padding": "5px",
                "background-color": "#white"  # Light background color for sidebar
            },
            "icon": {
                "color": "#black",  # Dark blue icon color
                "font-size": "18px"
            },
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "padding": "10px",
                "color": "#black"  # Dark blue for text
            },
            "nav-link-selected": {
                "background-color": "#4e73df",  # Blue background for selected item
                "color": "white",  # White text when selected
            },
        }
    )

if choice == "Introduction":
    # Title and Subtitle
    st.title("Commodity Portfolio Optimization Dashboard")
    st.subheader("Harness Data-Driven Insights to Empower Your Investment Strategies")

    # Add the banner image using the provided URL
    st.image(
        "https://think.ing.com/uploads/hero/_w800h450/Commodities_montage_.PNG",
        use_column_width=True,
        caption="Optimize your portfolio with advanced insights."
    )

    # Introduction Text
    st.write(""" 
    Welcome to a cutting-edge platform designed to revolutionize how you approach commodity investments. 
    This dashboard integrates advanced financial models with real-time data, providing you with an 
    interactive experience to optimize your portfolio based on both market data and investor sentiment. 
    """)

    st.divider()

    # Key Features Section
    st.header("Key Features")

    st.subheader("🔍 Data Exploration")
    st.write(""" 
    - Visualize expected annualized returns and volatility of various commodities. 
    - Analyze correlation matrices to identify diversification opportunities. 
    """)

    st.subheader("📈 Efficient Frontier Analysis")
    st.write(""" 
    - Adjust the risk-free rate to see its impact on optimal portfolios. 
    - Discover the tangency portfolio to maximize your Sharpe ratio. 
    """)

    st.subheader("🎯 Risk Aversion Questionnaire")
    st.write(""" 
            - Assess your risk tolerance based on your preferences.
        - Answer two questions to discover your risk aversion score.
            """)

    st.subheader("🛠️ Minimum Variance Portfolio Optimization")
    st.write(""" 
    - Customize your portfolio according to your risk tolerance. 
    - Obtain optimal asset weights and visualize them with interactive charts. 
    """)

    st.subheader("📊 Performance Analysis with Rolling Window")
    st.write(""" 
    - Track portfolio performance over time using rolling windows. 
    - Compare your portfolio against major benchmarks like the S&P 500. 
    """)

    st.subheader("🧠 Black-Litterman Model Implementation")
    st.write(""" 
    - Integrate market equilibrium with your own views and sentiment data. 
    - Optimize your portfolio using the advanced Black-Litterman approach. 
    """)
    st.subheader("📊 Comparison: Minimum Variance vs Black-Litterman")
    st.write(""" 
            - Explore how risk minimization contrasts with incorporating market views. 
            - Learn which method aligns best with your investment strategy. 
            """)

    st.subheader("📚 Comprehensive Methodology Overview")
    st.write(""" 
    - Dive deep into the data sourcing, preprocessing, and modeling techniques. 
    - Understand the statistical foundations behind each optimization method. 
    """)



    st.divider()

    # Why Use This Dashboard Section
    st.header("🌟 Why Use This Dashboard?")

    st.write(""" 
    *Interactive Learning Experience*   
    Engage with financial models in a hands-on manner to deepen your understanding of portfolio optimization and market dynamics. 

    *Customized Insights*   
    Adjust key parameters to see real-time impacts on your portfolio, allowing for personalized investment strategies. 

    *Professional Visualizations*   
    Benefit from high-quality charts and graphs that make complex data easily interpretable. 
    """)

    st.divider()

    # Get Started Section
    st.header("🚀 Get Started")

    st.write(""" 
    Use the sidebar to navigate through the dashboard. Each section is designed to build upon the previous, 
    guiding you through a comprehensive analysis journey. 

    Feel free to adjust sliders, interact with charts, and explore different scenarios to see how changes in 
    inputs affect your portfolio outcomes. 
    """)

    st.divider()

    # Footer
    st.write("### 🌍 Embark on a Data-Driven Journey")
    st.write(""" 
    Optimize your investment strategies with confidence. Whether you're an experienced investor,
    a finance professional, or a student eager to learn, this dashboard is your gateway to informed decision-making. 
    """)


# Load data
@st.cache_data
def load_data():
    sentiment_df = pd.read_csv(
        'ressources/sentiment.csv')
    commodities_data_df = pd.read_csv(
        'ressources/cleaned_commodities_data.csv')

    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'].str.replace('.', '-', regex=False), errors='coerce')
    commodities_data_df['Date'] = pd.to_datetime(commodities_data_df['Date'], errors='coerce')

    sentiment_df['Date'] = sentiment_df['Date'].dt.tz_localize(None)
    commodities_data_df['Date'] = commodities_data_df['Date'].dt.tz_localize(None)

    return sentiment_df, commodities_data_df


sentiment_df, commodities_data_df = load_data()

# Prepare data
commodities_data_df_return = commodities_data_df.set_index('Date').pct_change()
commodities_data_df_return.columns = commodities_data_df_return.columns.str.replace("Price_", "")
covariance_matrix = commodities_data_df_return.cov() * 252  # Annualized covariance


# Define optimization functions
def portfolio_variance(weights, covariance_matrix):
    return weights.T @ covariance_matrix @ weights


def optimal_portfolio_Markowitz(cov_matrix):
    n_assets = len(cov_matrix)
    initial_weights = np.ones(n_assets) / n_assets
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))

    result = minimize(portfolio_variance, initial_weights,
                      args=(cov_matrix,), method='SLSQP',
                      bounds=bounds, constraints=constraints)

    if not result.success:
        st.error(f"Optimization failed: {result.message}")

    return result.x


if choice == "Methodology":
    st.title("Methodology")

    st.subheader("Commodity Price Data Acquisition")
    st.write(""" 
        The commodity price data utilized in this project was sourced from **Investing.com**, a reputable and widely-used platform for accessing historical financial and commodity market data. The commodities selected for this analysis encompass a diverse range of market sectors to provide a comprehensive understanding of commodity dynamics: 

        1. **Precious Metals**: 
            - **Gold**: Traditionally considered a safe-haven asset and a hedge against inflation and economic uncertainty. 
            - **Copper**: An essential industrial metal, often regarded as a barometer for global economic activity due to its widespread use in manufacturing and construction. 
            - **Aluminium**: Another critical industrial metal, integral to various industries including transportation, construction, and packaging. 

        2. **Energy Commodities**: 
            - **Oil**: A cornerstone of the global energy market, influencing a wide array of economic sectors and geopolitical strategies. 
            - **Natural Gas**: A key energy source for heating and electricity, with demand often influenced by seasonal patterns and economic conditions. 

        3. **Agricultural Commodities**: 
            - **Wheat**: A staple food commodity, essential for global food security and affected by agricultural policies and climate conditions. 
            - **Sugar**: A widely consumed commodity, reflective of agricultural production trends, consumption patterns, and biofuel policies. 
    """)

    st.subheader("Data Specifications")
    st.write(""" 
        The dataset encompasses daily closing prices for each commodity, covering a substantial period from **October 1, 2014, to October 31, 2024**. This 10-year timeframe captures significant market cycles, economic events, and price fluctuations, providing a robust foundation for analysis. 

        For each commodity, the dataset includes: 

        - **Daily Closing Prices**: Reflecting the end-of-day market valuation for each commodity, providing granular insight into price movements. 
        - **Consistent Timeframe**: Ensuring an overlapping and synchronized time range across all commodities to facilitate comparative analysis and integration with sentiment data. 

        The data extraction process involved meticulous steps: 

        - **Data Download**: Historical price data was downloaded in CSV format directly from Investing.com. 
        - **Data Preprocessing**: Rigorous preprocessing steps were undertaken, including data cleaning, formatting, and verification to ensure accuracy and readiness for analysis. 
    """)

    st.subheader("Return Calculation")
    st.latex(r"R_t = \frac{P_t - P_{t-1}}{P_{t-1}}")
    st.write(""" 
        The above formula was employed to calculate the **daily return** of each commodity, providing a measure of the relative change in price from one day to the next. 

        Where: 

        - **$R_t$**: Return on day $t$. 
        - **$P_t$**: Closing price on day $t$. 
        - **$P_{t-1}$**: Closing price on the previous trading day $t-1$. 

        This calculation captures the percentage change in price, which is essential for understanding the volatility and performance of each commodity over time. 
    """)

    st.subheader("Weekly Return Calculation")
    st.latex(r"R_{\text{weekly}} = \prod_{t=1}^{n} (1 + R_t) - 1")
    st.write(r""" 
        The **weekly return** for each commodity was calculated by aggregating the daily returns over a sentiment-defined time interval, typically a week. 

        Where: 

        - **$R_{\text{weekly}}$**: Cumulative return over the week. 
        - **$R_t$**: Daily return for day $t$. 
        - **$n$**: Number of trading days within the week. 

        This approach accounts for the compounding effect of daily returns over the week and aligns the return calculations with the frequency of sentiment data. 
    """)

    st.subheader("Data Preprocessing and Alignment")
    st.write(""" 
        1. **Handling Missing Data**: 

            - **Forward-Filling**: Missing price data, often due to non-trading days such as weekends and holidays, were addressed using forward-filling methods to carry forward the last known price. 
            - **Outlier Detection**: Statistical methods were employed to detect anomalies and extreme outliers, ensuring that the data used for analysis was reliable and representative. 

        2. **Alignment with Sentiment Data**: 

            - **Temporal Alignment**: Weekly returns were carefully aligned with the corresponding sentiment data to ensure that the analysis accurately reflected the relationship between sentiment and commodity returns. 
            - **Data Synchronization**: The datasets were synchronized based on dates to facilitate seamless integration in subsequent regression analyses. 

        This meticulous preprocessing ensures that the dataset is clean, accurate, and properly aligned for meaningful analysis. 
    """)

    st.write(""" 
        The resulting daily and weekly return datasets form the foundation for the regression modeling and portfolio optimization processes, enabling a detailed examination of how investor sentiment influences commodity price dynamics. 
    """)

    st.subheader("Sentiment Data Acquisition and Transformation")
    st.write(""" 
        The sentiment data was sourced from the **American Association of Individual Investors (AAII) Sentiment Survey**, a respected weekly survey that has been conducted since 1987. This survey captures individual investors' expectations regarding the stock market's direction over the next six months. Participants indicate whether they are: 

        - **Bullish**: Expecting market prices to increase. 
        - **Neutral**: Anticipating little to no change in market prices. 
        - **Bearish**: Expecting market prices to decrease. 

        The survey results are presented as percentages, reflecting the proportion of respondents in each category. This data provides valuable insights into market sentiment, which can influence commodity prices. More information about the survey can be found on the [AAII Sentiment Survey website](https://www.aaii.com/sentimentsurvey). 
    """)

    st.subheader("Data Transformation")
    st.write(""" 
        To quantitatively incorporate investor sentiment into our analysis, the categorical sentiment data was transformed into a continuous **polarity score** ranging from **-1** to **1**. This numerical representation allows for the integration of sentiment data with financial return metrics in the regression models. 

        The polarity score is calculated using the formula: 
    """)
    st.latex(r"\text{Polarity Score} = \frac{\text{Bullish \%} - \text{Bearish \%}}{100}")
    st.write(""" 
        Where: 

        - **Bullish %**: Percentage of respondents with a bullish outlook. 
        - **Bearish %**: Percentage of respondents with a bearish outlook. 

        This calculation results in a polarity score where: 

        - **-1**: Indicates an entirely bearish sentiment (100% bearish, 0% bullish). 
        - **0**: Reflects a neutral sentiment (equal percentages of bullish and bearish respondents). 
        - **1**: Represents an entirely bullish sentiment (100% bullish, 0% bearish). 

        By quantifying sentiment in this way, we can directly assess the impact of investor sentiment on commodity returns through regression analysis. 
    """)

    st.subheader("Regression Analysis")
    st.write(""" 
        To investigate the relationship between investor sentiment and commodity returns, we conducted a **simple linear regression** for each commodity. This analysis examines how changes in sentiment polarity scores affect the weekly returns of commodities. 

        The regression model is specified as: 
    """)
    st.latex(r"R_{\text{commodity}, t} = \alpha + \beta \cdot \text{Sentiment}_t + \epsilon_t")
    st.write(r""" 
        Where: 

        - **$R_{\text{commodity}, t}$**: Weekly return of the commodity at time $t$. 
        - **$\alpha$**: Intercept term, representing the expected return when sentiment is neutral. 
        - **$\beta$**: Coefficient representing the sensitivity of the commodity's return to changes in sentiment. 
        - **$\text{Sentiment}_t$**: Polarity score at time $t$. 
        - **$\epsilon_t$**: Error term capturing the variation not explained by the model. 

        From each regression, we extracted key metrics: 

        - **Beta ($\beta$)**: Indicates the direction and magnitude of the relationship between sentiment and returns. 
        - **Intercept ($\alpha$)**: Provides the baseline return when sentiment is neutral. 
        - **R-squared ($R^2$)**: Measures the proportion of variance in returns explained by sentiment. 
        - **P-value**: Assesses the statistical significance of the sentiment coefficient. 
    """)

    st.subheader("Regression Results")
    st.write(""" 
        The table below presents the regression results for each commodity: 
    """)

    data = {
        "Commodity": ["Aluminium", "Copper", "Gas", "Gold", "Oil", "Sugar", "Wheat"],
        "Coefficient": [0.0128, 0.0177, 0.0070, -0.0059, 0.0279, 0.0028, -0.000008],
        "Intercept": [0.00088, 0.00073, 0.00153, 0.00204, 0.00024, 0.00122, 0.00121],
        "R-squared": [0.00537, 0.00965, 0.00022, 0.00230, 0.00779, 0.00014, 0.0000009],
        "P-value": [0.0957, 0.0253, 0.7134, 0.2758, 0.0447, 0.7893, 0.9995],
    }
    regression_results = pd.DataFrame(data).set_index("Commodity")
    def custom_format(value):
        return f"{value:.8f}".rstrip('0').rstrip('.')


    # Apply custom formatting only for Wheat
    formatted_results = regression_results.copy()
    formatted_results.loc["Wheat", :] = formatted_results.loc["Wheat", :].apply(custom_format)
    st.dataframe(formatted_results)

    st.subheader("Interpretation of Results")
    st.write(r""" 
        1. **Significance of Sentiment Coefficient ($\beta$)**: 

            - **Copper**: Shows a statistically significant positive relationship with sentiment ($p = 0.0253$), indicating that increased bullish sentiment is associated with higher returns. 
            - **Oil**: Also exhibits a significant positive relationship ($p = 0.0447$). 
            - **Aluminium**: Approaches significance at the 10% level ($p = 0.0957$). 
            - **Other Commodities**: Gas, gold, sugar, and wheat do not show significant relationships, as indicated by their high p-values. 

        2. **Direction of Impact**: 

            - **Positive Coefficients**: Most commodities have positive $\beta$ coefficients, suggesting that bullish sentiment tends to correlate with higher returns. 
            - **Negative Coefficients**: Gold and wheat have slightly negative coefficients, implying a minor inverse relationship with sentiment, which may reflect their roles as safe-haven assets. 

        3. **Goodness of Fit ($R^2$)**: 

            - The $R^2$ values are low for all commodities, indicating that sentiment explains a small portion of the variance in returns. 
            - This suggests that while sentiment has some impact, other factors play significant roles in determining commodity returns. 

        Overall, the regression analysis reveals that investor sentiment, as measured by the AAII Sentiment Survey, has a statistically significant impact on certain commodities, particularly **copper** and **oil**. However, the low $R^2$ values highlight the complexity of commodity markets and the influence of additional factors such as supply and demand dynamics, geopolitical events, and macroeconomic indicators. 
    """)

    st.subheader("Minimum Variance Portfolio Construction")
    st.write(""" 
        The **Minimum Variance Portfolio** aims to minimize the overall portfolio risk (variance) while adhering to certain constraints. This approach focuses solely on the risk characteristics of the assets, disregarding expected returns. 

        **Optimization Objective**: 

        The optimization problem is formulated as: 
    """)
    st.latex(r"\min_{\mathbf{w}} \ \sigma_p^2 = \mathbf{w}^T \Sigma \mathbf{w}")
    st.write(""" 
        Where: 

        - **$\sigma_p^2$**: Portfolio variance to be minimized. 
        - **$\mathbf{w}$**: Vector of asset weights in the portfolio. 
        - **$\Sigma$**: Covariance matrix of asset returns. 

        **Constraints**: 

        - **Full Investment Constraint**: 
    """)
    st.latex(r"\sum_{i=1}^n w_i = 1")
    st.write(""" 
        - **No Short-Selling Constraint**: 
    """)
    st.latex(r"w_i \geq 0 \quad \forall \ i")
    st.write(""" 
        These constraints ensure that all available capital is invested and that no asset is sold short, resulting in a realistic and practical portfolio. 

        **Rolling Window Implementation**: 

        To capture the dynamic nature of financial markets, we implemented a rolling window approach: 

        - **Dynamic Covariance Matrix**: The covariance matrix $\Sigma$ was recalculated periodically using the most recent data within the rolling window. 
        - **Portfolio Rebalancing**: Asset weights $\mathbf{w}$ were updated at each rolling period to reflect changes in asset correlations and volatilities. 

        This methodology allows the portfolio to adapt to changing market conditions, potentially improving risk-adjusted returns over time. 
    """)

    st.subheader("Allocation to the Risk-Free Asset Based on Risk Aversion")
    st.write(r""" 
        Incorporating the risk-free asset allows investors to adjust the overall risk profile of the portfolio according to their risk tolerance. The allocation between the risky portfolio and the risk-free asset is determined by the investor's **risk aversion coefficient ($\lambda$)**. 

        **Determining the Risky Asset Allocation ($w_{\text{risky}}$)**: 

        The proportion of the total portfolio allocated to the risky assets is calculated using the following formula: 
    """)
    st.latex(r"w_{\text{risky}} = \max\left(0, \min\left(1, \frac{\mu_p - r_f}{\lambda \cdot \sigma_p^2}\right)\right)")
    st.write(r""" 
        Where: 

        - **$\mu_p$**: Expected return of the risky portfolio. 
        - **$r_f$**: Risk-free rate of return. 
        - **$\lambda$**: Investor's risk aversion coefficient. 
        - **$\sigma_p^2$**: Variance of the risky portfolio. 

        **Steps**: 

        1. **Calculate $\mu_p$ and $\sigma_p^2$**: 

            - $\mu_p = \mathbf{w}^T \mathbf{E}$, where $\mathbf{E}$ is the vector of expected returns. 
            - $\sigma_p^2 = \mathbf{w}^T \Sigma \mathbf{w}$. 

        2. **Compute $w_{\text{risky}}$**: 

            - The formula determines the optimal allocation to risky assets based on the trade-off between expected return and risk, scaled by the investor's risk aversion. 

        3. **Adjust for Constraints**: 

            - The allocation is bounded between 0 and 1 to ensure practical feasibility (no borrowing or short-selling of the risk-free asset). 

        **Final Portfolio Composition**: 

        - **Risky Assets**: The weights $\mathbf{w}$ are scaled by $w_{\text{risky}}$. 
        - **Risk-Free Asset**: The remaining portion $w_{\text{risk-free}} = 1 - w_{\text{risky}}$ is allocated to the risk-free asset. 

        This approach results in a tailored portfolio that aligns with the investor's risk preferences while aiming to optimize returns. 
    """)

    st.subheader("Black-Litterman Model Implementation")
    st.write(r""" 
        The **Black-Litterman Model** is an advanced portfolio optimization technique that combines market equilibrium returns with investor views to produce a set of expected returns that reflect both sources of information. This model addresses some of the limitations of traditional mean-variance optimization by incorporating subjective views and reducing estimation errors. 

        **Step 1: Calculate Implied Equilibrium Returns ($\tilde{\mu}$)** 

        We started with an **equally weighted market portfolio** due to the challenges in estimating the market capitalization weights of commodities. The implied equilibrium returns are calculated using: 
    """)
    st.latex(
        r"\tilde{\mu} = r_f + \text{SR}(\mathbf{x_0} \mid r_f) \cdot \frac{\Sigma \mathbf{x_0}}{\sqrt{\mathbf{x_0}^T \Sigma \mathbf{x_0}}}")
    st.write(r""" 
        Where: 

        - **$\mathbf{x_0}$**: Weights of the market portfolio (equal weights). 
        - **$\text{SR}(\mathbf{x_0} \mid r_f)$**: Sharpe Ratio of the market portfolio. 
        - **$r_f$**: Risk-free rate. 
        - **$\Sigma$**: Covariance matrix. 

        **Rationale**: 

        - Using an equally weighted portfolio simplifies the estimation and avoids biases introduced by uncertain market capitalizations. 
        - The implied returns represent the market's consensus expectations under equilibrium conditions. 
    """)

    st.subheader("Step 2: Incorporate Investor Views ($Q$) and Confidence ($\Omega$)")
    st.write(r""" 
        **Investor Views ($Q$)**: 

        - Derived from the regression analysis, reflecting the expected returns based on investor sentiment. 
        - Expressed in the form $Q = \alpha + \beta \cdot \text{Sentiment}$, capturing the influence of sentiment on returns. 

        **Confidence Matrix ($\Omega$)**: 

        - Represents the uncertainty (inverse of confidence) associated with each view. 
        - Calculated using the p-values from the regression analysis, standardized to ensure consistency. 

        **View Matrix ($P$)**: 

        - A selector matrix that identifies which assets are affected by each view. 
        - In this case, each view corresponds to a single commodity, so $P$ is an identity matrix. 

        **Adjustments for Confidence**: 

        - Lower p-values (higher statistical significance) correspond to higher confidence and lower uncertainty in $\Omega$. 
        - This ensures that more reliable views have a greater impact on the revised expected returns. 
    """)

    st.subheader(r"Step 3: Compute Revised Expected Returns ($\bar{\mu}$)")
    st.write(""" 
        The revised expected returns are calculated using the Black-Litterman formula: 
    """)
    st.latex(r""" 
        \bar{\mu} = \tilde{\mu} + \tau \Sigma P^T \cdot \Big( P \tau \Sigma P^T + \Omega \Big)^{-1} \cdot \Big( Q - P \tilde{\mu} \Big) 
        """)
    st.write(r""" 
        Where: 

        - **$\tau$**: Scalar representing the uncertainty in the prior (often set to a small value). 
        - **$\tilde{\mu}$**: Implied equilibrium returns. 
        - **$P$**: View matrix. 
        - **$Q$**: Vector of investor views. 
        - **$\Omega$**: Confidence matrix. 

        **Interpretation**: 

        - The formula adjusts the equilibrium returns by blending them with the investor views, weighted by the confidence levels. 
        - The result is a set of expected returns that incorporate both market information and subjective insights. 
    """)

    st.subheader("Step 4: Optimize the Portfolio")
    st.write(r""" 
        With the revised expected returns ($\bar{\mu}$) and the covariance matrix ($\Sigma$), we performed a **mean-variance optimization**: 
    """)
    st.latex(r"\max_{\mathbf{w}} \ \mathbf{w}^T \bar{\mu} - \frac{\gamma}{2} \mathbf{w}^T \Sigma \mathbf{w}")
    st.write(""" 
        Where: 

        - **$\gamma$**: Risk aversion parameter. 
        - **$\mathbf{w}$**: Asset weights to be determined. 

        **Constraints**: The constraints are identical to the ones found under our minimum variance portfolio. 

        **Methodology**: 

        - The optimization was solved using the **Sequential Least Squares Programming (SLSQP)** algorithm. 
        - A **rolling window approach** was adopted to update the portfolio periodically, allowing it to adapt to new information. 

        **Integration with Risk-Free Asset**: 

        - Similar to the minimum variance portfolio, we combined the optimized risky portfolio with the risk-free asset based on the investor's risk aversion. 
        - This step ensures that the final portfolio aligns with the investor's overall risk-return preferences. 
    """)

    st.write(""" 
        **Comparison with Benchmark**: 

        - To evaluate the performance of the optimized portfolios, we compared their cumulative returns against the **S&P 500 Index** over the same period. 
        - Data for the S&P 500 was obtained using the **yfinance** library, ensuring consistency and reliability. 

        **Conclusion**: 

        The implementation of the Black-Litterman model, alongside the minimum variance portfolio, provides valuable insights into portfolio optimization in the context of commodity investments influenced by investor sentiment. By integrating market data with investor views, we achieve a balanced and dynamic portfolio that can potentially enhance returns while managing risk. The comparative analysis against the S&P 500 offers a benchmark to assess the effectiveness of our strategies. This comprehensive approach underscores the importance of combining quantitative methods with qualitative insights to navigate the complexities of financial markets. 
    """)

# Data Exploration
if choice == "Data Exploration":
    st.header("🔍 Data Analysis and Exploration")

    st.info(""" 
    This section provides an in-depth analysis of commodity performance.  
    We examine the expected returns, volatility, and correlations to uncover key insights  
    about each commodity and how they interact, laying the groundwork for portfolio optimization. 
    """)

    # Expected Annualized Returns
    st.subheader("Expected Annualized Returns (mu)")
    expected_returns_annualized = commodities_data_df_return.mean() * 252

    st.write(""" 
    Expected annualized returns give a measure of how much return each commodity is expected  
    to deliver annually, based on historical data. This helps identify strong performers over time. 
    """)


    # Colors matching the donut chart
    st.bar_chart(expected_returns_annualized)

    # Annualized Volatility
    st.subheader("Annualized Volatility (sigma)")
    volatility_annualized = commodities_data_df_return.std() * np.sqrt(252)

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

    summary_df = pd.DataFrame({
        "Commodity": commodities_data_df_return.columns,
        "Expected Annualized Return (mu)": expected_returns_annualized.values,
        "Annualized Volatility (sigma)": volatility_annualized.values
    })
    summary_df.set_index("Commodity", inplace=True)
    st.dataframe(summary_df.style.format({"Expected Annualized Return (mu)": "{:.2%}",
                                          "Annualized Volatility (sigma)": "{:.2%}"}))

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    st.write(""" 
    The correlation matrix shows the relationships between commodities, revealing how their prices move  
    relative to each other. A value closer to 1 indicates strong positive correlation, while a value closer  
    to -1 suggests a strong negative correlation. Diversification is key when managing correlated assets. 
    """)

    corr_matrix = commodities_data_df_return.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                cbar_kws={'label': 'Correlation Coefficient'})
    ax.set_title("Correlation Matrix of Commodities")
    st.pyplot(fig)

# Efficient Frontier
if choice == "Efficient Frontier":
    st.header("Efficient Frontier with Adjustable Risk-Free Rate")
    st.info(
        "This page calculates the efficient frontier, treating all commodities as risky assets. You can adjust the risk-free rate dynamically.")

    # Adjustable Risk-Free Rate
    risk_free_return = st.slider("Adjust Risk-Free Rate (Risk-Free Asset)", 0.0, 0.07, 0.04, 0.01)
    st.write(f"Selected Risk-Free Rate: {risk_free_return:.2%}")

    # Annualized Returns and Volatility
    expected_returns = commodities_data_df_return.mean() * 252
    cov_matrix = commodities_data_df_return.cov() * 252
    mu = expected_returns

    # Calculate Efficient Frontier
    mu_efficient_frontier = []
    vol_efficient_frontier = []
    gammas = np.linspace(0.01, 3, 50)

    for gam in gammas:
        def objective(weights):
            return 0.5 * np.dot(weights.T, np.dot(cov_matrix, weights)) - gam * np.dot(weights, mu)


        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(mu)))

        res = minimize(objective, np.ones(len(mu)) / len(mu), method='SLSQP', bounds=bounds, constraints=cons)

        if res.success:
            weights = res.x
            mu_portfolio = np.dot(weights, mu)
            vol_portfolio = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            mu_efficient_frontier.append(mu_portfolio)
            vol_efficient_frontier.append(vol_portfolio)

    # Tangency Portfolio Calculation
    weights_tangency = (np.linalg.inv(cov_matrix) @ (mu - risk_free_return)) / (
            np.ones(cov_matrix.shape[0]) @ np.linalg.inv(cov_matrix) @ (mu - risk_free_return)
    )
    weights_tangency = np.clip(weights_tangency, 0, 1)
    weights_tangency /= weights_tangency.sum()

    mu_tangency = np.dot(weights_tangency, mu)
    vol_tangency = np.sqrt(np.dot(weights_tangency.T, np.dot(cov_matrix, weights_tangency)))

    # Prepare Chart Data
    chart_data = pd.DataFrame({
        "vol": vol_efficient_frontier,
        "mu": mu_efficient_frontier,
        "color": "#0000FF",  # Blue for Efficient Frontier
        "size": 8
    })

    # Add Risk-Free Point
    chart_data = pd.concat([
        chart_data,
        pd.DataFrame({"vol": [0], "mu": [risk_free_return], "color": ["#FF0000"], "size": [10]})  # Red for Risk-Free
    ], ignore_index=True)

    # Add Tangency Portfolio Point
    chart_data = pd.concat([
        chart_data,
        pd.DataFrame({"vol": [vol_tangency], "mu": [mu_tangency], "color": ["#FFA500"], "size": [10]})
        # Orange for Tangency Portfolio
    ], ignore_index=True)

    # Display Efficient Frontier
    st.scatter_chart(chart_data, x="vol", y="mu", color="color", size="size")

    # Add Legends
    st.write("Legend:")
    st.markdown("- Blue points: Efficient Frontier portfolios (optimized for different risk aversions)")
    st.markdown("- Red point: Risk-Free Rate")
    st.markdown("- Orange point: Tangency Portfolio (maximum Sharpe ratio)")

    # Display Tangency Portfolio Weights
    st.subheader("Tangency Portfolio Weights")
    tangency_weights_df = pd.DataFrame({
        "Asset": commodities_data_df_return.columns,
        "Weight": weights_tangency
    }).sort_values(by="Weight", ascending=False)
    st.write(tangency_weights_df)

    st.write(f"Expected Return of Tangency Portfolio: {mu_tangency:.2%}")
    st.write(f"Volatility of Tangency Portfolio: {vol_tangency:.2%}")

# Ajouter des variables globales pour éviter les problèmes de portée entre sections
final_combined_weights = None
expected_returns = None
covariance_matrix = None

# Calculer les données nécessaires globalement
expected_returns = commodities_data_df_return.mean() * 252  # Annualized expected returns
covariance_matrix = commodities_data_df_return.cov() * 252  # Annualized covariance matrix

if choice == "Minimum Variance Startegy without Rolling Window":
    st.header("Optimal Portfolio Weights with Adjustable Risk-Free Rate")

    st.info(
        "Use the sliders below to adjust the risk-free rate and risk aversion, and observe how the portfolio weights and metrics update dynamically."
    )

    # Interactive parameters
    risk_free_rate = st.slider("Adjust Risk-Free Rate", 0.0, 0.09, 0.04, 0.01)
    risk_aversion = st.slider("Select Risk Aversion", 1.00, 5.0, 3.0, 1.0)
    st.session_state['risk_free_rate'] = risk_free_rate
    st.session_state['risk_aversion'] = risk_aversion

    # Calculate annualized returns and covariance matrix
    annualized_returns = commodities_data_df_return.mean() * 252
    covariance_matrix = commodities_data_df_return.cov() * 252

    # Step 1: Compute optimal risky portfolio weights
    optimal_weights = optimal_portfolio_Markowitz(covariance_matrix)

    # Step 2: Compute risky portfolio return and variance
    portfolio_return = np.dot(optimal_weights, annualized_returns)
    portfolio_variance_value = portfolio_variance(optimal_weights, covariance_matrix)

    # Step 3: Proportion allocated to risky portfolio
    w_risky = (portfolio_return - risk_free_rate) / (risk_aversion * portfolio_variance_value)
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

    # Define colors for the assets
    color_map = {
        'gold': '#FFDDC1',  # Soft peach
        'oil': '#FFABAB',  # Light coral
        'gas': '#FFC3A0',  # Pastel orange
        'copper': '#D5AAFF',  # Lavender
        'aluminium': '#85E3FF',  # Light sky blue
        'wheat': '#FFFFB5',  # Soft yellow
        'sugar': '#FF9CEE',  # Light pink
        'Risk-Free Asset': '#B9FBC0'  # Mint green
    }

    # Filter weights below 0.1% and prepare data for the donut chart
    filtered_weights = [(name, weight) for name, weight in zip(asset_names_with_rf, final_combined_weights) if
                        weight >= 0.001]
    filtered_names, filtered_values = zip(*filtered_weights) if filtered_weights else ([], [])
    filtered_colors = [color_map[name] for name in filtered_names]

    # Compute combined portfolio metrics
    combined_portfolio_variance = w_risky ** 2 * portfolio_variance_value
    combined_portfolio_return = (w_risky * portfolio_return) + (risk_free_weight * risk_free_rate)

    # Display weights and metrics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Optimal Portfolio Weights")
        weights_df = pd.DataFrame({
            "Asset": asset_names_with_rf,
            "Weight": final_combined_weights
        }).sort_values(by="Weight", ascending=False)
        st.dataframe(weights_df.set_index("Asset"))

    with col2:
        st.subheader("Portfolio Metrics")
        st.markdown(f"- **Proportion Allocated to Risky:** {w_risky:.2%}")
        st.markdown(f"- **Proportion Allocated to Risk-Free:** {risk_free_weight:.2%}")
        st.markdown(f"- **Expected Portfolio Return:** {combined_portfolio_return:.2%}")
        st.markdown(f"- **Portfolio Volatility:** {np.sqrt(combined_portfolio_variance):.2%}")

    # Donut chart for portfolio allocation
    st.subheader("Portfolio Allocation")

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        filtered_values,
        labels=filtered_names,
        autopct=lambda p: f'{p:.1f}%' if p > 0 else '',
        startangle=90,
        colors=filtered_colors,
        wedgeprops=dict(width=0.3),  # Makes it a donut chart
        textprops=dict(color="black")  # Text color for better readability
    )
    ax.axis("equal")  # Equal aspect ratio ensures the donut is circular
    ax.set_title("Portfolio Allocation", fontsize=16)
    st.pyplot(fig)

    # Save final weights to session state for later use
    st.session_state["final_combined_weights"] = final_combined_weights
    st.session_state["expected_returns"] = annualized_returns

if choice == "Performance of Minimun Variance Strategy with Rolling Window":

    st.header("Performance Analysis with Rolling Window")
    st.info("Analyze portfolio performance over time using a rolling window approach and compare it with the S&P 500.")

    # Variables partagées
    try:
        # Variables shared via session state
        final_combined_weights = st.session_state["final_combined_weights"]
        expected_returns = st.session_state["expected_returns"]
        risk_aversion = st.session_state['risk_aversion']
        risk_free_rate = st.session_state['risk_free_rate']
    except KeyError as e:
        st.success("Make sure you have completed the previous steps named : Weights with only Min Var before analyzing the performance.")

        # Logic for Performance Analysis...

        # Ensure final results are saved in session state
    # Paramètres interactifs
    risk_free_rate = st.slider("Adjust Risk-Free Rate ", 0.0, 0.09, 0.04, 0.01)
    risk_aversion = st.slider("Select Risk Aversion ", 1.00, 5.0, 3.0, 1.0)
    st.session_state['risk_free_rate'] = risk_free_rate
    st.session_state['risk_aversion'] = risk_aversion

    rolling_window = 20  # Taille de la fenêtre mobile
    initial_weights = final_combined_weights[:-1]

    optimal_weights_list = []

    for i in range(rolling_window, len(commodities_data_df_return)):
        # Get the rolling returns window
        window_returns = commodities_data_df_return.iloc[i - rolling_window:i]

        # Calculate the covariance matrix for the rolling window
        covariance_matrix = window_returns.cov().values * 252  # Annualized covariance matrix

        # Use the previous weights as the starting point for optimization
        n_assets = len(covariance_matrix)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum of weights = 1
        bounds = [(0, 1)] * n_assets  # Long-only constraints

        # Optimize portfolio weights
        result = minimize(portfolio_variance,
                          initial_weights,
                          args=(covariance_matrix,),
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        # Store the optimized weights for this window
        optimal_weights = result.x
        optimal_weights_list.append(optimal_weights)

        # Update initial_weights for the next iteration
        initial_weights = optimal_weights

    # Convert list to DataFrame for easier handling
    optimal_weights_df = pd.DataFrame(optimal_weights_list,
                                      index=commodities_data_df_return.index[rolling_window:],
                                      columns=commodities_data_df_return.columns)

    optimal_weights_df.columns = optimal_weights_df.columns.str.replace('Price_', '', regex=False)

    portfolio_with_risk_free = []

    # Loop through the existing optimal weights DataFrame
    for index, row in optimal_weights_df.iterrows():
        # Extract the optimal weights for the risky assets
        optimal_weights = row.values

        # Calculate portfolio return and variance using the optimal weights
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_variance_value = np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights))

        # Calculate w_risky
        w_risky = max(0, min(1, (portfolio_return - risk_free_rate) / (risk_aversion * portfolio_variance_value)))

        # Calculate combined weights for risky assets and risk-free weight
        combined_portfolio_weights = optimal_weights * w_risky
        risk_free_weight = 1 - w_risky

        # Append combined weights with the risk-free weight
        portfolio_with_risk_free.append(np.append(combined_portfolio_weights, risk_free_weight))

    # Create a new DataFrame for the portfolio with risk-free weights
    columns = list(optimal_weights_df.columns) + ['Risk-Free Asset']
    portfolio_with_risk_free_df = pd.DataFrame(portfolio_with_risk_free, index=optimal_weights_df.index,
                                               columns=columns)

    # Align daily returns and weights
    aligned_daily_returns = commodities_data_df_return.loc[portfolio_with_risk_free_df.index]  # Align indices
    weights = portfolio_with_risk_free_df.iloc[:, :-1]  # Exclude risk-free asset column
    risk_free_weight = portfolio_with_risk_free_df['Risk-Free Asset']
    risk_free_returns = risk_free_rate / 252  # Daily risk-free return

    # Calculate daily portfolio returns
    portfolio_daily_returns = (weights.values * aligned_daily_returns.values).sum(
        axis=1) + risk_free_weight.values * risk_free_returns

    # Calculate cumulative return
    cumulative_return = (1 + portfolio_daily_returns).cumprod()

    # Create a DataFrame for daily returns with dates
    portfolio_daily_returns_df = pd.DataFrame({
        'Date': portfolio_with_risk_free_df.index,
        'Daily Return': portfolio_daily_returns
    })
    portfolio_daily_returns_df['Year'] = portfolio_daily_returns_df['Date'].dt.year

    # Group by year and calculate annualized return per year
    annualized_returns_per_year = portfolio_daily_returns_df.groupby('Year').apply(
        lambda x: (1 + x['Daily Return'].mean()) ** 252 - 1
    )
    st.session_state['annualized_returns_per_year'] = annualized_returns_per_year
    # Fetch S&P 500 data
    sp500_data = yf.download('^GSPC',
                             start=portfolio_with_risk_free_df.index[0],
                             end=portfolio_with_risk_free_df.index[-1])

    # Calculate S&P 500 daily and cumulative returns
    sp500_data['Return'] = sp500_data['Adj Close'].pct_change()
    sp500_data.dropna(inplace=True)
    sp500_data['Cumulative_Return'] = (1 + sp500_data['Return']).cumprod()

    # Calculate S&P 500 annualized return per year
    sp500_data['Year'] = sp500_data.index.year
    sp500_annualized_returns_per_year = sp500_data.groupby('Year')['Return'].apply(
        lambda x: (1 + x.mean()) ** 252 - 1
    )
    st.session_state['cumulative_return'] = cumulative_return
    # === Graph 1: Portfolio Cumulative Returns ===
    st.subheader("1. Portfolio Cumulative Returns")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(portfolio_with_risk_free_df.index, cumulative_return, label="Portfolio Cumulative Return", color="blue")
    ax3.set_title("Portfolio Cumulative Returns")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Cumulative Return")
    ax3.legend()
    ax3.grid()
    st.pyplot(fig3)

    # === Graph 2: Pie Chart for Portfolio Weights at a Specific Date ===
    st.subheader("2. Portfolio Allocation on Selected Date")

    # Assuming portfolio_with_risk_free_df is already loaded and contains pre-computed weights
    portfolio_with_risk_free_df.index = pd.to_datetime(portfolio_with_risk_free_df.index)

    # Resample the data to quarterly frequency for fewer slider options
    # Ensure that the latest date (2024-10-31) is the last date in the slider
    quarterly_data = portfolio_with_risk_free_df.resample('Q').last()
    latest_date = pd.Timestamp("2024-10-31")
    if latest_date not in quarterly_data.index:
        # Include the latest date manually if it's missing from the resampled data
        quarterly_data.loc[latest_date] = portfolio_with_risk_free_df.loc[latest_date]

    # Sort the data to maintain chronological order
    quarterly_data = quarterly_data.sort_index()

    # Define colors for the assets
    color_map = {
        'gold': '#FFDDC1',
        'oil': '#FFABAB',
        'gas': '#FFC3A0',
        'copper': '#D5AAFF',
        'aluminium': '#85E3FF',
        'wheat': '#FFFFB5',
        'sugar': '#FF9CEE',
        'Risk-Free Asset': '#B9FBC0'
    }


    # Filter function to remove values below 0.01%
    def filter_small_allocations(data):
        return data[data >= 0.0001]


    # Initialize the Plotly figure
    fig = go.Figure()

    # Create a list of frames for the slider (one per date)
    frames = []
    for date in quarterly_data.index:
        weights = filter_small_allocations(quarterly_data.loc[date].dropna())
        frames.append(go.Frame(
            data=[
                go.Pie(
                    labels=weights.index,
                    values=weights.values,
                    hole=0.4,  # Donut chart effect
                    marker=dict(colors=[color_map[asset] for asset in weights.index])
                )
            ],
            name=str(date.date())
        ))

    # Add the first frame (initial state) to the figure
    initial_date = quarterly_data.index[0]
    initial_weights = filter_small_allocations(quarterly_data.loc[initial_date].dropna())
    fig.add_trace(go.Pie(
        labels=initial_weights.index,
        values=initial_weights.values,
        hole=0.4,
        marker=dict(colors=[color_map[asset] for asset in initial_weights.index])
    ))

    # Set up the layout for the figure
    fig.update_layout(
        title="Portfolio Allocation Over Time (Quarterly Data, Filtered)",
        annotations=[dict(text='Portfolio', x=0.5, y=0.5, font_size=20, showarrow=False)],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Date: ", "xanchor": "center", "font": {"size": 14}},
                "pad": {"t": 20},  # Slight padding between chart and slider
                "len": 1.0,  # Slider spans the full figure width
                "x": 0,  # Align slider to the left
                "steps": [
                    {
                        "method": "animate",
                        "args": [[str(date.date())], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}],
                        "label": str(date.date())
                    } for date in quarterly_data.index
                ]
            }
        ]
    )

    # Add frames to the figure for the slider
    fig.frames = frames

    # Adjust layout for better alignment and a smaller chart size
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=50),  # Reduce margins to avoid excessive spacing
        height=400,  # Smaller chart height
        showlegend=True,
        template="plotly_white"
    )

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # === Graph 2: Annualized Returns ===
    st.subheader("3. Annualized Portfolio Returns by Year")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    annualized_returns_per_year.plot(kind='bar', ax=ax2, color='green', alpha=0.7, label='Portfolio')
    ax2.set_title("Annualized Returns by Year")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Annualized Return")
    ax2.legend()
    ax2.grid(axis='y')
    st.pyplot(fig2)

    # === Graph 4: Comparison of Cumulative Returns ===
    st.subheader("4. Portfolio vs S&P 500 Cumulative Returns")
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    ax4.plot(portfolio_with_risk_free_df.index, cumulative_return, label='Portfolio Cumulative Return', color='blue')
    ax4.plot(sp500_data.index, sp500_data['Cumulative_Return'], label='S&P 500 Cumulative Return', color='orange')
    ax4.set_title("Portfolio vs S&P 500 Cumulative Returns")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Cumulative Return")
    ax4.legend()
    ax4.grid()
    st.pyplot(fig4)
    st.session_state['sp500_data'] = sp500_data
    st.session_state['portfolio_with_risk_free_df'] = portfolio_with_risk_free_df




if choice == "Performance of Black-litterman":

    st.header("Black-litterman portfolio with Rolling Window")
    st.info(
        "Analyze Black-litterman portfolio performance over time using a rolling window approach and compare it with the S&P 500.")
    try:
        risk_free_rate = st.session_state['risk_free_rate']
        risk_aversion = st.session_state['risk_aversion']

    except KeyError as e:
        st.success("Make sure you have completed the 2 previous steps before analyzing the performance.")


    risk_free_rate = st.slider("Adjust Risk-Free Rate ", 0.0, 0.09, 0.04, 0.01)
    risk_aversion = st.slider("Select Risk Aversion ", 1.00, 5.0, 3.0, 1.0)
    sp500_data = st.session_state['sp500_data']
    all_data = commodities_data_df
    all_data['Date'] = pd.to_datetime(all_data['Date'])
    all_data.set_index('Date', inplace=True)

    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    sentiment_df.set_index('Date', inplace=True)

    daily_return = all_data.pct_change()
    daily_return = daily_return.dropna()
    weekly_returns = (1 + daily_return).resample('W-THU').prod() - 1
    aligned_data = pd.concat([weekly_returns, sentiment_df], axis=1, join="inner")

    X = aligned_data['Sentiment'].values.reshape(-1, 1)
    X = sm.add_constant(X)

    results_list = []

    for commodity in weekly_returns.columns:
        y = aligned_data[commodity].values

        model = sm.OLS(y, X).fit()

        results_list.append({
            'Commodity': commodity,
            'Beta': model.params[1],
            'Alpha': model.params[0],
            'R-squared': model.rsquared,
            'P-value': model.pvalues[1]
        })

    regression_results = pd.DataFrame(results_list)

    commodities = ['gold', 'oil', 'gas', 'copper', 'cobalt', 'wheat', 'sugar']
    P = np.eye(len(commodities))

    p_values = regression_results['P-value'].values / 100
    omega = np.diag(p_values)
    tau = 0.05


    def gamma_matrix(tau):
        return tau * covariance_matrix


    def QP(x, sigma, mu, gamma):

        v = 0.5 * x.T @ sigma @ x - gamma * x.T @ mu

        return v


    Q = pd.DataFrame(index=sentiment_df.index)

    for _, row in regression_results.iterrows():
        commodity = row['Commodity']
        Alpha = row['Alpha']
        Beta = row['Beta']

        Q[commodity] = Alpha + Beta * sentiment_df['Sentiment']

    num_commodities = 7

    equal_weights = np.ones(num_commodities) / num_commodities

    covariance_matrix = weekly_returns.cov()

    weeks_per_year = 52

    weekly_portfolio_return = weekly_returns.mean() @ equal_weights
    portfolio_annualized_return = (1 + weekly_portfolio_return) ** weeks_per_year - 1

    weekly_portfolio_volatility = np.sqrt(
        equal_weights @ covariance_matrix @ equal_weights)
    portfolio_annualized_volatility = weekly_portfolio_volatility * np.sqrt(weeks_per_year)

    sharpe_ratio = (portfolio_annualized_return - risk_free_rate) / portfolio_annualized_volatility

    SR_x0 = sharpe_ratio

    numerator = SR_x0 * (covariance_matrix @ equal_weights)
    denominator = np.sqrt(equal_weights @ covariance_matrix @ equal_weights)
    implied_mu = (risk_free_rate + numerator / denominator) / 52

    implied_mu_df = pd.DataFrame(
        [implied_mu],
        index=['Implied Mu'],
        columns=weekly_returns.columns
    )

    mu_bar_df = pd.DataFrame(index=Q.index, columns=implied_mu_df.columns)

    for date in Q.index:
        Q_week = Q.loc[date].values
        implied_mu = implied_mu_df.loc["Implied Mu"].values
        gamma = gamma_matrix(tau)
        adjustment_term = Q_week - (P @ implied_mu)
        mu_bar_week = implied_mu + (gamma @ P.T) @ np.linalg.inv(P @ gamma @ P.T + omega) @ adjustment_term
        mu_bar_df.loc[date] = mu_bar_week


    def compute_gamma(implied_phi):
        return 1 / implied_phi


    def QP(w, cov_matrix, mu_bar, gamma):

        return -1 * (mu_bar.T @ w - gamma * 0.5 * w.T @ cov_matrix @ w)


    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    num_assets = mu_bar_df.shape[1]
    bounds = [(0, 1) for _ in range(num_assets)]
    rolling_window_size = 4
    optimized_weights_dict = {}

    for i, (date, mu_bar) in enumerate(mu_bar_df.iterrows()):

        if i < rolling_window_size - 1:
            continue

        rolling_data = weekly_returns.iloc[i - rolling_window_size + 1:i + 1]
        cov_matrix = covariance_matrix * 52
        implied_phi = risk_aversion
        gamma = compute_gamma(implied_phi)
        x0 = np.full(num_assets, 1 / num_assets)
        res = minimize(
            QP,
            x0,
            args=(cov_matrix, mu_bar.values, gamma),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False}
        )

        if res.success:
            optimized_weights_dict[date] = res.x
        else:
            print(f"Optimization failed for {date}: {res.message}")

    optimized_weights_df = pd.DataFrame(optimized_weights_dict).T

    optimized_weights_df.columns = mu_bar_df.columns
    optimized_weights_df.index.name = "Date"

    portfolio_with_risk_free = []

    for index, row in optimized_weights_df.iterrows():
        optimal_weights = row.values
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_variance_value_bl = np.dot(optimal_weights.T, np.dot(covariance_matrix * 52, optimal_weights))
        w_risky = max(0, min(1, (portfolio_return - risk_free_rate) / (risk_aversion * portfolio_variance_value_bl)))
        combined_portfolio_weights = optimal_weights * w_risky
        risk_free_weight = 1 - w_risky
        portfolio_with_risk_free.append(np.append(combined_portfolio_weights, risk_free_weight))

    columns = list(optimized_weights_df.columns) + ['Risk-Free Asset']
    portfolio_with_risk_free_df_bl = pd.DataFrame(portfolio_with_risk_free, index=optimized_weights_df.index,
                                                  columns=columns)

    portfolio_returns = (portfolio_with_risk_free_df_bl.iloc[:, :-1] * aligned_data).sum(axis=1)
    risk_free_contribution = portfolio_with_risk_free_df_bl['Risk-Free Asset'] * (risk_free_rate / 52)
    total_portfolio_returns = portfolio_returns + risk_free_contribution
    cumulative_returns = (1 + total_portfolio_returns).cumprod()
    # Ensure the indices of both DataFrames are in datetime format
    weekly_returns.index = pd.to_datetime(weekly_returns.index)
    portfolio_with_risk_free_df_bl.index = pd.to_datetime(portfolio_with_risk_free_df_bl.index)

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
        yearly_returns = portfolio_weekly_returns[portfolio_weekly_returns.index.year == year]

        # Calculate the annualized return
        annualized_return = (1 + yearly_returns).prod() ** (52 / len(yearly_returns)) - 1
        annualized_returns_by_year[year] = annualized_return

    # Convert results into a DataFrame for better presentation
    annualized_returns_df = pd.DataFrame.from_dict(annualized_returns_by_year, orient="index",
                                                   columns=["Annualized Return"])

    st.subheader("1.Portfolio Cumulative Returns")

    # Create a single plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot S&P 500 cumulative returns

    ax.plot(cumulative_returns, label='Black-litterman portfolio', color='green')
    # Set titles and labels
    ax.set_title("Black-litterman portfolio ")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Returns")
    ax.legend()
    ax.grid()

    # Display the plot in Streamlit
    st.pyplot(fig)

    # === Graph 2: Pie Chart for Portfolio Weights at a Specific Date ===
    st.subheader("2. Portfolio Allocation on Selected Date")

    # Assuming portfolio_with_risk_free_df is already loaded and contains pre-computed weights
    portfolio_with_risk_free_df_bl.index = pd.to_datetime(portfolio_with_risk_free_df_bl.index)

    # Resample the data to quarterly frequency for fewer slider options
    quarterly_data = portfolio_with_risk_free_df_bl.resample('Q').last()

    # Clean column names by removing "Price_" prefix
    quarterly_data.columns = [col.replace("Price_", "") for col in quarterly_data.columns]

    # Define colors for the assets
    color_map = {
        'gold': '#FFDDC1',  # Soft peach
        'oil': '#FFABAB',  # Light coral
        'gas': '#FFC3A0',  # Pastel orange
        'copper': '#D5AAFF',  # Lavender
        'aluminium': '#85E3FF',  # Light sky blue
        'wheat': '#FFFFB5',  # Mint green
        'sugar': '#FF9CEE',  # Light pink
        'Risk-Free Asset': '#B9FBC0'  # Soft yellow
    }


    # Filter function to remove values below 0.01%
    def filter_small_allocations(data):
        return data[data >= 0.0001]


    # Initialize the Plotly figure
    fig = go.Figure()

    # Create a list of frames for the slider (one per date)
    frames = []
    for date in quarterly_data.index:
        weights = filter_small_allocations(quarterly_data.loc[date])
        frames.append(go.Frame(
            data=[
                go.Pie(
                    labels=weights.index,
                    values=weights.values,
                    hole=0.4,  # Donut chart effect
                    marker=dict(colors=[color_map[asset] for asset in weights.index])
                )
            ],
            name=str(date.date())
        ))

    # Add the first frame (initial state) to the figure
    initial_date = quarterly_data.index[0]
    initial_weights = filter_small_allocations(quarterly_data.loc[initial_date])
    fig.add_trace(go.Pie(
        labels=initial_weights.index,
        values=initial_weights.values,
        hole=0.4,
        marker=dict(colors=[color_map[asset] for asset in initial_weights.index])
    ))

    # Set up the layout for the figure
    fig.update_layout(
        title="Portfolio Allocation Over Time (Quarterly Data, Filtered)",
        annotations=[dict(text='Portfolio', x=0.5, y=0.5, font_size=20, showarrow=False)],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Date: ", "xanchor": "center", "font": {"size": 14}},
                "pad": {"t": 20},  # Slight padding between chart and slider
                "len": 1.0,  # Slider spans the full figure width
                "x": 0,  # Align slider to the left
                "steps": [
                    {
                        "method": "animate",
                        "args": [[str(date.date())], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}],
                        "label": str(date.date())
                    } for date in quarterly_data.index
                ]
            }
        ]
    )

    # Add frames to the figure for the slider
    fig.frames = frames

    # Adjust layout for better alignment and a smaller chart size
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=50),  # Reduce margins to avoid excessive spacing
        height=400,  # Smaller chart height
        showlegend=True,
        template="plotly_white"
    )

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # === Graph 3: Annualized Returns ===
    st.subheader("3. Annualized Portfolio Returns by Year")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    annualized_returns_df.plot(kind='bar', ax=ax2, color='blue', alpha=0.7, label='Portfolio')
    ax2.set_title("Annualized Returns by Year")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Annualized Return")
    ax2.legend()
    ax2.grid(axis='y')
    st.pyplot(fig2)

    st.subheader("4.Portfolio vs S&P 500 Cumulative Returns")

    # Create a single plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot S&P 500 cumulative returns
    ax.plot(sp500_data.index, sp500_data['Cumulative_Return'], label='S&P 500 ', color='orange')
    ax.plot(cumulative_returns, label='Black-litterman portfolio', color='green')
    # Set titles and labels
    ax.set_title("Black-litterman portfolio vs S&P 500")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Returns")
    ax.legend()
    ax.grid()

    # Display the plot in Streamlit
    st.pyplot(fig)
    st.session_state['cumulative_returns'] = cumulative_returns
    st.session_state['annualized_returns_df'] = annualized_returns_df

if choice == "Comparison : Minimum Variance vs Blacklitterman":
    st.title('Comparaison betweem Min Var and Black-litterman strategy')
    try:
        annualized_returns_per_year = st.session_state['annualized_returns_per_year']
        annualized_returns_df = st.session_state['annualized_returns_df']
    except KeyError as e:
        st.success("Make sure you have completed the 3 previous steps before analyzing the performance.")

    # === Graph 1: Annualized Returns Comparison ===
    st.subheader("1. Comparison of the Annualized Portfolio Returns by Year")

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    annualized_returns_per_year.plot(kind='bar', ax=ax1, color='green', alpha=0.7, label='Portfolio')
    ax1.set_title("Annualized Returns by Year of Min Var")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Annualized Return")
    ax1.legend()
    ax1.grid(axis='y')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    annualized_returns_df.plot(kind='bar', ax=ax2, color='blue', alpha=0.7, label='Portfolio')
    ax2.set_title("Annualized Returns by Year of Black-Litterman")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Annualized Return")
    ax2.legend()
    ax2.grid(axis='y')
    st.pyplot(fig2)

    # Graph 2
    st.subheader("2.Min Var vs Black-litterman Cumulative Returns")
    cumulative_returns = st.session_state['cumulative_returns']
    cumulative_return = st.session_state['cumulative_return']
    portfolio_with_risk_free_df = st.session_state['portfolio_with_risk_free_df']
    sp500_data = st.session_state['sp500_data']
    # Create a single plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sp500_data.index, sp500_data['Cumulative_Return'], label='S&P 500 ', color='orange')
    ax.plot(portfolio_with_risk_free_df.index, cumulative_return, label='Min Var portfolio', color='blue')
    ax.plot(cumulative_returns, label='Black-litterman portfolio', color='green')
    # Set titles and labels
    ax.set_title("Black-litterman portfolio vs S&P 500")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Returns")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

if choice == "Risk Aversion Questionnaire":
    st.header("Risk Aversion Questionnaire")

    st.write(
        """
        This questionnaire assesses your level of risk aversion. You will answer two questions, 
        each offering a choice between a lottery and a guaranteed payout. Your final risk aversion 
        score will range from *1 (highly risk-seeking)* to *5 (highly risk-averse)*.
        """
    )

    # Starting Risk Aversion Score
    risk_aversion_score = 3  # Neutral starting point

    # Question 1
    q1 = st.radio(
        "1. What would you prefer?",
        options=[
            None,
            "A: A lottery with a 50% chance of winning $100 or $0",
            "B: A guaranteed payout of $50",
            "C: Indifferent"
        ],
        format_func=lambda x: "" if x is None else x.replace("$", "\\$")
    )

    if q1 is None:  # No option selected initially
        st.warning("Please select an option to continue.")
        st.stop()

    # Logic for Question 2 based on Question 1
    if q1 == "A: A lottery with a 50% chance of winning $100 or $0":
        q2 = st.radio(
            "2. Now, what would you prefer?",
            options=[
                "A: A lottery with a 50% chance of winning $100 or $0",
                "B: A guaranteed payout of $70"
            ],
            format_func=lambda x: x.replace("$", "\\$")
        )
    elif q1 == "B: A guaranteed payout of $50":
        q2 = st.radio(
            "2. Now, what would you prefer?",
            options=[
                "A: A lottery with a 50% chance of winning $100 or $0",
                "B: A guaranteed payout of $30"
            ],
            format_func=lambda x: x.replace("$", "\\$")
        )
    elif q1 == "C: Indifferent":
        st.success("You are risk-neutral. Your risk aversion score is: 3.")
        st.stop()

    # Determine Final Score
    if q1 == "A: A lottery with a 50% chance of winning $100 or $0" and q2 == "A: A lottery with a 50% chance of winning $100 or $0":
        risk_aversion_score = 1  # Highly risk-seeking
    elif q1 == "A: A lottery with a 50% chance of winning $100 or $0" and q2 == "B: A guaranteed payout of $70":
        risk_aversion_score = 2  # Low risk aversion
    elif q1 == "B: A guaranteed payout of $50" and q2 == "B: A guaranteed payout of $30":
        risk_aversion_score = 5  # Highly risk-averse
    elif q1 == "B: A guaranteed payout of $50" and q2 == "A: A lottery with a 50% chance of winning $100 or $0":
        risk_aversion_score = 4  # Moderately high risk aversion
    elif q1 == "C: Indifferent" or q2 == "C: Indifferent":
        risk_aversion_score = 3  # Neutral

    # Display Final Score
    if st.button("Calculate My Risk Aversion Score"):
        st.success(f"Your risk aversion score is: {risk_aversion_score}")

        # Interpretation of Results
        if risk_aversion_score == 1:
            st.write("You have a *very low risk aversion* (highly risk-seeking).")
        elif risk_aversion_score == 2:
            st.write("You have a *low risk aversion*.")
        elif risk_aversion_score == 3:
            st.write("You have a *neutral risk aversion* (balanced between risk and reward).")
        elif risk_aversion_score == 4:
            st.write("You have a *moderately high risk aversion*.")
        elif risk_aversion_score == 5:
            st.write("You have a *very high risk aversion* (strong preference for safety).")