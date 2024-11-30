import pandas as pd
import streamlit as st


def display_methodology():
    st.session_state['last_page'] = "methodology"
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
    st.latex(
        r"\text{Polarity Score} = \frac{\text{Bullish \%} - \text{Bearish \%}}{100}"
    )
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
    st.latex(
        r"R_{\text{commodity}, t} = \alpha + \beta \cdot \text{Sentiment}_t + \epsilon_t"
    )
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
        return f"{value:.8f}".rstrip("0").rstrip(".")

    # Apply custom formatting only for Wheat
    formatted_results = regression_results.copy()
    formatted_results.loc["Wheat", :] = formatted_results.loc["Wheat", :].apply(
        custom_format
    )
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
    st.latex(
        r"w_{\text{risky}} = \max\left(0, \min\left(1, \frac{\mu_p - r_f}{\lambda \cdot \sigma_p^2}\right)\right)"
    )
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
        r"\tilde{\mu} = r_f + \text{SR}(\mathbf{x_0} \mid r_f) \cdot \frac{\Sigma \mathbf{x_0}}{\sqrt{\mathbf{x_0}^T \Sigma \mathbf{x_0}}}"
    )
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
    st.latex(
        r"\max_{\mathbf{w}} \ \mathbf{w}^T \bar{\mu} - \frac{\gamma}{2} \mathbf{w}^T \Sigma \mathbf{w}"
    )
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
