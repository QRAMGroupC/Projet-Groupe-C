import streamlit as st


def display_introduction():
    st.session_state['last_page'] = "introduction"
    # Title and Subtitle
    st.title("Commodity Portfolio Optimization Dashboard")
    st.subheader("Harness Data-Driven Insights to Empower Your Investment Strategies")

    # Add the banner image using the provided URL
    st.image(
        "https://think.ing.com/uploads/hero/_w800h450/Commodities_montage_.PNG",
        use_column_width=True,
        caption="Optimize your portfolio with advanced insights.",
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
