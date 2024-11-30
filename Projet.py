import streamlit as st
from streamlit_option_menu import option_menu

from computations.data_exploration import compute_exploration_data
from data_management import initialize_session_state
from displays.aversion_questionnaire import display_aversion_questionnaire
from displays.black_litterman import display_black_litterman_performance
from displays.compare import display_comparison
from displays.data_exploration import display_data_exploration
from displays.efficient_frontier import display_efficient_frontier
from displays.introduction import display_introduction
from displays.methodology import display_methodology
from displays.mvs_no_rolling import display_minimum_variance_strategy
from displays.mvs_rolling import display_rolling_window_performance


def app():
    initialize_session_state()
    if "last_page" not in st.session_state:
        st.session_state["last_page"] = None

    # Customize the sidebar layout
    with st.sidebar:
        # Adding an option menu with icons for better navigation
        choice = option_menu(
            "Portfolio Optimization",  # Sidebar title
            [
                "Introduction",
                "Data Exploration",
                "Efficient Frontier",
                "Risk Aversion Questionnaire",
                "Minimum Variance Strategy without Rolling Window",
                "Performance of Minimun Variance Strategy with Rolling Window",
                "Performance of Black-litterman",
                "Comparison : Minimum Variance vs Blacklitterman",
                "Methodology",
            ],  # Options list
            icons=[
                "house",
                "bar-chart",
                "graph-up",
                "book",
                "calculator",
                "search",
                "search",
                "graph-up",
                "book",
            ],
            # Icons for each option
            menu_icon="cast",  # Sidebar menu icon
            default_index=0,  # Default selected index
            orientation="vertical",  # Vertical menu style
            styles={
                "container": {
                    "padding": "5px",
                    "background-color": "#white",  # Light background color for sidebar
                },
                "icon": {
                    "color": "#black",  # Dark blue icon color
                    "font-size": "18px",
                },
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "5px",
                    "padding": "10px",
                    "color": "#black",  # Dark blue for text
                },
                "nav-link-selected": {
                    "background-color": "#4e73df",  # Blue background for selected item
                    "color": "white",  # White text when selected
                },
            },
        )

    if choice == "Introduction":
        display_introduction()
    elif choice == "Methodology":
        display_methodology()
    elif choice == "Data Exploration":
        compute_exploration_data()
        display_data_exploration()
    elif choice == "Efficient Frontier":
        display_efficient_frontier()
    elif choice == "Minimum Variance Strategy without Rolling Window":
        display_minimum_variance_strategy()
    elif choice == "Performance of Minimun Variance Strategy with Rolling Window":
        display_rolling_window_performance()
    elif choice == "Performance of Black-litterman":
        display_black_litterman_performance()
    elif choice == "Comparison : Minimum Variance vs Blacklitterman":
        display_comparison()
    elif choice == "Risk Aversion Questionnaire":
        display_aversion_questionnaire()


if __name__ == "__main__":
    app()
