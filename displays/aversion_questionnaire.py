import streamlit as st


def display_aversion_questionnaire():
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
            "C: Indifferent",
        ],
        format_func=lambda x: "" if x is None else x.replace("$", "\\$"),
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
                "B: A guaranteed payout of $70",
            ],
            format_func=lambda x: x.replace("$", "\\$"),
        )
    elif q1 == "B: A guaranteed payout of $50":
        q2 = st.radio(
            "2. Now, what would you prefer?",
            options=[
                "A: A lottery with a 50% chance of winning $100 or $0",
                "B: A guaranteed payout of $30",
            ],
            format_func=lambda x: x.replace("$", "\\$"),
        )
    elif q1 == "C: Indifferent":
        st.success("You are risk-neutral. Your risk aversion score is: 3.")
        st.stop()

    # Determine Final Score
    if (
        q1 == "A: A lottery with a 50% chance of winning $100 or $0"
        and q2 == "A: A lottery with a 50% chance of winning $100 or $0"
    ):
        risk_aversion_score = 1  # Highly risk-seeking
    elif (
        q1 == "A: A lottery with a 50% chance of winning $100 or $0"
        and q2 == "B: A guaranteed payout of $70"
    ):
        risk_aversion_score = 2  # Low risk aversion
    elif (
        q1 == "B: A guaranteed payout of $50" and q2 == "B: A guaranteed payout of $30"
    ):
        risk_aversion_score = 5  # Highly risk-averse
    elif (
        q1 == "B: A guaranteed payout of $50"
        and q2 == "A: A lottery with a 50% chance of winning $100 or $0"
    ):
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
            st.write(
                "You have a *neutral risk aversion* (balanced between risk and reward)."
            )
        elif risk_aversion_score == 4:
            st.write("You have a *moderately high risk aversion*.")
        elif risk_aversion_score == 5:
            st.write(
                "You have a *very high risk aversion* (strong preference for safety)."
            )
