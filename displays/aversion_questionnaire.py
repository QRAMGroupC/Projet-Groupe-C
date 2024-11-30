import streamlit as st


def display_aversion_questionnaire():
    st.header("Risk Aversion Questionnaire")

    st.write(
        """
        This questionnaire assesses your level of risk aversion. You will answer five questions, 
        each offering a choice between a riskier option and a safer alternative. Your final risk aversion 
        score will range from *1 (highly risk-seeking)* to *5 (highly risk-averse)*.
        """
    )

    scores = []  # List to store scores from each question

    # Scoring function
    def assign_score(choice):
        if choice.startswith("A"):
            return 1  # Risk-seeking
        elif choice.startswith("B"):
            return 5  # Risk-averse
        else:
            return 3  # Neutral

    # Question 1
    q1 = st.radio(
        "1. You have an opportunity to invest in a new startup company. There's a 30% chance you'll triple your investment, but a 70% chance you'll lose it all. Alternatively, you can invest in government bonds with a guaranteed modest return. What do you choose?",
        options=[
            "A: Invest in the startup company.",
            "B: Invest in government bonds.",
            "C: Indifferent",
        ],
    )
    scores.append(assign_score(q1))

    # Question 2
    q2 = st.radio(
        "2. You're considering a job offer at a small startup where your salary could be very high if the company succeeds, but there's a risk it might fail. Alternatively, you can accept a stable job with a moderate salary at a large corporation. What do you choose?",
        options=[
            "A: Accept the job at the small startup.",
            "B: Accept the job at the large corporation.",
            "C: Indifferent",
        ],
    )
    scores.append(assign_score(q2))

    # Question 3
    # Question 3
    q3 = st.radio(
        "3. You are deciding on a travel route. The first route is shorter but has a 20% chance of heavy traffic causing significant delays. The second route is longer but guarantees a predictable arrival time. What do you choose?",
        options=[
            "A: Take the shorter route with a chance of delays.",
            "B: Take the longer but predictable route.",
            "C: Indifferent",
        ],
    )
    scores.append(assign_score(q3))

    # Question 4
    q4 = st.radio(
        "4. You're planning a vacation. You can go to an adventurous destination with unpredictable weather and activities or choose a relaxing resort with guaranteed good weather and known amenities. What do you choose?",
        options=[
            "A: Go to the adventurous destination.",
            "B: Go to the relaxing resort.",
            "C: Indifferent",
        ],
    )
    scores.append(assign_score(q4))

    # Question 5
    q5 = st.radio(
        "5. You can participate in a game where you have a 10% chance to win 10,000, but a 90% chance of winning nothing. Alternatively, you can receive a guaranteed $1,000. What do you choose?",
        options=[
            "A: Participate in the game.",
            "B: Take the guaranteed $1,000.",
            "C: Indifferent",
        ],
    )
    scores.append(assign_score(q5))

    # Display Final Score
    if st.button("Calculate My Risk Aversion Score"):
        average_score = sum(scores) / len(scores)
        risk_aversion_score = round(average_score)

        st.success(f"Your risk aversion score is: {risk_aversion_score}")

        # Interpretation of Results
        if risk_aversion_score == 1:
            st.write("You have a *very low risk aversion* (highly risk-seeking).")
        elif risk_aversion_score == 2:
            st.write("You have a *low risk aversion*.")
        elif risk_aversion_score == 3:
            st.write("You have a *moderate risk aversion*.")
        elif risk_aversion_score == 4:
            st.write("You have a *high risk aversion*.")
        else:
            st.write("You have a *very high risk aversion* (strong preference for safety).")
