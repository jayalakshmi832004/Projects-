import pandas as pd
import streamlit as st

# Load the dataset
courses_df = pd.read_csv("student_course.csv")


# Streamlit UI
st.title("Course Recommendation System")

# Form inputs
with st.form("user_input_form"):
    st.subheader("Student Preferences")

    location = st.selectbox("Select Your Location:", courses_df["Student_Location"].unique())
    topic = st.selectbox("Select Your Preferred Topic:", courses_df["Preferred_Topics"].unique())
    category = st.selectbox("Select Course Category:", courses_df["Category"].unique())
    difficulty = st.radio("Select Difficulty Level:", courses_df["Difficulty_Level"].unique())
    num_recommendations = st.slider("How many recommendations do you want?", 1, 10, 5)

    submitted = st.form_submit_button("Get Recommendations")

# Process input and generate recommendations
if submitted:
    # Filter courses with partial matching
    filtered_courses = courses_df[
        (courses_df["Student_Location"] == location) |
        (courses_df["Preferred_Topics"] == topic) |
        (courses_df["Category"] == category) |
        (courses_df["Difficulty_Level"] == difficulty)
    ]

    # Sort by relevance (number of matches) and then by average rating
    filtered_courses["Relevance_Score"] = (
        (filtered_courses["Student_Location"] == location).astype(int) +
        (filtered_courses["Preferred_Topics"] == topic).astype(int) +
        (filtered_courses["Category"] == category).astype(int) +
        (filtered_courses["Difficulty_Level"] == difficulty).astype(int)
    )
    recommendations = (
        filtered_courses.sort_values(by=["Relevance_Score", "Average_Rating"], ascending=[False, False])
        .head(num_recommendations)
    )

    # Display recommendations
    if not recommendations.empty:
        st.subheader("Recommended Courses")
        st.table(recommendations[["Course_ID", "Course_Title", "Average_Rating", "Category"]])
    else:
        st.write("No matching courses found. Please try adjusting your preferences.")
