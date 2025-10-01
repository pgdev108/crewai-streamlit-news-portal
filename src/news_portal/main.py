import os
import json
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from crew import NewsPortalCrew

load_dotenv()
st.set_page_config(page_title="AI News Portal", layout="wide")

# --- Topics dropdown ---
DEFAULT_TOPICS = [
    "Cancer Health Research",
    "Obesity Research",
    "Cardiovascular Health",
    "Neuroscience",
    "Diabetes & Metabolism",
    "Climate & Health",
]

st.title("📰 AI News Portal (CrewAI + Streamlit)")
topic = st.selectbox("Choose a topic:", DEFAULT_TOPICS, index=0)

# Run button
run_clicked = st.button("Run Agents", type="primary")

# Placeholder for layout
left, right = st.columns([1, 3])

# Session state to hold results
if "results" not in st.session_state:
    st.session_state["results"] = None
if "active_menu" not in st.session_state:
    st.session_state["active_menu"] = "Home"

def render_home(final):
    st.subheader("🏠 Home: Best Articles & Editorial")
    home = final.get("home", {})
    best_articles = home.get("best_articles", [])
    best_editorial = home.get("best_editorial")

    st.markdown("### Top News (per sub-topic)")
    for a in best_articles:
        with st.container(border=True):
            st.markdown(f"**{a.get('title','(title)')}**  \n"
                        f"[Open Link]({a.get('url','')})  \n"
                        f"*{a.get('source','?')}*  |  {a.get('published_date','')}")
    st.markdown("---")
    st.markdown("### Best Editorial (overall)")
    if best_editorial:
        with st.container(border=True):
            st.markdown(f"**{best_editorial.get('title','(title)')}**  \n"
                        f"[Open Link]({best_editorial.get('url','')})  \n"
                        f"*{best_editorial.get('source','?')}*  |  {best_editorial.get('published_date','')}")
    else:
        st.info("No editorial selected yet.")

def render_subtopic(final, subtopic):
    st.subheader(f"📚 {subtopic}")
    ps = final.get("per_subtopic", {}).get(subtopic, {})
    cands = ps.get("candidates", [])
    eds = ps.get("editorial_candidates", [])
    best = ps.get("best_article")

    st.markdown("#### Best Article")
    if best:
        with st.container(border=True):
            st.markdown(f"**{best.get('title','(title)')}**  \n"
                        f"[Open Link]({best.get('url','')})  \n"
                        f"*{best.get('source','?')}*  |  {best.get('published_date','')}")
    else:
        st.info("No best article chosen for this sub-topic.")

    st.markdown("---")
    st.markdown("#### Candidates")
    for c in cands:
        with st.container(border=True):
            st.markdown(f"**{c.get('title','(title)')}**  \n"
                        f"[Open Link]({c.get('url','')})  \n"
                        f"*{c.get('source','?')}*  |  {c.get('published_date','')}")

    st.markdown("---")
    st.markdown("#### Editorial Candidate")
    if eds:
        e = eds[0]
        with st.container(border=True):
            st.markdown(f"**{e.get('title','(title)')}**  \n"
                        f"[Open Link]({e.get('url','')})  \n"
                        f"*{e.get('source','?')}*  |  {e.get('published_date','')}")
    else:
        st.info("No editorial candidate for this sub-topic.")

# Run the pipeline
if run_clicked:
    with st.spinner("Running Crew Agents..."):
        crew = NewsPortalCrew()
        results = crew.run_pipeline(topic)
        st.session_state["results"] = results
        st.session_state["active_menu"] = "Home"

# If we have results, render the two-pane layout
results = st.session_state.get("results")
if results:
    final = results.get("final", {})
    subtopics = final.get("subtopics", [])

    with left:
        st.markdown("### Sections")
        if st.button("Home", use_container_width=True):
            st.session_state["active_menu"] = "Home"
        # Show exactly 2 subtopics as buttons
        for stp in subtopics[:2]:
            if st.button(stp, use_container_width=True):
                st.session_state["active_menu"] = stp

        st.caption("Results are saved to ./output as JSON.")

    with right:
        active = st.session_state.get("active_menu", "Home")
        if active == "Home":
            render_home(final)
        else:
            render_subtopic(final, active)
else:
    with right:
        st.info("Pick a topic and click **Run Agents** to populate the portal.")
