import traceback
import streamlit as st
from crew import NewsPortalCrew

st.set_page_config(page_title="Cancer Health Care News Portal", layout="wide")
st.title("🧬 Cancer Health Care News Portal")

SUBTOPICS = [
    "Cancer Research & Prevention",
    "Early Detection and Diagnosis",
    "Cancer Drug Discovery and Development",
    "Cancer Treatment Methods",
    "Precision Oncology",
]

# ---- Session State ----
if "results" not in st.session_state:
    st.session_state["results"] = None
if "error" not in st.session_state:
    st.session_state["error"] = None
if "active_menu" not in st.session_state:
    st.session_state["active_menu"] = "Home"
if "running" not in st.session_state:
    st.session_state["running"] = False

# ---- Helpers to render ----
def card_article(a: dict):
    st.markdown(
        f"**{a.get('title','(title)')}**  \n"
        f"[Open Link]({a.get('url','')})  \n"
        f"*Published:* {a.get('published_date','?')}  \n\n"
        f"{a.get('summary','(no summary)')}"
    )

def render_home(final):
    st.subheader("🏠 Home")
    home = final.get("home", {})
    best_articles = home.get("best_articles", [])
    main_editorial = home.get("main_editorial", "")
    st.markdown("### Featured Articles (Best per Sub-topic)")
    for a in best_articles:
        with st.container(border=True):
            card_article(a)
    st.markdown("---")
    st.markdown("### Main Editorial")
    if main_editorial:
        st.write(main_editorial)
    else:
        st.info("Main editorial not available.")

def render_subtopic(final, subtopic):
    st.subheader(f"📚 {subtopic}")
    ps = final.get("per_subtopic", {}).get(subtopic, {})
    st.markdown("#### Editorial")
    editorial = ps.get("editorial")
    if editorial:
        st.write(editorial)
    else:
        st.info("Editorial not available for this sub-topic.")
    st.markdown("---")
    st.markdown("#### News Articles (5)")
    arts = ps.get("articles", [])
    if arts:
        for a in arts:
            with st.container(border=True):
                card_article(a)
    else:
        st.info("No articles found for this sub-topic.")

# ---- Runner callback (ensures state is set and UI refreshes) ----
def run_agents():
    st.session_state["running"] = True
    st.session_state["error"] = None
    try:
        with st.spinner("Running agents for Cancer Health Care..."):
            crew = NewsPortalCrew()
            results = crew.run_pipeline()  # topic is fixed inside crew
            st.session_state["results"] = results
            st.session_state["active_menu"] = "Home"
    except Exception as e:
        st.session_state["error"] = f"{e}\n\n{traceback.format_exc()}"
        st.session_state["results"] = None
    finally:
        st.session_state["running"] = False
        # Force a fresh render so the results show immediately
        st.rerun()

# ---- Toolbar ----
cols = st.columns([1, 1, 6])
with cols[0]:
    st.button("Run Agents", type="primary", disabled=st.session_state["running"], on_click=run_agents)
with cols[1]:
    debug = st.toggle("Debug", value=False)

# ---- Layout ----
left, right = st.columns([1, 3])

# ---- Error surface ----
if st.session_state["error"]:
    st.error(st.session_state["error"])

# ---- Render results ----
results = st.session_state["results"]
if not results:
    with right:
        st.info("Click **Run Agents** to populate the portal.")
else:
    # Show debug if needed
    if debug:
        with right:
            st.write("### Debug: Raw results")
            st.json(results)

    final = results.get("final")
    # Be lenient: if "final" missing, show the last step’s JSON so you can see what came back
    if not final:
        # Try QA output, then chief-editor, then last step
        final = (results.get("step_12", {}) or {}).get("final") \
             or (results.get("step_11", {}) or {}).get("final")

    with left:
        st.markdown("### Sections")
        if st.button("Home", use_container_width=True, disabled=st.session_state["running"]):
            st.session_state["active_menu"] = "Home"
        for stp in SUBTOPICS:
            if st.button(stp, use_container_width=True, disabled=st.session_state["running"]):
                st.session_state["active_menu"] = stp
        st.caption("Results saved under ./output")

    with right:
        active = st.session_state.get("active_menu", "Home")
        if not final:
            st.warning("No `final` object found. Showing last available step:")
            last_key = max([k for k in results.keys() if k.startswith("step_")], default=None)
            if last_key:
                st.json(results[last_key])
            else:
                st.json(results)
        else:
            if active == "Home":
                render_home(final)
            else:
                render_subtopic(final, active)
