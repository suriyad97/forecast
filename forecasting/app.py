import streamlit as st

st.set_page_config(page_title="Feature Engineering", layout="wide")

if "redirected_to_feature" not in st.session_state:
    st.session_state["redirected_to_feature"] = True
    try:
        st.switch_page("pages/2_Feature_Engineering.py")
    except Exception:
        st.warning("Use the sidebar to open the Feature Engineering page.")
        st.stop()
else:
    st.warning("Use the sidebar to open the Feature Engineering page.")
    st.stop()
