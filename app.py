import streamlit as st
import os

# Page Config must be the first Streamlit command
st.set_page_config(
    page_title="MIGET Modern",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import UI Modules
# Note: Ensure the package path is correct
from miget.ui.inverse import render_inverse_problem
from miget.ui.direct import render_direct_problem

def load_css():
    """Load custom CSS from assets"""
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_home():
    """Render the Home/Landing Page with Mode Selection"""
    st.markdown("<div style='text-align: center; margin-bottom: 50px;'>", unsafe_allow_html=True)
    st.title("MIGET Software 2025")
    st.markdown("##### Multiple Inert Gas Elimination Technique Analysis")
    st.markdown("</div>", unsafe_allow_html=True)

    # Centered Selection Cards using Columns
    _, col1, col2, _ = st.columns([1, 4, 4, 1], gap="large")
    
    with col1:
        st.markdown(
            """
            <div class="home-card card-forward">
                <h3>Forward Problem</h3>
                <p>
                    Simulate V/Q distributions based on log-normal parameters and predict gas retention/excretion data.
                </p>
            </div>
            """, unsafe_allow_html=True
        )
        if st.button("Start Forward Simulation ➡️", key="btn_direct", use_container_width=True, type="primary"):
            st.session_state['mode'] = 'direct'
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class="home-card card-reverse">
                <h3>Reverse Problem</h3>
                <p>
                    Recover the V/Q distribution from measured inert gas retention data using robust algorithms.
                </p>
            </div>
            """, unsafe_allow_html=True
        )
        if st.button("Start Reverse Analysis ⬅️", key="btn_inverse", use_container_width=True, type="primary"):
            st.session_state['mode'] = 'inverse'
            st.rerun()
            
    # Footer
    st.markdown("---")
    st.caption("© 2025 Mattia Busana | Version 1.0.1")

def main():
    # Load Styles
    load_css()
    
    # Session State Init
    if 'mode' not in st.session_state:
        st.session_state['mode'] = None
        
    # Router
    if st.session_state['mode'] is None:
        render_home()
        
    elif st.session_state['mode'] == 'inverse':
        with st.sidebar:
            if st.button("🏠 Home", use_container_width=True):
                st.session_state['mode'] = None
                st.rerun()
        render_inverse_problem()
        
    elif st.session_state['mode'] == 'direct':
        with st.sidebar:
            if st.button("🏠 Home", use_container_width=True):
                st.session_state['mode'] = None
                st.rerun()
        render_direct_problem()

if __name__ == "__main__":
    main()
