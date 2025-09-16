import streamlit as st

st.set_page_config(page_title="VDE Analyzer", page_icon="⚡", layout="wide")

def main():
    st.title("⚡ Vehicle Demanded Energy Analyzer")
    st.markdown("""
    Welcome!  
    - Go to **📥 Data & Setup** to upload a cycle and parameters.  
    - Use **⚙️ VDE & Gain** to compute KPIs.  
    - Compare scenarios in **📊 Comparison & Report**.  
    """)

if __name__ == "__main__":
    main()
