import streamlit as st

def main():
    st.title("📊 Comparison & Report")
    st.markdown("Compare Config A vs Config B and export results.")

    st.write("Table of KPIs will go here.")
    st.download_button("Export CSV", "data,kpi,values", "report.csv")

if __name__ == "__main__":
    main()
