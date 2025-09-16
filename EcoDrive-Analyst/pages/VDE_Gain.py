import streamlit as st

def main():
    st.title("⚙️ VDE & Gain")
    st.markdown("Compute VDE_NET and estimate fuel/energy consumption.")

    st.metric("VDE_NET [kWh/100km]", "---")
    st.metric("Fuel consumption [L/100km]", "---")

if __name__ == "__main__":
    main()