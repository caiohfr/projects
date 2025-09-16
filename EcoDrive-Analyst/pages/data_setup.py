import streamlit as st

def main():
    st.title("📥 Data & Setup")
    st.markdown("Upload a drive cycle and define vehicle parameters here.")

    uploaded = st.file_uploader("Upload cycle CSV (t,v)", type=["csv"])
    if uploaded:
        st.success("File uploaded (parsing not yet implemented).")

    st.sidebar.header("Vehicle Parameters")
    f0 = st.sidebar.number_input("f0 [N]", 0.0, 200.0, 10.0)
    f1 = st.sidebar.number_input("f1 [N·s/m]", 0.0, 5.0, 0.8)
    f2 = st.sidebar.number_input("f2 [N·s²/m²]", 0.0, 1.0, 0.12)
    mass = st.sidebar.number_input("Mass [kg]", 600.0, 3000.0, 1500.0)

if __name__ == "__main__":
    main()
