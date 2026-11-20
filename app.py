import streamlit as st
import pandas as pd

# 1. Page setup
st.set_page_config(page_title="Rail KPI Dashboard", layout="wide")
st.title("Rail KPI Dashboard 2025")  # <- your title
st.caption("Fictional data: daily operations at 6 stations in 2025")  # <- your subtitle

# 2. Load the data (the CSV sits in the same folder as app.py)
df = pd.read_csv("daily_operations.csv")  # <- your CSV file name
df["date"] = pd.to_datetime(df["date"])

# 3. Station dropdown
stations = ["All stations"] + sorted(df["station"].unique())
choice = st.selectbox("Station", stations)
if choice != "All stations":
    df = df[df["station"] == choice]

# 4. Three KPI tiles
col1, col2, col3 = st.columns(3)
col1.metric("Total passengers", f"{df['passengers'].sum():,}", border=True)
col2.metric("Punctuality rate", f"{df['punctuality_rate'].mean():.1f} %", border=True)
col3.metric("Average delay", f"{df['avg_delay_minutes'].mean():.1f} min", border=True)

# 5. One chart: average punctuality rate per month
df["month"] = df["date"].dt.month
monthly = df.groupby("month")["punctuality_rate"].mean()
st.subheader("Punctuality rate by month (%)")
st.bar_chart(monthly, x_label="Month (1 = January)", y_label="Punctuality rate (%)", height=450)
