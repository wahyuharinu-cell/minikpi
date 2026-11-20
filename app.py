import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Page setup
st.set_page_config(page_title="Train Journey Performance", layout="wide")
st.title("Train Journey Performance Dashboard")  # <- your title
st.caption("Company dataset: 30 train journeys, March–April 2026")  # <- your subtitle

# 2. Load the data (the CSV sits in the same folder as app.py)
df = pd.read_csv("train_journeys.csv")  # <- your CSV file name
df["date"] = pd.to_datetime(df["date"])
df["REVENUE"] = df["passengers"] * df["ticket_price"]
overall_mean = df["delay_minutes"].mean()

# 3. Four KPI tiles
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total passengers", f"{df['passengers'].sum():,}", border=True)
col2.metric("On-time rate", f"{(df['on_time'] == 'Yes').mean() * 100:.1f} %", border=True)
col3.metric("Average delay", f"{overall_mean:.1f} min", border=True)
col4.metric("Total revenue", f"{df['REVENUE'].sum():,.0f}", border=True)

st.divider()

# =========================================================
# Q1. Which departure stations have the longest delays?
# =========================================================
st.header("1. Which departure stations have the longest delays?")

station_delay = (
    df.groupby("station")
        .agg(TRIPS=("journey_id", "count"),
             MEAN_DELAY=("delay_minutes", "mean"),
             MAX_DELAY=("delay_minutes", "max"),
             ON_TIME_RATE=("on_time", lambda s: (s == "Yes").mean() * 100))
        .sort_values("MEAN_DELAY", ascending=False)
        .round(1)
)

c = ['darkblue'] * len(station_delay)
c[0] = "firebrick"  # first row = highest mean delay (sorted)

fig, ax = plt.subplots(figsize=(9, 3))
bars = ax.bar(station_delay.index, station_delay["MEAN_DELAY"], color=c)
ax.set_title("AVERAGE DELAY BY DEPARTURE STATIONS", fontsize=12, fontweight="bold")
ax.set_xlabel("Departure station", fontsize=11)
ax.set_ylabel("Mean delay (minutes)", fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.8)
ax.set_axisbelow(True)
ax.bar_label(bars, fmt="%.1f", padding=1, fontsize=10)
ax.axhline(overall_mean, color="darkgreen", linestyle="--", linewidth=1.5,
           label=f"Overall mean ({overall_mean:.1f} min)")
ax.legend()
ax.set_ylim(0, 11)
plt.tight_layout()
st.pyplot(fig)

with st.expander("Show table"):
    st.dataframe(station_delay)

st.info("**Finding:** Daejeon has the longest average delay at 9.6 minutes, more than double "
        "the overall mean of 4.5 minutes, while Daegu is the most punctual at 1.2 minutes. "
        "Seoul has a moderate average (6.5 minutes) but recorded the single worst delay "
        "(21 minutes), so both stations deserve operational review.")

st.divider()

# =========================================================
# Q2. Are busier trains more likely to be delayed?
# =========================================================
st.header("2. Are busier trains more likely to be delayed?")

r = df["passengers"].corr(df["delay_minutes"])

df["LOAD_GROUP"] = pd.cut(df["passengers"],
                          bins=[0, 150, 250, 350],
                          labels=["Low (<150)", "Medium (150-250)", "High (>250)"])

load_delay = (
    df.groupby("LOAD_GROUP", observed=True)
        .agg(TRIPS=("journey_id", "count"),
             MEAN_DELAY=("delay_minutes", "mean"),
             MAX_DELAY=("delay_minutes", "max"))
        .round(1)
)

slope, intercept = np.polyfit(df["passengers"], df["delay_minutes"], 1)
x = np.linspace(df["passengers"].min(), df["passengers"].max(), 100)

fig, ax = plt.subplots(figsize=(9, 4))
ax.scatter(df["passengers"], df["delay_minutes"], color="darkblue", s=50, alpha=0.7, edgecolor="white")
ax.plot(x, slope * x + intercept, color="firebrick", linestyle="--", linewidth=1.5,
        label=f"Trend line (r = {r:.2f})")
ax.set_title("PASSENGERS vs DELAY", fontsize=12, fontweight="bold")
ax.set_xlabel("Number of passengers", fontsize=11)
ax.set_ylabel("Delay (minutes)", fontsize=11)
ax.grid(linestyle='--', alpha=0.8)
ax.set_axisbelow(True)
ax.legend()
plt.tight_layout()
st.pyplot(fig)

with st.expander("Show table"):
    st.dataframe(load_delay)

st.info("**Finding:** There is a weak negative correlation between passenger numbers and delay "
        "(r = −0.32), so busier trains were not more likely to be delayed; trips with more than "
        "250 passengers averaged only 2.4 minutes. This is unlikely to be causal and probably "
        "reflects other factors, such as busier services running on better-resourced lines.")

st.divider()

# =========================================================
# Q3. Does performance change by day of the week or over time?
# =========================================================
st.header("3. Does performance change by day of the week or over time?")

df["WEEKDAY"] = df["date"].dt.day_name()
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

weekday_delay = (
    df.groupby("WEEKDAY")
        .agg(TRIPS=("journey_id", "count"),
             MEAN_DELAY=("delay_minutes", "mean"),
             MAX_DELAY=("delay_minutes", "max"))
        .reindex(day_order)
        .round(1)
)

df["WEEK"] = df["date"].dt.to_period("W").dt.start_time
weekly_delay = (
    df.groupby("WEEK")
        .agg(TRIPS=("journey_id", "count"),
             MEAN_DELAY=("delay_minutes", "mean"))
        .round(1)
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

bars = ax1.bar(weekday_delay.index, weekday_delay["MEAN_DELAY"], color="darkblue")
ax1.set_title("MEAN DELAY BY DAY OF WEEK", fontsize=12, fontweight="bold")
ax1.set_xlabel("Day of week", fontsize=11)
ax1.set_ylabel("Mean delay (minutes)", fontsize=11)
ax1.grid(axis='y', linestyle='--', alpha=0.8)
ax1.set_axisbelow(True)
ax1.bar_label(bars, fmt="%.1f", padding=1, fontsize=10)
ax1.set_ylim(0, 10)

ax2.plot(df["date"], df["delay_minutes"], color="darkblue", marker="o", markersize=4, linewidth=1, label="Daily delay")
ax2.plot(weekly_delay.index, weekly_delay["MEAN_DELAY"], color="firebrick", marker="s", linewidth=2, label="Weekly mean")
ax2.set_title("DELAY OVER TIME", fontsize=12, fontweight="bold")
ax2.set_xlabel("Date", fontsize=11)
ax2.set_ylabel("Delay (minutes)", fontsize=11)
ax2.grid(linestyle='--', alpha=0.8)
ax2.set_axisbelow(True)
ax2.tick_params(axis="x", rotation=45)
ax2.legend()

plt.tight_layout()
st.pyplot(fig)

with st.expander("Show tables"):
    left, right = st.columns(2)
    left.dataframe(weekday_delay)
    right.dataframe(weekly_delay)

st.info("**Finding:** Delays are highest midweek (Tuesday 7.0 and Wednesday 6.5 minutes) and "
        "lowest on Friday (2.3 minutes). Punctuality improved over the period, with weekly mean "
        "delays falling from about 6.6 minutes in early March to about 2.4 minutes from late "
        "March onward.")

st.divider()

# =========================================================
# Q4. Which routes generate the most revenue?
# =========================================================
st.header("4. Which routes generate the most revenue?")

df["ROUTE"] = df["station"] + " → " + df["destination"]

route_revenue = (
    df.groupby("ROUTE")
        .agg(TRIPS=("journey_id", "count"),
             PASSENGERS=("passengers", "sum"),
             AVG_PRICE=("ticket_price", "mean"),
             REVENUE=("REVENUE", "sum"))
        .sort_values("REVENUE", ascending=False)
        .round(1)
)

top = route_revenue.head(10).sort_values("REVENUE")
c = ['darkblue'] * len(top)
c[-1] = "firebrick"  # last row = highest revenue

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(top.index, top["REVENUE"], color=c)
ax.set_title("TOP 10 ROUTES BY REVENUE", fontsize=12, fontweight="bold")
ax.set_xlabel("Total revenue", fontsize=11)
ax.set_ylabel("Route", fontsize=11)
ax.grid(axis='x', linestyle='--', alpha=0.8)
ax.set_axisbelow(True)
ax.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=10)
ax.set_xlim(0, top["REVENUE"].max() * 1.15)
plt.tight_layout()
st.pyplot(fig)

with st.expander("Show table (all routes)"):
    st.dataframe(route_revenue)

st.info("**Finding:** Incheon → Daegu generated the most revenue (27,013), followed by "
        "Daegu → Busan, which earned the second most because of its high fares despite carrying "
        "fewer passengers. Seoul → Daejeon carried almost as many passengers as the top route "
        "but ranked only 9th because of its low average fare.")

st.divider()

# 5. Conclusion
st.caption("With only 30 journeys and one trip per day, these findings are indicative rather than "
           "conclusive, and more data would be needed before recommending service changes.")
