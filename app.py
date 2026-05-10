import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Explorer",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 House Price Dataset — Interactive Dashboard")
st.markdown(
    "Explore **214 houses** across location, condition, size, and year built. "
    "Use the controls in each section to show or hide charts."
)

# ─────────────────────────────────────────────
# 1. Load Data with Caching
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("House_Price_Dataset.csv")
    df = df.drop(columns=["Id"])
    return df

df = load_data()

# ─────────────────────────────────────────────
# 2. Raw Data Preview
# ─────────────────────────────────────────────
st.header("1. Dataset Overview")

if st.checkbox("Show raw data"):
    st.write(df)

col1, col2, col3 = st.columns(3)
col1.metric("Total Houses", len(df))
col2.metric("Average Price", f"${df['Price'].mean():,.0f}")
col3.metric("Price Range", f"${df['Price'].min():,} – ${df['Price'].max():,}")

st.dataframe(df.describe().round(2))

# ─────────────────────────────────────────────
# 3. Price Distribution
# ─────────────────────────────────────────────
st.header("2. Price Distribution")
st.markdown("How is the target variable — `Price` — distributed across all houses?")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Histogram")
    fig, ax = plt.subplots(figsize=(6, 4))
    df["Price"].hist(bins=30, ax=ax, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Count")
    ax.set_title("House Price Distribution")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k")
    )
    st.pyplot(fig)
    plt.close()

with col_b:
    st.subheader("KDE — Plotly Distribution Plot")
    fig_kde = ff.create_distplot(
        [df["Price"].tolist()],
        group_labels=["Price"],
        colors=["steelblue"],
        bin_size=30000,
        show_rug=False,
    )
    fig_kde.update_layout(
        title="Price Density (KDE)",
        xaxis_title="Price ($)",
        yaxis_title="Density",
        height=350,
    )
    st.plotly_chart(fig_kde, use_container_width=True)

# Line chart of sorted prices
st.subheader("Sorted Price Trend")
st.markdown("Each house sorted by price — shows the overall spread and any step changes.")
sorted_prices = df["Price"].sort_values().reset_index(drop=True).rename("Price ($)")
st.line_chart(sorted_prices)

# ─────────────────────────────────────────────
# 4. Numerical Features
# ─────────────────────────────────────────────
st.header("3. Numerical Feature Distributions")

num_cols = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"]

st.subheader("Histograms of All Numerical Features")
fig2, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    axes[i].hist(df[col], bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    axes[i].set_title(col)
    axes[i].set_ylabel("Count")
axes[-1].set_visible(False)
plt.tight_layout()
st.pyplot(fig2)
plt.close()

# Area chart — running average price by area bin
st.subheader("Area vs Average Price (Area Chart)")
st.markdown(
    "Houses are binned by area (500 sq ft buckets). "
    "The area chart shows average price per bucket."
)
df["AreaBin"] = (df["Area"] // 500) * 500
area_avg = df.groupby("AreaBin")["Price"].mean().reset_index()
area_avg.columns = ["Area Bin (sq ft)", "Avg Price"]
area_avg = area_avg.set_index("Area Bin (sq ft)")
st.area_chart(area_avg)

# ─────────────────────────────────────────────
# 5. Location Data (with checkbox interactivity)
# ─────────────────────────────────────────────
st.header("4. Location Analysis")

if st.checkbox("Show Location raw counts"):
    st.write(df["Location"].value_counts())

st.subheader("Number of Houses by Location")
location_counts = df["Location"].value_counts()
st.bar_chart(location_counts)

st.subheader("Average Price by Location")
location_avg = df.groupby("Location")["Price"].mean().sort_values(ascending=False)
st.bar_chart(location_avg)

if st.checkbox("Show Price distribution per Location (Plotly)"):
    loc_order = ["Downtown", "Suburban", "Urban", "Rural"]
    groups = [df[df["Location"] == loc]["Price"].tolist() for loc in loc_order]
    fig_loc = ff.create_distplot(
        groups,
        group_labels=loc_order,
        bin_size=40000,
        show_rug=False,
    )
    fig_loc.update_layout(
        title="Price Distribution by Location",
        xaxis_title="Price ($)",
        yaxis_title="Density",
        height=400,
    )
    st.plotly_chart(fig_loc, use_container_width=True)

# ─────────────────────────────────────────────
# 6. Condition Data
# ─────────────────────────────────────────────
st.header("5. Condition Analysis")

if st.checkbox("Show Condition Information"):
    st.write(df["Condition"].value_counts())

st.subheader("Number of Houses by Condition")
cond_order = ["Excellent", "Good", "Fair", "Poor"]
cond_counts = df["Condition"].value_counts().reindex(cond_order)
st.bar_chart(cond_counts)

if st.button("Click to see Average Price by Condition"):
    cond_avg = df.groupby("Condition")["Price"].mean().reindex(cond_order)
    st.subheader("Average Price by Condition")
    st.bar_chart(cond_avg)

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    colors = {
        "Excellent": "#2ecc71",
        "Good": "#3498db",
        "Fair": "#f39c12",
        "Poor": "#e74c3c",
    }
    ax3.bar(
        cond_order,
        [cond_avg[c] for c in cond_order],
        color=[colors[c] for c in cond_order],
        edgecolor="white",
        alpha=0.85,
    )
    ax3.set_ylabel("Average Price ($)")
    ax3.set_title("Average Price by House Condition")
    ax3.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k")
    )
    st.pyplot(fig3)
    plt.close()

# ─────────────────────────────────────────────
# 7. Garage Data
# ─────────────────────────────────────────────
st.header("6. Garage Analysis")

st.subheader("Garage Distribution")
garage_counts = df["Garage"].value_counts()
st.bar_chart(garage_counts)

if st.button("Click to see Price comparison: Garage vs No Garage"):
    garage_avg = df.groupby("Garage")["Price"].mean()
    st.metric(
        "Garage Premium",
        f"${garage_avg['Yes'] - garage_avg['No']:,.0f}",
        delta_color="normal",
    )
    col_g1, col_g2 = st.columns(2)
    col_g1.metric("Avg Price — With Garage", f"${garage_avg['Yes']:,.0f}")
    col_g2.metric("Avg Price — No Garage", f"${garage_avg['No']:,.0f}")

    groups_g = [
        df[df["Garage"] == "Yes"]["Price"].tolist(),
        df[df["Garage"] == "No"]["Price"].tolist(),
    ]
    fig_g = ff.create_distplot(
        groups_g,
        group_labels=["Has Garage", "No Garage"],
        colors=["#27ae60", "#e74c3c"],
        bin_size=40000,
        show_rug=False,
    )
    fig_g.update_layout(
        title="Price Distribution: Garage vs No Garage",
        xaxis_title="Price ($)",
        yaxis_title="Density",
        height=400,
    )
    st.plotly_chart(fig_g, use_container_width=True)

# ─────────────────────────────────────────────
# 8. Year Built Trend
# ─────────────────────────────────────────────
st.header("7. Year Built Trend")
st.markdown("Average house price grouped by construction decade.")

df["Decade"] = (df["YearBuilt"] // 10) * 10
decade_avg = df.groupby("Decade")["Price"].mean().rename("Avg Price ($)")
st.line_chart(decade_avg)

if st.checkbox("Show decade data as bar chart"):
    st.bar_chart(decade_avg)

# ─────────────────────────────────────────────
# 9. Correlation Heatmap
# ─────────────────────────────────────────────
st.header("8. Correlation Heatmap")
st.markdown("Pearson correlation between all numerical features including `Price`.")

fig4, ax4 = plt.subplots(figsize=(7, 5))
corr = df[["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt", "Price"]].corr()
import matplotlib.colors as mcolors
cmap = plt.get_cmap("coolwarm")
im = ax4.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
plt.colorbar(im, ax=ax4)
ax4.set_xticks(range(len(corr.columns)))
ax4.set_yticks(range(len(corr.columns)))
ax4.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
ax4.set_yticklabels(corr.columns, fontsize=9)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax4.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                 fontsize=8, color="black")
ax4.set_title("Correlation Matrix")
plt.tight_layout()
st.pyplot(fig4)
plt.close()

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("House Price Dataset · EDA Dashboard · Built with Streamlit")
