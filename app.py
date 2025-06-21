import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet

st.set_page_config(page_title="Lulu Retail Command Center", layout="wide")

@st.cache_data
def load_data():
    df_products = pd.read_csv("products_master.csv")
    df_stores = pd.read_csv("stores_master.csv")
    df_calendar = pd.read_csv("calendar_master.csv")
    df_inventory = pd.read_csv("inventory_transactions.csv")
    df_sales = pd.read_csv("sales_transactions.csv")
    return df_products, df_stores, df_calendar, df_inventory, df_sales

df_products, df_stores, df_calendar, df_inventory, df_sales = load_data()
df_sales['Date'] = pd.to_datetime(df_sales['Date'])
df_inventory['Date'] = pd.to_datetime(df_inventory['Date'])

# Merge for easy filtering
df_sales = df_sales.merge(df_products[['Product_ID', 'Category', 'Brand']], on='Product_ID', how='left')
df_inventory = df_inventory.merge(df_products[['Product_ID', 'Category', 'Brand']], on='Product_ID', how='left')

# Sidebar filters
st.sidebar.header("Global Filters")
store_list = ['All'] + df_stores['Store_Name'].tolist()
category_list = ['All'] + sorted(df_products['Category'].unique())
store_sel = st.sidebar.selectbox("Store", store_list)
category_sel = st.sidebar.selectbox("Category", category_list)
date_range = st.sidebar.date_input("Date Range", [df_sales['Date'].min(), df_sales['Date'].max()])

# Filter logic
sales_data = df_sales.copy()
inv_data = df_inventory.copy()
if store_sel != 'All':
    store_id = df_stores[df_stores['Store_Name'] == store_sel]['Store_ID'].values[0]
    sales_data = sales_data[sales_data['Store_ID'] == store_id]
    inv_data = inv_data[inv_data['Store_ID'] == store_id]
if category_sel != 'All':
    sales_data = sales_data[sales_data['Category'] == category_sel]
    inv_data = inv_data[inv_data['Category'] == category_sel]
sales_data = sales_data[(sales_data['Date'] >= pd.to_datetime(date_range[0])) & (sales_data['Date'] <= pd.to_datetime(date_range[1]))]
inv_data = inv_data[(inv_data['Date'] >= pd.to_datetime(date_range[0])) & (inv_data['Date'] <= pd.to_datetime(date_range[1]))]

# Tabs
tabs = st.tabs([
    "🏠 Overview",
    "📈 Sales Analytics",
    "📦 Inventory Analytics",
    "📊 Outlier & Anomaly Detection",
    "💡 Promotions & Uplift",
    "📅 Forecasting",
    "🏬 Store & Brand",
    "🗂️ Raw Data"
])

# --- Overview Tab ---
with tabs[0]:
    st.header("Executive Overview Dashboard")
    total_sales = sales_data['Net_Sales_AED'].sum()
    total_units = sales_data['Units_Sold'].sum()
    promo_sales = sales_data[sales_data['Promotion_Flag'] == 'Y']['Net_Sales_AED'].sum()
    promo_pct = promo_sales / total_sales * 100 if total_sales > 0 else 0
    unique_skus = sales_data['Product_ID'].nunique()
    stores_count = sales_data['Store_ID'].nunique()
    categories = sales_data['Category'].nunique()
    brands = sales_data['Brand'].nunique()
    outlier_days = sales_data[sales_data['Units_Sold'] > sales_data['Units_Sold'].quantile(0.99)].shape[0]

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.metric("Net Sales (AED)", f"{total_sales:,.0f}")
    col2.metric("Units Sold", f"{total_units:,}")
    col3.metric("Promo Sales (%)", f"{promo_pct:.1f}")
    col4.metric("Active SKUs", unique_skus)
    col5.metric("Categories", categories)
    col6.metric("Brands", brands)
    col7.metric("High Outlier Days", outlier_days)

    # Top SKUs
    st.subheader("Top 10 SKUs by Net Sales")
    top_skus = sales_data.groupby(['Product_ID', 'Category'])['Net_Sales_AED'].sum().reset_index().sort_values('Net_Sales_AED', ascending=False).head(10)
    top_skus = top_skus.merge(df_products[['Product_ID', 'Product_Name']], on='Product_ID', how='left')
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=top_skus, x='Product_Name', y='Net_Sales_AED', hue='Category', ax=ax1, dodge=False)
    ax1.set_xlabel('SKU')
    ax1.set_ylabel('Net Sales (AED)')
    ax1.set_title('Top 10 SKUs by Net Sales')
    plt.xticks(rotation=35, ha='right')
    st.pyplot(fig1)

    # Net Sales Trend
    st.subheader("Net Sales Over Time")
    trend = sales_data.groupby('Date')['Net_Sales_AED'].sum().reset_index()
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(trend['Date'], trend['Net_Sales_AED'], marker='o', color="#388E3C")
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Net Sales (AED)')
    ax2.set_title('Net Sales Over Time')
    plt.xticks(rotation=25)
    st.pyplot(fig2)

# --- Sales Analytics Tab ---
with tabs[1]:
    st.header("Sales Analytics")
    st.subheader("Monthly & Weekly Trends")
    monthly = sales_data.groupby(sales_data['Date'].dt.to_period('M'))['Net_Sales_AED'].sum().reset_index()
    monthly['Date'] = monthly['Date'].dt.to_timestamp()
    weekly = sales_data.groupby(sales_data['Date'].dt.to_period('W'))['Net_Sales_AED'].sum().reset_index()
    weekly['Date'] = weekly['Date'].dt.to_timestamp()

    fig3, ax3 = plt.subplots(figsize=(8, 3))
    ax3.plot(monthly['Date'], monthly['Net_Sales_AED'], marker='s', label='Monthly')
    ax3.plot(weekly['Date'], weekly['Net_Sales_AED'], marker='o', label='Weekly', alpha=0.5)
    ax3.set_title("Net Sales: Monthly & Weekly")
    ax3.set_ylabel("Net Sales (AED)")
    ax3.legend()
    st.pyplot(fig3)

    st.subheader("Sales Heatmap: Category vs. Month")
    sales_data['Month'] = sales_data['Date'].dt.strftime("%b-%Y")
    heatmap_data = sales_data.groupby(['Category','Month'])['Net_Sales_AED'].sum().unstack(fill_value=0)
    fig4, ax4 = plt.subplots(figsize=(11, 4))
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax4)
    st.pyplot(fig4)

    st.subheader("Top 10 Brands")
    brand_sales = sales_data.groupby('Brand')['Net_Sales_AED'].sum().sort_values(ascending=False).head(10)
    fig5, ax5 = plt.subplots(figsize=(8, 3))
    sns.barplot(x=brand_sales.index, y=brand_sales.values, ax=ax5)
    ax5.set_ylabel('Net Sales (AED)')
    ax5.set_title('Top 10 Brands')
    plt.xticks(rotation=35)
    st.pyplot(fig5)

    st.subheader("Sales Outliers Table (Top 1%)")
    threshold = sales_data['Units_Sold'].quantile(0.99)
    st.dataframe(sales_data[sales_data['Units_Sold'] > threshold][['Date','Product_ID','Units_Sold','Net_Sales_AED','Promotion_Flag','Category','Brand']])

# --- Inventory Analytics Tab ---
with tabs[2]:
    st.header("Inventory Analytics")
    st.subheader("Closing Stock by Store (Top 5)")
    stock_sum = inv_data.groupby('Store_ID')['Closing_Stock'].sum().reset_index()
    stock_sum = stock_sum.merge(df_stores[['Store_ID', 'Store_Name']], on='Store_ID', how='left').sort_values('Closing_Stock', ascending=False).head(5)
    fig6, ax6 = plt.subplots(figsize=(7, 3))
    sns.barplot(x='Store_Name', y='Closing_Stock', data=stock_sum, ax=ax6, color="#1976D2")
    ax6.set_xlabel('Store')
    ax6.set_ylabel('Closing Stock (Total Units)')
    plt.xticks(rotation=20)
    st.pyplot(fig6)

    st.subheader("Stock Aging (Avg Days of Cover)")
    avg_daily_sales = sales_data.groupby('Product_ID')['Units_Sold'].mean().reset_index()
    avg_stock = inv_data.groupby('Product_ID')['Closing_Stock'].mean().reset_index()
    stock_aging = avg_stock.merge(avg_daily_sales, on='Product_ID', how='left')
    stock_aging['Days_of_Cover'] = stock_aging['Closing_Stock'] / (stock_aging['Units_Sold']+1)
    stock_aging = stock_aging.merge(df_products[['Product_ID','Product_Name']], on='Product_ID')
    st.dataframe(stock_aging[['Product_Name','Days_of_Cover']].sort_values('Days_of_Cover', ascending=False).head(12))

    st.subheader("Stock-Out Frequency by Store")
    stockout = inv_data[inv_data['Closing_Stock']==0].groupby('Store_ID').size().reset_index(name='StockoutDays')
    stockout = stockout.merge(df_stores[['Store_ID','Store_Name']], on='Store_ID', how='left')
    fig7, ax7 = plt.subplots(figsize=(7, 3))
    sns.barplot(x='Store_Name', y='StockoutDays', data=stockout, ax=ax7, color="#D32F2F")
    ax7.set_ylabel('Stockout Days')
    plt.xticks(rotation=15)
    st.pyplot(fig7)

# --- Outlier & Anomaly Detection Tab ---
with tabs[3]:
    st.header("Outlier & Anomaly Diagnostics")
    st.subheader("Extreme Sales/Returns Days")
    # High outliers (possible promo/surge), low outliers (possible returns)
    high_out = sales_data[sales_data['Units_Sold'] > sales_data['Units_Sold'].quantile(0.995)]
    low_out = sales_data[sales_data['Units_Sold'] <= 0]
    st.write(f"High Sales Days (Top 0.5%): {len(high_out)}")
    st.dataframe(high_out[['Date','Product_ID','Units_Sold','Net_Sales_AED','Promotion_Flag','Category','Brand']])
    st.write(f"Negative/Zero Sales Days (Returns/OOS): {len(low_out)}")
    st.dataframe(low_out[['Date','Product_ID','Units_Sold','Net_Sales_AED','Promotion_Flag','Category','Brand']])

    st.subheader("Promotion Impact Outliers")
    promo_impact = sales_data.groupby('Promotion_Flag')['Units_Sold'].describe()
    st.dataframe(promo_impact)

    st.subheader("Store Anomalies (Stockouts >10 days)")
    store_stockout = stockout[stockout['StockoutDays'] > 10]
    st.dataframe(store_stockout)

# --- Promotions & Uplift Tab ---
with tabs[4]:
    st.header("Promotions & Price Uplift Analysis")
    st.subheader("Promo Uplift by Category")
    promo = sales_data[sales_data['Promotion_Flag']=='Y']
    nonpromo = sales_data[sales_data['Promotion_Flag']=='N']
    promo_cat = promo.groupby('Category')['Net_Sales_AED'].mean()
    nonpromo_cat = nonpromo.groupby('Category')['Net_Sales_AED'].mean()
    uplift = (promo_cat - nonpromo_cat) / (nonpromo_cat+1) * 100
    uplift_table = pd.DataFrame({'Promo_Sales':promo_cat, 'Non_Promo_Sales':nonpromo_cat, 'Uplift_%':uplift}).fillna(0)
    st.dataframe(uplift_table)

    st.subheader("Promo Share Top 10 SKUs")
    promo_sku = promo.groupby('Product_ID')['Net_Sales_AED'].sum().sort_values(ascending=False).head(10)
    top_promo_skus = promo_sku.index
    promo_sku_df = df_products[df_products['Product_ID'].isin(top_promo_skus)][['Product_ID','Product_Name']]
    promo_share = promo_sku.reset_index().merge(promo_sku_df, on='Product_ID', how='left')
    st.dataframe(promo_share)

    st.subheader("Promo Trend Over Time")
    promo_trend = promo.groupby('Date')['Net_Sales_AED'].sum().reset_index()
    nonpromo_trend = nonpromo.groupby('Date')['Net_Sales_AED'].sum().reset_index()
    fig8, ax8 = plt.subplots(figsize=(9,3))
    ax8.plot(promo_trend['Date'], promo_trend['Net_Sales_AED'], label="Promo Sales", color='blue')
    ax8.plot(nonpromo_trend['Date'], nonpromo_trend['Net_Sales_AED'], label="Non-Promo Sales", color='orange')
    ax8.set_title("Promo vs Non-Promo Sales Trend")
    ax8.legend()
    st.pyplot(fig8)

# --- Forecasting Tab ---
with tabs[5]:
    st.header("Advanced Forecasting")
    sku_list = df_products['Product_ID']
    selected_sku = st.selectbox("Select SKU for Forecasting", sku_list)
    fc_days = st.slider("Forecast Days", min_value=14, max_value=60, value=30)
    sku_df = sales_data[sales_data['Product_ID'] == selected_sku].groupby('Date')['Units_Sold'].sum().reset_index()
    sku_df.columns = ['ds','y']
    if len(sku_df) > 30:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        m.fit(sku_df)
        future = m.make_future_dataframe(periods=fc_days)
        fc = m.predict(future)
        fig9 = m.plot(fc)
        st.pyplot(fig9)
        st.dataframe(fc[['ds','yhat','yhat_lower','yhat_upper']].tail(fc_days))
    else:
        st.warning("Not enough historical data for this SKU for Prophet forecasting.")

# --- Store & Brand Tab ---
with tabs[6]:
    st.header("Store & Brand Deep Dives")
    st.subheader("Net Sales by Store")
    store_sales = sales_data.groupby('Store_ID')['Net_Sales_AED'].sum().reset_index()
    store_sales = store_sales.merge(df_stores[['Store_ID', 'Store_Name']], on='Store_ID', how='left')
    fig10, ax10 = plt.subplots(figsize=(9, 4))
    sns.barplot(y='Store_Name', x='Net_Sales_AED', data=store_sales.sort_values('Net_Sales_AED', ascending=False), ax=ax10)
    ax10.set_title("Net Sales by Store")
    st.pyplot(fig10)

    st.subheader("Brand by Store Heatmap")
    brand_store = sales_data.groupby(['Brand','Store_ID'])['Net_Sales_AED'].sum().unstack(fill_value=0)
    fig11, ax11 = plt.subplots(figsize=(11, 4))
    sns.heatmap(brand_store, annot=False, fmt=".0f", cmap="Blues", ax=ax11)
    ax11.set_title("Brand x Store: Net Sales")
    st.pyplot(fig11)

# --- Raw Data Tab ---
with tabs[7]:
    st.header("Raw Data - Download/Explore")
    with st.expander("Sales Data (first 500 rows)"):
        st.dataframe(sales_data.head(500))
    with st.expander("Inventory Data (first 500 rows)"):
        st.dataframe(inv_data.head(500))
    st.download_button("Download Sales Data (CSV)", sales_data.to_csv(index=False), "sales_data.csv")
    st.download_button("Download Inventory Data (CSV)", inv_data.to_csv(index=False), "inventory_data.csv")

st.markdown("""
---
<b>Next-Gen Retail Dashboard • Realistic Data • Executive Command Center</b>
""", unsafe_allow_html=True)
