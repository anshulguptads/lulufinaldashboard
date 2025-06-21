import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet

st.set_page_config(page_title="Lulu Retail Executive Analytics", layout="wide")

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

# Add category, brand, etc. to sales
df_sales = df_sales.merge(df_products[['Product_ID', 'Category', 'Brand']], on='Product_ID', how='left')
df_inventory = df_inventory.merge(df_products[['Product_ID', 'Category', 'Brand']], on='Product_ID', how='left')

tabs = st.tabs([
    "Overview",
    "Sales Analytics",
    "Promotion Analysis",
    "Inventory & Stock",
    "Forecasting",
    "Store/Brand",
    "Raw Data"
])

## ----- Overview -----
with tabs[0]:
    st.header("Executive Overview")
    total_sales = df_sales['Net_Sales_AED'].sum()
    total_units = df_sales['Units_Sold'].sum()
    promo_sales = df_sales[df_sales['Promotion_Flag'] == 'Y']['Net_Sales_AED'].sum()
    promo_pct = promo_sales / total_sales * 100 if total_sales > 0 else 0
    unique_skus = df_sales['Product_ID'].nunique()
    stores_count = df_sales['Store_ID'].nunique()
    categories = df_products['Category'].nunique()
    brands = df_products['Brand'].nunique()
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Net Sales (AED)", f"{total_sales:,.0f}")
    col2.metric("Total Units Sold", f"{total_units:,}")
    col3.metric("Active SKUs", unique_skus)
    col4.metric("Stores", stores_count)
    col5.metric("Categories", categories)
    col6.metric("Brands", brands)

    # Top 10 SKUs
    st.subheader("Top 10 SKUs by Net Sales")
    top_skus = df_sales.groupby(['Product_ID', 'Category'])['Net_Sales_AED'].sum().reset_index().sort_values('Net_Sales_AED', ascending=False).head(10)
    top_skus = top_skus.merge(df_products[['Product_ID', 'Product_Name']], on='Product_ID', how='left')
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    sns.barplot(data=top_skus, x='Product_Name', y='Net_Sales_AED', hue='Category', dodge=False, ax=ax1)
    ax1.set_xlabel('SKU')
    ax1.set_ylabel('Net Sales (AED)')
    ax1.set_title('Top 10 SKUs by Net Sales')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig1)

    # Sales Trend
    st.subheader("Total Net Sales Trend")
    sales_trend = df_sales.groupby('Date')['Net_Sales_AED'].sum().reset_index()
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(sales_trend['Date'], sales_trend['Net_Sales_AED'], marker='o', color="#388E3C")
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Net Sales (AED)')
    ax2.set_title('Net Sales Over Time')
    plt.xticks(rotation=45)
    st.pyplot(fig2)

## ----- Sales Analytics -----
with tabs[1]:
    st.header("Detailed Sales Analytics")
    # YOY/MOM growth (use first and last 30 days for demo)
    st.subheader("Month-on-Month (MOM) Sales Growth")
    df_sales['Month'] = df_sales['Date'].dt.to_period('M')
    mom = df_sales.groupby('Month')['Net_Sales_AED'].sum().pct_change()*100
    fig3, ax3 = plt.subplots(figsize=(8,4))
    ax3.bar(mom.index.astype(str), mom.values)
    ax3.set_ylabel('% Growth')
    ax3.set_title('Month-on-Month Net Sales Growth')
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    # Category/Brand Heatmap
    st.subheader("Sales Heatmap: Category vs. Month")
    cat_month = df_sales.groupby(['Category','Month'])['Net_Sales_AED'].sum().unstack(fill_value=0)
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.heatmap(cat_month, annot=True, fmt=".0f", cmap="Blues", ax=ax4)
    ax4.set_title("Net Sales by Category & Month")
    st.pyplot(fig4)

    st.subheader("Brand Analysis")
    brand_sales = df_sales.groupby('Brand')['Net_Sales_AED'].sum().sort_values(ascending=False).head(10)
    fig5, ax5 = plt.subplots(figsize=(7,4))
    sns.barplot(x=brand_sales.index, y=brand_sales.values, ax=ax5)
    ax5.set_xlabel('Brand')
    ax5.set_ylabel('Net Sales (AED)')
    ax5.set_title('Top 10 Brands by Net Sales')
    plt.xticks(rotation=45)
    st.pyplot(fig5)

## ----- Promotion Analysis -----
with tabs[2]:
    st.header("Promotion and Price Analysis")
    st.subheader("Promotion Uplift")
    promo = df_sales[df_sales['Promotion_Flag']=='Y']
    nonpromo = df_sales[df_sales['Promotion_Flag']=='N']
    promo_group = promo.groupby('Category')['Net_Sales_AED'].sum()
    nonpromo_group = nonpromo.groupby('Category')['Net_Sales_AED'].sum()
    uplift = ((promo_group / (nonpromo_group+1)) * 100).sort_values(ascending=False)
    st.dataframe(pd.DataFrame({'Promo Sales': promo_group, 'Non-Promo Sales': nonpromo_group, 'Uplift %': uplift.round(2)}))

    st.subheader("Promo Share by SKU")
    promo_share = promo.groupby('Product_ID')['Net_Sales_AED'].sum().sort_values(ascending=False).head(10)
    top_promo_skus = promo_share.index
    promo_sku_df = df_products[df_products['Product_ID'].isin(top_promo_skus)][['Product_ID','Product_Name']]
    promo_share = promo_share.reset_index().merge(promo_sku_df, on='Product_ID', how='left')
    st.dataframe(promo_share)

## ----- Inventory & Stock -----
with tabs[3]:
    st.header("Inventory, Stock Aging, and Stock-Out")
    st.subheader("Closing Stock by Store")
    inv_summary = df_inventory.groupby('Store_ID')['Closing_Stock'].sum().reset_index()
    inv_summary = inv_summary.merge(df_stores[['Store_ID', 'Store_Name']], on='Store_ID', how='left')
    inv_top5 = inv_summary.sort_values('Closing_Stock', ascending=False).head(5)
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    ax6.bar(inv_top5['Store_Name'], inv_top5['Closing_Stock'], color="#FFA000")
    ax6.set_xlabel('Store')
    ax6.set_ylabel('Closing Stock (Total Units)')
    ax6.set_title('Top 5 Stores by Closing Stock')
    plt.xticks(rotation=20)
    st.pyplot(fig6)

    st.subheader("Stock Aging (Average Days of Cover)")
    # Days of cover = Closing_Stock / Average Daily Sales
    inv_age = df_inventory.groupby(['Product_ID'])['Closing_Stock'].mean() / (df_sales.groupby('Product_ID')['Units_Sold'].mean()+1)
    inv_age = inv_age.reset_index().merge(df_products[['Product_ID','Product_Name']], on='Product_ID')
    inv_age.columns = ['Product_ID','AvgDaysCover','Product_Name']
    st.dataframe(inv_age.sort_values('AvgDaysCover', ascending=False).head(10))

    st.subheader("Stock-Out Frequency by Store")
    stockout = df_inventory[df_inventory['Closing_Stock']==0].groupby('Store_ID').size().reset_index(name='StockoutDays')
    stockout = stockout.merge(df_stores[['Store_ID','Store_Name']], on='Store_ID', how='left')
    fig7, ax7 = plt.subplots(figsize=(8, 5))
    ax7.bar(stockout['Store_Name'], stockout['StockoutDays'], color="#D32F2F")
    ax7.set_xlabel('Store')
    ax7.set_ylabel('Stockout Days')
    ax7.set_title('Stock-Out Frequency by Store')
    plt.xticks(rotation=20)
    st.pyplot(fig7)

## ----- Forecasting -----
with tabs[4]:
    st.header("SKU/Category Level Forecasting")
    sku_selected = st.selectbox("Select SKU for Forecasting", df_products['Product_ID'])
    sales_sku = df_sales[df_sales['Product_ID']==sku_selected].groupby('Date')['Units_Sold'].sum().reset_index()
    sales_sku.columns = ['ds','y']
    if len(sales_sku) > 30:
        m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
        m.fit(sales_sku)
        future = m.make_future_dataframe(periods=30)
        fc = m.predict(future)
        fig8 = m.plot(fc)
        st.pyplot(fig8)
        st.dataframe(fc[['ds','yhat','yhat_lower','yhat_upper']].tail(10))
    else:
        st.warning("Not enough data for forecasting (need at least 30 days per SKU).")

## ----- Store/Brand -----
with tabs[5]:
    st.header("Store & Brand Heatmaps")
    st.subheader("Sales by Store")
    store_sales = df_sales.groupby('Store_ID')['Net_Sales_AED'].sum().reset_index()
    store_sales = store_sales.merge(df_stores[['Store_ID', 'Store_Name']], on='Store_ID', how='left')
    fig9, ax9 = plt.subplots(figsize=(8, 5))
    sns.barplot(y='Store_Name', x='Net_Sales_AED', data=store_sales.sort_values('Net_Sales_AED', ascending=False), ax=ax9)
    ax9.set_title("Net Sales by Store")
    st.pyplot(fig9)

    st.subheader("Category by Store Heatmap")
    cat_store = df_sales.groupby(['Category','Store_ID'])['Net_Sales_AED'].sum().unstack(fill_value=0)
    fig10, ax10 = plt.subplots(figsize=(10, 5))
    sns.heatmap(cat_store, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax10)
    ax10.set_title("Category x Store: Net Sales")
    st.pyplot(fig10)

## ----- Raw Data -----
with tabs[6]:
    st.header("Raw Data Samples")
    st.write("**Sales Data:**")
    st.dataframe(df_sales.head(300))
    st.write("**Inventory Data:**")
    st.dataframe(df_inventory.head(300))

