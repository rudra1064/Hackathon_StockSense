import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load Data
df_demand = pd.read_csv("demand_forecasting.csv").rename(columns={'Sales Quantity': 'Sales Volume', 'Promotions': 'Is_Promotion'})
df_inventory = pd.read_csv("inventory_monitoring.csv")
df_price = pd.read_csv("pricing_optimization.csv")

# Ensure Product ID is integer
for df in [df_demand, df_inventory, df_price]:
    df['Product ID'] = df['Product ID'].astype(str).str.strip().astype(int)

# Fill missing values with median/mode
df_demand.fillna(df_demand.median(numeric_only=True), inplace=True)
df_demand.fillna(df_demand.mode().iloc[0], inplace=True)

df_inventory.fillna(df_inventory.median(numeric_only=True), inplace=True)
df_inventory.fillna(df_inventory.mode().iloc[0], inplace=True)

df_price.fillna(df_price.median(numeric_only=True), inplace=True)
df_price.fillna(df_price.mode().iloc[0], inplace=True)

# Preprocess Data
df_demand['Is_Promotion'] = df_demand['Is_Promotion'].map({'Yes': 1, 'No': 0})

# Demand Forecasting using Naive Bayes
X_demand = pd.get_dummies(df_demand[['Price', 'Is_Promotion', 'Seasonality Factors', 'External Factors']])
y_demand = (df_demand['Sales Volume'] > df_demand['Sales Volume'].median()).astype(int)
scaler_demand = StandardScaler()
X_demand_scaled = scaler_demand.fit_transform(X_demand)
demand_model = GaussianNB().fit(X_demand_scaled, y_demand)
df_demand['Demand_Pred'] = demand_model.predict(X_demand_scaled)

# Inventory Monitoring using Random Forest
X_inventory = df_inventory[['Stock Levels', 'Stockout Frequency', 'Reorder Point', 'Warehouse Capacity']]
y_inventory = ((df_inventory['Stock Levels'] < df_inventory['Reorder Point']) | (df_inventory['Stockout Frequency'] > 5)).astype(int)
inventory_model = RandomForestClassifier(random_state=42).fit(X_inventory, y_inventory)
df_inventory['Inventory_Pred'] = inventory_model.predict(X_inventory)

# Pricing Optimization using XGBoost
X_price = df_price[['Price', 'Competitor Prices', 'Discounts', 'Return Rate (%)']]
y_price = (df_price['Elasticity Index'] > df_price['Elasticity Index'].median()).astype(int)
scaler_price = StandardScaler()
X_price_scaled = scaler_price.fit_transform(X_price)
price_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X_price_scaled, y_price)
df_price['Price_Pred'] = price_model.predict(X_price_scaled)

# Merge Predictions
final_df = df_demand[['Product ID', 'Demand_Pred', 'Seasonality Factors']].merge(
    df_inventory[['Product ID', 'Inventory_Pred', 'Order Fulfillment Time (days)', 'Supplier Lead Time (days)', 'Stockout Frequency']], on='Product ID', how='left'
).merge(
    df_price[['Product ID', 'Price_Pred']], on='Product ID', how='left'
)

# Determine Final Stock Status
def classify_stock(row):
    if row['Inventory_Pred'] == 1:
        return 'Understock'
    elif row['Demand_Pred'] == 1 and row['Price_Pred'] == 1:
        return 'Overstock'
    return 'Normal'

final_df['Final_Stock_Status'] = final_df.apply(classify_stock, axis=1)

# Predict Supply Chain Adjustment & Fulfillment Delays
def check_supply_chain(row):
    if row['Demand_Pred'] == 1:
        if row['Supplier Lead Time (days)'] > 10 or row['Stockout Frequency'] > 5:
            return 'Increase product intake'
        else:
            return 'Maintain current level'
    elif row['Demand_Pred'] == 0 and row['Stockout Frequency'] < 3:
        return 'Consider reducing intake'
    return 'No change needed'

def check_delay(row):
    if row['Demand_Pred'] == 1 and row['Order Fulfillment Time (days)'] > 7:
        return 'Potential Delay'
    return 'On-Time'

final_df['Supply_Chain_Action'] = final_df.apply(check_supply_chain, axis=1)
final_df['Fulfillment_Status'] = final_df.apply(check_delay, axis=1)

# Suggest Stock Availability Based on Seasonality
def festival_stock_plan(row):
    if row['Seasonality Factors'] in ['Festive', 'Holiday', 'Peak']:
        return 'Ensure higher stock availability'
    return 'Normal stocking'

final_df['Stock_Plan_Advice'] = final_df.apply(festival_stock_plan, axis=1)

# Streamlit UI
st.title("AI Stock Prediction")
st.write("Enter Product ID to check stock status, supply chain suggestion, fulfillment forecast, and seasonal stocking advice.")

product_id = st.text_input("Product ID:")
if product_id:
    try:
        product_id = int(product_id)
        result = final_df[final_df['Product ID'] == product_id]
        if not result.empty:
            stock_status = result.iloc[0]['Final_Stock_Status']
            supply_action = result.iloc[0]['Supply_Chain_Action']
            fulfillment_status = result.iloc[0]['Fulfillment_Status']
            season_advice = result.iloc[0]['Stock_Plan_Advice']

            st.write(f"### Stock Status: {stock_status}")
            st.write(f"🔄 Supply Chain Suggestion: {supply_action}")
            st.write(f"🚚 Fulfillment Forecast: {fulfillment_status}")
            st.write(f"📅 Seasonal Stock Advice: {season_advice}")
        else:
            st.write("### Product ID not found.")
    except:
        st.write("Please enter a valid numeric Product ID.")
