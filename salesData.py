import yfinance as yf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objs as go
import seaborn as sns
import base64
image = 'loandata/analyst2.jpg'
st.image(image, caption=None, width=400, use_column_width=None, clamp=False,
         channels="RGB", output_format="auto")


st.write(""" # Analysis of Sales Data""")
data = pd.read_csv('loandata/sales_data_sample.csv', encoding='latin-1')
st.subheader("view of the Sales Data")
st.write(data)
st.subheader("Selecting filtered columns")
columns = st.multiselect("Columns:",data.columns)
filter = st.radio("Choose by:", ("inclusion","exclusion"))

if filter == "exclusion":
    columns = [col for col in data.columns if col not in columns]
data[columns]

st.subheader("Sum of Null Values")
st.write(data.isnull().sum())

data['ADDRESSLINE2'] = data['ADDRESSLINE2'].fillna('No Address Line 2')
most_common_state = data['STATE'].mode()[0]
data['STATE'] = data['STATE'].fillna(most_common_state)
most_common_postalcode = data['POSTALCODE'].mode()[0]
data['POSTALCODE'] = data['POSTALCODE'].fillna(most_common_postalcode)
# Fill missing territory with the most common value in each country
# The original line was causing an error because the index of the result from
# groupby and apply was incompatible with the original DataFrame's index.
# We reset the index to align it with the original DataFrame before assignment.
data['TERRITORY'] = data.groupby('COUNTRY')['TERRITORY'].apply(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown')).reset_index(level=0, drop=True)
st.subheader("removing null values")
st.write(data.isnull().sum())
st.subheader("Descriptive Statistics",divider="green")
st.write(data.describe())
st.subheader("Total sales across the entire dataset")
total_sales = data['SALES'].sum()
st.write(f'The total Sales is: {total_sales}')
print("\n")
st.subheader("The average sales amount per order")
average_sales = data['SALES'].mean()
st.write(f' The average sales amount per order is : {average_sales}')
print("\n")
st.subheader("Total number of orders  in each year")
orders_per_year = data['YEAR_ID'].value_counts()
st.write(f'The total number of orders in each year {orders_per_year}')
print("\n")
st.subheader("The distribution of order statuses")
order_status_distribution = data['STATUS'].value_counts()
st.write(f'The distribution of order statuses is: {order_status_distribution}')
print("\n")
st.subheader("Top 5 countries by total sales")
top_countries_by_sales = data.groupby('COUNTRY')['SALES'].sum().sort_values(ascending=False).head(5)
st.write(f'The top 5 countries by total sales are: {top_countries_by_sales}')
print("\n")
st.subheader("Top 5 states by total sales in the USA")
top_states_by_sales_usa = data[data['COUNTRY'] == 'USA'].groupby('STATE')['SALES'].sum().sort_values(ascending=False).head(5)
st.write(f'The top 5 states by total sales in the USA are: {top_states_by_sales_usa}')
print("\n")
st.subheader("The most common deal size")
common_deal_size = data['DEALSIZE'].mode()[0]
st.write(f'The most common deal size is: {common_deal_size}')
print("\n")
st.subheader("The month that has the highest sales across all years")
monthly_sales = data.groupby('MONTH_ID')['SALES'].sum().sort_values(ascending=False).head(1)
st.write(f'The month with the highest sales across all years is: {monthly_sales}')
print("\n")
st.subheader("The distribution of deal sizes (Small, Medium, Large)")
deal_size_distribution = data['DEALSIZE'].value_counts()
st.write(f'The distribution of deal sizes is: {deal_size_distribution}')

st.subheader("Distribution of Sales per Deal Size")
fig, ax = plt.subplots()
sns.countplot(x='DEALSIZE', data=data, ax=ax, hue='DEALSIZE')
st.pyplot(fig)

st.subheader("Top 10 countries by Sales")
fig, ax = plt.subplots()
top_countries_sales = data.groupby('COUNTRY')['SALES'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=top_countries_sales.values, y=top_countries_sales.index, hue=top_countries_sales.values, ax=ax, legend=False)
plt.xlabel('Sales (in $)')
st.pyplot(fig)

st.subheader("Sales by Year")
fig, ax = plt.subplots()
sns.countplot(x='YEAR_ID', data=data, ax=ax, hue='YEAR_ID',  legend=False)
plt.ylabel('Number of Orders')
st.pyplot(fig)

st.subheader("Sales by Month")
fig, ax = plt.subplots()
monthly_sales_plot = data.groupby('MONTH_ID')['SALES'].sum().sort_values(ascending=False)
sns.barplot(x=monthly_sales_plot.index, y=monthly_sales_plot.values, hue=monthly_sales_plot.index, ax=ax, legend=False)
plt.xlabel('Month')
plt.ylabel('Total Sales (in $)')
st.pyplot(fig)

st.subheader("Sales by Deal Size")
fig, ax = plt.subplots()
deal_size_sales = data.groupby('DEALSIZE')['SALES'].sum()
sns.barplot(x=deal_size_sales.index, y=deal_size_sales.values, hue=deal_size_sales.index, ax=ax, legend=False)
plt.xlabel('Deal Size')
plt.ylabel('Total Sales (in $)')
st.pyplot(fig)

st.subheader("Sales by Order Status")
fig, ax = plt.subplots()
sns.countplot(x='STATUS', data=data, hue='STATUS',ax=ax, legend=False)
st.pyplot(fig)

st.subheader("Top States by Sales in the USA")
fig, ax = plt.subplots()
top_states_sales = data[data['COUNTRY'] == 'USA'].groupby('STATE')['SALES'].sum().sort_values(ascending=False).head(5)
sns.barplot(x=top_states_sales.values, y=top_states_sales.index, hue=top_states_sales.values, ax=ax, legend=False)
plt.xlabel('Sales (in $)')
st.pyplot(fig)

st.subheader("Correlation between Quantity Ordered and Sales")
fig, ax = plt.subplots()
sns.scatterplot(x='QUANTITYORDERED', y='SALES', data=data, hue='DEALSIZE', ax=ax, palette='Dark2')
plt.xlabel('Quantity Ordered')
plt.ylabel('Sales (in $)')
st.pyplot(fig)

st.subheader("Heatmap of Sales Correlations")
fig, ax = plt.subplots()
sns.heatmap(data[['QUANTITYORDERED', 'PRICEEACH', 'SALES']].corr(), annot=True, cmap='coolwarm')
st.pyplot(fig)

st.subheader("Sales by Quarter")
fig, ax = plt.subplots()
sns.barplot(x='QTR_ID', y='SALES', data=data, estimator=sum, errorbar=None, ax=ax, hue='QTR_ID', legend=False)
st.pyplot(fig)

st.subheader("Sales by Product Line (if column available)")
if 'PRODUCTLINE' in data.columns:
    fig, ax = plt.subplots()
    productline_sales = data.groupby('PRODUCTLINE')['SALES'].sum().sort_values(ascending=False)
    sns.barplot(x=productline_sales.values, y=productline_sales.index, hue=productline_sales.values, legend=False)
    plt.xlabel('Sales (in $)')
    st.pyplot(fig)

st.subheader("Sales Trend Over Time (by order date)")
fig, ax = plt.subplots()
data['ORDERDATE'] = pd.to_datetime(data['ORDERDATE'], errors='coerce')
data['ORDER_MONTH'] = data['ORDERDATE'].dt.to_period('M')
fig, ax = plt.subplots()
monthly_sales_trend = data.groupby('ORDER_MONTH')['SALES'].sum()
monthly_sales_trend.plot()
plt.xlabel('Order Month')
plt.ylabel('Total Sales (in $)')
st.pyplot(fig)

st.subheader('Sales Distribution by Country')
fig, ax = plt.subplots()
sns.boxplot(x='COUNTRY', y='SALES', data=data)
plt.xticks(rotation=90)
plt.xlabel('Country')
plt.ylabel('Sales (in $)')
st.pyplot(fig)

st.subheader('Relationship Between Quantity Ordered and Sales')
fig, ax = plt.subplots()
sns.regplot(x='QUANTITYORDERED', y='SALES', data=data, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.xlabel('Quantity Ordered')
plt.ylabel('Sales (in $)')
st.pyplot(fig)

st.subheader('Top 15 Products by Sales')
fig, ax = plt.subplots()
product_sales = data.groupby('PRODUCTCODE')['SALES'].sum().sort_values(ascending=False).head(15)
sns.barplot(x=product_sales.values, y=product_sales.index, hue=product_sales.values, legend=False)
plt.xlabel('Sales (in $)')
plt.ylabel('Product Code')
st.pyplot(fig)

plt.close()

st.subheader('Order Status Across Deal Sizes')
fig, ax = plt.subplots()
sns.countplot(x='DEALSIZE', hue='STATUS', data=data, palette='Set1')
plt.xlabel('Deal Size')
plt.ylabel('Order Count')
st.pyplot(fig)

st.subheader('Sales by Product line')
if 'PRODUCTLINE' in data.columns:
    fig, ax = plt.subplots()
    product_line_sales = data.groupby('PRODUCTLINE')['SALES'].sum().sort_values(ascending=False)
    sns.barplot(x=product_line_sales.values, y=product_line_sales.index, hue=product_line_sales.values, legend=False)
    plt.xlabel('Sales (in $)')
    plt.ylabel('Product Line')
    st.pyplot(fig)

st.subheader("Sales by Year and Quarter")
fig, ax = plt.subplots(figsize=(12, 8))
sales_by_qtr_year = data.groupby(['YEAR_ID', 'QTR_ID'])['SALES'].sum().unstack()
sales_by_qtr_year.plot(kind='bar', stacked=True, ax=ax, colormap='Set3')
plt.xlabel('Year')
plt.ylabel('Total Sales (in $)')
ax.legend(title='Quarter')
st.pyplot(fig)

sales_heatmap = data.pivot_table(index='COUNTRY', columns='DEALSIZE', values='SALES', aggfunc='sum', fill_value=0)


st.subheader("Heatmap of Sales by Country and Deal Size")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(sales_heatmap, annot=True, fmt='.0f', cmap='YlGnBu', linewidths=.5, ax=ax)
ax.set_xlabel('Deal Size')
ax.set_ylabel('Country')
st.pyplot(fig)
plt.close()
st.subheader("Sales Distribution by Year")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='YEAR_ID', y='SALES', data=data, hue='YEAR_ID', ax=ax)
ax.set_xlabel('Year')
ax.set_ylabel('Sales (in $)')
st.pyplot(fig)
plt.close()

st.subheader('Total Sales by Territory')
if 'TERRITORY' in data.columns:
    territory_sales = data.groupby('TERRITORY')['SALES'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=territory_sales.values, y=territory_sales.index, ax=ax)
    ax.set_xlabel('Sales (in $)')
    ax.set_ylabel('Territory')
    st.pyplot(fig)
else:
    st.warning("The dataset does not contain a 'TERRITORY' column.")

plt.close()
st.subheader("Sales Distribution by Deal Size")
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(x='DEALSIZE', y='SALES', data=data, hue='DEALSIZE', ax=ax)
ax.set_xlabel('Deal Size')
ax.set_ylabel('Sales (in $)')
st.pyplot(fig)


# Get list of columns
columns = data.columns.tolist()

# Streamlit multiselect for column selection
selected_columns = st.multiselect("Select a column", columns, default="SALES")

# Check if the selected column exists and is a string column
if selected_columns:
    selected_column = selected_columns[0]

    if data[selected_column].dtype == 'object':  # Checking if the column is of type 'string'/'object'
        s = data[selected_column].str.strip().value_counts()
        trace = go.Bar(x=s.index, y=s.values, showlegend=True)
        layout = go.Layout(title=f"Count of Each Value in {selected_column}")
        fig = go.Figure(data=[trace], layout=layout)
        st.plotly_chart(fig)
    else:
        # Display a custom error message if a non-string column is selected
        st.error(f"The selected column '{selected_column}' is not a string column. Please select a string column.")
else:
    st.info("Please select a column to display its value counts.")


@st.cache_data
def load_data():
    df = pd.read_csv('loandata/sales_data_sample.csv',encoding='latin-1')
    return df

df = load_data()







