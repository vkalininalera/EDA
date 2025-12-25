import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pandas import DataFrame
from matplotlib.ticker import FuncFormatter, MaxNLocator
import geopandas as gpd

data: DataFrame = pd.read_csv('Sample - Superstore.csv', encoding='windows-1252')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# Convert dates
data['Order Date'] = pd.to_datetime(data['Order Date'])
data['Ship Date'] = pd.to_datetime(data['Ship Date'])
# print(data.info())

concentration = data[['Country', 'Region', 'State']].value_counts()

state_sales = (
    data.groupby("State")["Sales"]
    .sum()
    .reset_index()
)
state_sales = state_sales[state_sales["Sales"] >= 10]
state_sales = state_sales.sort_values(by="Sales", ascending=False)

# Plot 1 'Total Sales by State'
plt.figure(figsize=(14, 6))
plt.bar(state_sales["State"], state_sales["Sales"], color="steelblue")
plt.title("Total Sales by State")
plt.xlabel("State")
plt.ylabel("Total Sales")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_1_states_provinces.zip"
states = gpd.read_file(url)

us_states = states[states["admin"] == "United States of America"]
map_data = us_states.merge(
    state_sales,
    left_on="name",
    right_on="State",
    how="left"
)

# Plot 2 'Total Sales by State on map'
fig: Figure
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
map_data.plot(
    column="Sales",
    cmap="Blues",
    linewidth=0.8,
    edgecolor="black",
    legend=True,
    ax=ax
)
ax.set_title("Total Sales by U.S. State", fontsize=16)
ax.axis("off")
plt.show()

start_date = data['Order Date'].min()
end_date = data['Order Date'].max()

data['Year'] = data['Order Date'].dt.year
sales_by_year = data.groupby('Year')['Sales'].sum()

# Plot 3 'Total Sales Over Time'
plt.figure(figsize=(6, 4))
plt.plot(sales_by_year.index, sales_by_year.values, marker='o')
plt.xticks(sales_by_year.index)
plt.title('Total Sales Over Time')
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()

profit_by_year = (
    data.groupby('Year')['Profit'].sum().reset_index()
)
# Plot 4 'Total Profit by Year'
plt.figure(figsize=(6, 4))
plt.bar(profit_by_year['Year'], profit_by_year['Profit'], color='steelblue')
plt.xticks(profit_by_year['Year'])
plt.title('Total Profit by Year')
plt.xlabel('Year')
plt.ylabel('Profit')
plt.show()

discount_year_sales = (
    data
    .groupby(['Year', 'Discount'])['Sales']
    .sum()
    .reset_index()
)


# Plot 5 'Total Sales (Revenue in USD) by Discount Level per Year'
def dollar_formatter(x, pos):
    return f'${x:,.0f}'


fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
years = [2014, 2015, 2016, 2017]

fig.suptitle('Total Sales (Revenue in USD) by Discount Level per Year', fontsize=14)

for ax, year in zip(axes.flatten(), years):
    subset = discount_year_sales[discount_year_sales['Year'] == year]

    ax.bar(subset['Discount'], subset['Sales'], width=0.05)
    ax.set_title(f'{year}')
    ax.set_xlabel('Discount (%)')
    ax.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))

axes[0, 0].set_ylabel('Total Sales (USD)')
axes[1, 0].set_ylabel('Total Sales (USD)')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

quantity_by_year = (
    data.groupby('Year')['Quantity']
    .sum()
    .reset_index()
)

# Plot 6 'Total Quantity Sold by Year'
plt.figure(figsize=(6, 4))
plt.bar(quantity_by_year['Year'], quantity_by_year['Quantity'], color='steelblue')
plt.xticks(quantity_by_year['Year'])
plt.title('Total Quantity Sold by Year')
plt.xlabel('Year')
plt.ylabel('Quantity')
plt.show()

# Plot 7 'Total Profit by Product Category per Year'
profit_year_category = data.groupby(['Category', 'Year'])['Profit'].sum().reset_index()
categories = profit_year_category['Category'].unique()

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
for ax, year in zip(axes.flatten(), years):
    subset = profit_year_category[profit_year_category['Year'] == year]
    ax.bar(subset['Category'], subset['Profit'], color='steelblue')
    ax.set_title(str(year))
    ax.set_xlabel('Category')
    ax.set_ylabel('Total Profit (USD)')
    ax.tick_params(axis='x')
    ax.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))

fig.suptitle('Total Profit by Product Category per Year', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

profit_cat_sub = (
    data.groupby(['Year', 'Category', 'Sub-Category'])['Profit']
    .sum()
    .reset_index()
)
# Plot 8 'Profit Trend by Technology Sub-Category'
off_supp_profit = profit_cat_sub[profit_cat_sub['Category'] == 'Technology']
plt.figure(figsize=(8, 5))

for subcat in off_supp_profit['Sub-Category'].unique():
    subset = off_supp_profit[off_supp_profit['Sub-Category'] == subcat]
    plt.plot(
        subset['Year'],
        subset['Profit'],
        marker='o',
        label=subcat
    )

plt.axhline(0, color='red', linestyle='--', label='Zero Profit')
plt.xticks(sorted(off_supp_profit['Year'].unique()))
plt.title('Profit Trend by Technology Sub-Category')
plt.xlabel('Year')
plt.ylabel('Total Profit (USD)')
plt.gca().yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 9 'Profit Trend by Office-Suppliers Sub-Category'
off_supp_profit = profit_cat_sub[profit_cat_sub['Category'] == 'Office Supplies']
plt.figure(figsize=(8, 5))

for subcat in off_supp_profit['Sub-Category'].unique():
    subset = off_supp_profit[off_supp_profit['Sub-Category'] == subcat]
    plt.plot(
        subset['Year'],
        subset['Profit'],
        marker='o',
        label=subcat
    )
plt.axhline(0, color='red', linestyle='--', label='Zero Profit')
plt.xticks(sorted(off_supp_profit['Year'].unique()))
plt.title('Profit Trend by Office-Supplies Sub-Category')
plt.xlabel('Year')
plt.ylabel('Total Profit (USD)')
plt.gca().yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 10. Yearly Profit Trend for Copiers and Paper
subset = data[data['Sub-Category'].isin(['Copiers', 'Paper'])]

profit_year_subcat = (
    subset
    .groupby(['Year', 'Sub-Category'])['Profit']
    .sum()
    .reset_index()
)
plt.figure(figsize=(8, 5))
for subcat in ['Copiers', 'Paper']:
    sub = profit_year_subcat[profit_year_subcat['Sub-Category'] == subcat]
    plt.plot(sub['Year'], sub['Profit'], marker='o', label=subcat)

plt.axhline(0, color='gray', linestyle='--')
plt.xticks(sorted(profit_year_subcat['Year'].unique()))
plt.title('Yearly Profit Trend: Copiers vs Paper')
plt.xlabel('Year')
plt.ylabel('Total Profit (USD)')
plt.gca().yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 11. Correlation Matrix between yearly profits of copiers and paper.
pivot = profit_year_subcat.pivot(
    index='Year',
    columns='Sub-Category',
    values='Profit'
)
pivot.corr()
plt.figure(figsize=(4, 3))
sns.heatmap(
    pivot.corr(),
    annot=True,
    cmap='RdBu',
    fmt='.2f'
)
plt.title('Correlation Matrix: Copiers vs Paper')
plt.tight_layout()
plt.show()

# Data preparation for plot 12.
copiers = data[
    (data['Category'] == 'Technology') &
    (data['Sub-Category'] == 'Copiers')
    ].copy()
copiers['Year'] = copiers['Order Date'].dt.year
copiers['Unit_Price'] = copiers['Sales'] / copiers['Quantity']
avg_price_year = (
    copiers.groupby('Year')['Unit_Price']
    .mean()
    .reset_index()
)
# Plot 12. Discount Distribution for Copiers by Year
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=copiers,
    x='Year',
    y='Discount',
    showfliers=True
)

plt.title('Discount Distribution for Copiers by Year')
plt.xlabel('Year')
plt.ylabel('Discount')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Total quantity of copiers sold per year
copiers_qty_year = (
    copiers.groupby('Year')['Quantity']
    .sum()
    .reset_index()
)
# Plot 13. Total Quantity of Copiers Sold per Year
plt.figure(figsize=(6, 4))
plt.bar(
    copiers_qty_year['Year'],
    copiers_qty_year['Quantity'],
    color='steelblue'
)

plt.title('Total Quantity of Copiers Sold per Year')
plt.xticks(copiers_qty_year['Year'])
plt.xlabel('Year')
plt.ylabel('Total Units Sold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 14. Average Unit Price of Copiers by Year
plt.figure(figsize=(7, 4))

plt.plot(
    avg_price_year['Year'],
    avg_price_year['Unit_Price'],
    marker='o'
)

plt.title('Average Unit Price of Copiers by Year')
plt.xticks(avg_price_year['Year'])
plt.xlabel('Year')
plt.ylabel('Average Unit Price (USD)')

plt.grid(True)
plt.tight_layout()
plt.show()

# Data preparation for plot 15.
paper = data[
    (data['Category'] == 'Office Supplies') &
    (data['Sub-Category'] == 'Paper')
    ].copy()
paper['Year'] = paper['Order Date'].dt.year
paper['Unit_Price'] = paper['Sales'] / paper['Quantity']
avg_price_paper_year = (
    paper.groupby('Year')['Unit_Price']
    .mean()
    .reset_index()
)
discount_counts = (
    paper
    .groupby(['Year', 'Discount'])
    .size()
    .reset_index(name='Transactions')
)


def dollar_formatter_cents(x, pos):
    return f'${x:.2f}'


# Plot 15. Average Unit Price of Paper by Year

plt.figure(figsize=(7, 4))
plt.plot(
    avg_price_paper_year['Year'],
    avg_price_paper_year['Unit_Price'],
    marker='o'
)

plt.title('Average Unit Price of Paper by Year')
plt.xlabel('Year')
plt.ylabel('Average Unit Price (USD)')
plt.gca().yaxis.set_major_formatter(FuncFormatter(dollar_formatter_cents))
plt.xticks(avg_price_paper_year['Year'])

plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 16. Total Quantity of Paper Sold per Year
paper_qty_year = (
    paper.groupby('Year')['Quantity']
    .sum()
    .reset_index()
)

plt.figure(figsize=(6, 4))
plt.bar(
    paper_qty_year['Year'],
    paper_qty_year['Quantity'],
    color='steelblue'
)

plt.title('Total Quantity of Paper Sold per Year')
plt.xticks(paper_qty_year['Year'])
plt.xlabel('Year')
plt.ylabel('Total Units Sold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Plot 17. Paper: Number of Transactions by Discount Level per Year

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
years = sorted(discount_counts['Year'].unique())

for ax, year in zip(axes.flatten(), years):
    subset = discount_counts[discount_counts['Year'] == year]

    ax.bar(
        subset['Discount'] * 100,  # convert to %
        subset['Transactions'],
        width=8,
        color='steelblue'
    )

    ax.set_title(str(year))
    ax.set_xlabel('Discount (%)')
    ax.set_ylabel('Number of Transactions')
    ax.set_xticks([0, 20])

fig.suptitle('Paper: Number of Transactions by Discount Level per Year', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
