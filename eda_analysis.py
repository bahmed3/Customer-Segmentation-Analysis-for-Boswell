import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

customers_df = pd.read_csv('customers.csv')
transactions_df = pd.read_csv('transactions.csv')

print("Customers DataFrame Info:")
print(customers_df.info())
print("\nTransactions DataFrame Info:")
print(transactions_df.info())

print("\nDescriptive Statistics for Customers:")
print(customers_df.describe())

print("\nDescriptive Statistics for Transactions:")
print(transactions_df.describe())

print("\nMissing Values in Customers DataFrame:")
print(customers_df.isnull().sum())

print("\nMissing Values in Transactions DataFrame:")
print(transactions_df.isnull().sum())

plt.figure(figsize=(10, 6))
sns.histplot(customers_df['Age'], bins=15, kde=True, color='blue')
plt.title('Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'age_distribution.png'))
plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(x='Gender', data=customers_df, palette='Set2')
plt.title('Gender Distribution of Customers')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.savefig(os.path.join(output_dir, 'gender_distribution.png'))
plt.close()

customers_df['JoinDate'] = pd.to_datetime(customers_df['JoinDate'])
plt.figure(figsize=(12, 6))
customers_df['JoinDate'].hist(bins=30, color='green')
plt.title('Join Date Distribution of Customers')
plt.xlabel('Join Date')
plt.ylabel('Number of Customers')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'join_date_distribution.png'))
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(transactions_df['PurchaseAmount'], bins=20, kde=True, color='purple')
plt.title('Distribution of Purchase Amounts')
plt.xlabel('Purchase Amount')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'purchase_amount_distribution.png'))
plt.close()

transactions_df['PurchaseDate'] = pd.to_datetime(transactions_df['PurchaseDate'])
plt.figure(figsize=(12, 6))
transactions_df['PurchaseDate'].hist(bins=30, color='orange')
plt.title('Distribution of Transactions Over Time')
plt.xlabel('Purchase Date')
plt.ylabel('Number of Transactions')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'transaction_distribution_over_time.png'))
plt.close()

plt.figure(figsize=(12, 6))
avg_purchase_by_genre = transactions_df.groupby('Genre')['PurchaseAmount'].mean().sort_values()
sns.barplot(x=avg_purchase_by_genre.index, y=avg_purchase_by_genre.values, palette='viridis')
plt.title('Average Purchase Amount by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Purchase Amount')
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'average_purchase_by_genre.png'))
plt.close()

plt.figure(figsize=(10, 6))
corr_matrix = transactions_df[['PurchaseAmount']].join(customers_df[['Age']].set_index(customers_df['CustomerID']), on='CustomerID').corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
plt.close()

print(f"All plots have been saved to the '{output_dir}' folder.")
