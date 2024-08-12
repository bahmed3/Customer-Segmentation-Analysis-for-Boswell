import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
customers_df = pd.read_csv('customers.csv')
transactions_df = pd.read_csv('transactions.csv')

# Aggregate transaction data to get total purchase amount and number of transactions per customer
transaction_summary = transactions_df.groupby('CustomerID').agg(
    TotalPurchaseAmount=('PurchaseAmount', 'sum'),
    NumTransactions=('TransactionID', 'count')
).reset_index()

# Merge with customer demographic data
customer_data = pd.merge(customers_df, transaction_summary, on='CustomerID')

# Optional: Encode Gender as a numerical value
customer_data['Gender'] = customer_data['Gender'].map({'Male': 0, 'Female': 1})

# Features for clustering
features = customer_data[['Age', 'TotalPurchaseAmount', 'NumTransactions']]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method to Determine Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.show()

# Apply K-Means with the chosen number of clusters (e.g., 3)
optimal_clusters = 3  # Adjust based on the elbow method plot
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(features_scaled)

# Visualize the clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Age', y='TotalPurchaseAmount', hue='Cluster', data=customer_data, palette='viridis')
plt.title('Customer Segments by Age and Total Purchase Amount')
plt.xlabel('Age')
plt.ylabel('Total Purchase Amount')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
sns.scatterplot(x='NumTransactions', y='TotalPurchaseAmount', hue='Cluster', data=customer_data, palette='viridis')
plt.title('Customer Segments by Number of Transactions and Total Purchase Amount')
plt.xlabel('Number of Transactions')
plt.ylabel('Total Purchase Amount')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Analyze cluster characteristics
cluster_summary = customer_data.groupby('Cluster').agg({
    'Age': 'mean',
    'TotalPurchaseAmount': 'mean',
    'NumTransactions': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'NumCustomers'})

print("\nCluster Summary:")
print(cluster_summary)
