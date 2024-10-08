# Customer Segmentation Analysis for Boswell Book Company

## Project Overview
This project involves conducting a customer segmentation analysis for Boswell Book Company, a local bookstore in Milwaukee. The goal is to identify key customer segments based on purchasing behavior to help the company improve customer targeting and retention strategies.

## Tools Used
- Python
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib
- Tableau

## Dataset
The dataset includes customer and transaction data provided by Boswell Book Company. Sensitive information such as names, addresses, and emails has been excluded to ensure privacy.

### Customer Data
- `CustomerID`: Unique identifier for each customer.
- `Age`: Customer's age.
- `Gender`: Customer's gender.
- `JoinDate`: The date the customer joined the loyalty program or made their first purchase.

### Transaction Data
- `TransactionID`: Unique identifier for each transaction.
- `CustomerID`: Identifier linking the transaction to a specific customer.
- `BookTitle`: Title of the book purchased.
- `Genre`: Genre of the book purchased.
- `PurchaseAmount`: Amount spent on the purchase.
- `PurchaseDate`: Date of the purchase.

## Analysis Steps
1. **Data Cleaning and Preprocessing:** 
   - Checked for missing values and ensured data consistency across customer and transaction datasets.
   
2. **Exploratory Data Analysis (EDA):**
   - Analyzed the distribution of customer demographics (age, gender) and purchasing behavior (purchase amount, genre).
   - Generated visualizations to identify patterns and trends in the data.

3. **Customer Segmentation using Clustering Algorithms:**
   - Applied K-means clustering to segment customers based on their age, total purchase amount, and number of transactions.
   - Used the elbow method to determine the optimal number of clusters.

4. **Visualization of Customer Segments and Insights:**
   - Created scatter plots and bar charts to visualize the distinct customer segments and their characteristics.
   - Analyzed the characteristics of each segment to provide actionable recommendations.

## Analysis
This section details the insights gained from the Exploratory Data Analysis (EDA) and Customer Segmentation:

### 1. **Age Distribution of Customers:**
   - The age distribution of customers is relatively diverse, with a mix of young and older customers. However, certain age groups are more prominent, indicating potential target demographics for marketing efforts.

### 2. **Gender Distribution:**
   - The gender distribution shows a slight skew towards female customers. This information can be leveraged to tailor marketing campaigns and inventory to better suit the preferences of the dominant customer base.

### 3. **Purchase Behavior by Genre:**
   - Analysis of the purchase data revealed that certain genres, such as Fiction and Fantasy, are more popular among customers. This insight suggests a focus on these genres for promotions and stocking decisions.

### 4. **Customer Segments Identified:**
   - **Cluster 1:** High-value customers who frequently purchase books and have a higher total purchase amount. These customers are likely loyal and should be targeted with special offers and loyalty programs.
   - **Cluster 2:** Mid-range customers who make steady purchases. These customers might benefit from targeted upsell campaigns.
   - **Cluster 3:** Low-frequency customers with minimal purchases. These customers may require re-engagement strategies to increase their buying frequency.

### 5. **Recommendations:**
   - **Targeted Marketing:** Focus on the high-value customers (Cluster 1) with exclusive offers and personalized recommendations.
   - **Inventory Management:** Prioritize stocking and promoting popular genres like Fiction and Fantasy.
   - **Re-engagement Campaigns:** Develop campaigns to re-engage low-frequency customers and encourage more frequent purchases.
