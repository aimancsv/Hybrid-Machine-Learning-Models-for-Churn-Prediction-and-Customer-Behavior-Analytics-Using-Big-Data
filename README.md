# Hybrid-Machine-Learning-Models-for-Churn-Prediction-and-Customer-Behavior-Analytics-Using-Big-Data
By leveraging customer financial activity data, geographic factors, and demographic attributes, the system aims to identify customers at risk of churn and help banks implement retention strategies. Using a structured approach following the CRISP-DM methodology, integrating descriptive analysis, predictive modeling, and interactive dashboards. 


Here’s a detailed and structured **README file** that follows the flow of your report. This will make it easier for you to copy-paste images and sections from the report into your GitHub repository:

---

# **Hybrid Machine Learning Models for Customer Behavior Analytics and Churn Prediction in Banking Using Big Data**

## **Overview**
This project explores customer churn prediction in the banking sector using hybrid machine learning models and big data analytics. By analyzing customer demographics, financial activity patterns, and geographic factors, the system aims to predict customer churn and provide actionable insights for retention strategies. The project is built on the CRISP-DM methodology and integrates descriptive analysis, predictive modeling, and interactive dashboards.

---

## **Table of Contents**
1. [Introduction](#introduction)  
2. [Aim and Objectives](#aim-and-objectives)  
3. [Big Data Analytics Lifecycle](#big-data-analytics-lifecycle)  
4. [Descriptive Analysis](#descriptive-analysis)  
5. [Data Preparation](#data-preparation)  
6. [Modeling and Evaluation](#modeling-and-evaluation)  
7. [Results and Insights](#results-and-insights)  
8. [Future Enhancements](#future-enhancements)

---

## **1. Introduction**
Customer churn, the phenomenon where customers discontinue their relationship with a bank, is a critical challenge in the financial sector. This project uses machine learning and big data techniques to predict churn, leveraging patterns in customer behavior such as demographics, financial transactions, and tenure. The insights generated enable banks to proactively address churn risks and implement personalized retention strategies.

---

## **2. Aim and Objectives**

### **Aim**  
To develop accurate and reliable predictive models for forecasting customer churn in the banking sector.

### **Objectives**
- Analyze the impact of demographics and customer tenure on churn rates.  
- Assess financial activity patterns and their role in predicting churn.  
- Examine the correlation between temporal transaction patterns and churn.  
- Evaluate the influence of customer net worth and geographic factors on churn.  

---

## **3. Big Data Analytics Lifecycle**

### **Methodology**: CRISP-DM Framework
The project follows the six phases of the CRISP-DM framework:  
1. **Business Understanding**: Define churn prediction goals and strategies.  
2. **Data Understanding**: Analyze a dataset with over 28,000 customer records, including demographics, transactions, and financial metrics.
  
    <img width="1000" alt="image" src="https://github.com/user-attachments/assets/96955baa-0dc8-4546-a45c-c687387e8123" />
    
    <img width="350" alt="image" src="https://github.com/user-attachments/assets/441debf3-d28a-4c33-9843-47aef51ba56f" />
    <img width="250" alt="image" src="https://github.com/user-attachments/assets/4adcfabf-e3c4-4f12-a779-cd96bd8f61ee" />
    <img width="350" alt="image" src="https://github.com/user-attachments/assets/59246ad1-4643-4e15-be26-ab471a3d05a6" />
    <img width="350" alt="image" src="https://github.com/user-attachments/assets/835a1e83-b96d-4e8f-899f-d99d2ab6e7d8" />
    <img width="350" alt="image" src="https://github.com/user-attachments/assets/17e21b03-ee79-4558-8a07-9efc74c453dd" />
    <img width="350" alt="image" src="https://github.com/user-attachments/assets/65b206ef-5693-4b11-94d7-cc02e12a3f41" />
    <img width="350" alt="image" src="https://github.com/user-attachments/assets/59da3e33-878b-43c3-9b98-f8992361ed86" />

    ---
3. **Data Preparation**: Clean, preprocess, and engineer features for machine learning.  

    <img width="700" alt="Screenshot 2025-01-03 at 10 10 47 AM" src="https://github.com/user-attachments/assets/c704cfeb-205b-4450-aa8c-214318223ffd" />

    ---
    <img width="500" alt="Screenshot 2025-01-03 at 10 09 57 AM" src="https://github.com/user-attachments/assets/0516e7f2-3591-4ebb-8cc6-92947d91d9d8" />

    ### The missing value is replaced with the mode method, which is the most common method of data imputation. In this method, you replace all the missing data with the mean, median, or        mode of the column.
    ---
   
    <img width="700" alt="Screenshot 2025-01-03 at 10 06 09 AM" src="https://github.com/user-attachments/assets/29bfeff3-c90f-455f-bcca-16f7fe85763f" />

    ---
    <img width="700" alt="Screenshot 2025-01-03 at 10 13 27 AM" src="https://github.com/user-attachments/assets/f9c028d4-7307-4104-abce-6a45b3fc4a14" />

    ---
    <img width="700" alt="Screenshot 2025-01-03 at 10 14 48 AM" src="https://github.com/user-attachments/assets/3fe36888-1c73-4f4e-94db-fa030e629070" />

    ---
    <img width="700" alt="Screenshot 2025-01-03 at 10 21 18 AM" src="https://github.com/user-attachments/assets/09c91310-3123-4da5-8f1f-8a5785f6cd3d" />

    The code snippet above uses the get_dummies function to convert the 'occupation' column into multiple binary (0 or 1) columns, each representing a different occupation category. This        process, known as one-hot encoding, is essential for preparing categorical data for machine learning models, which typically require numerical input.

    ---
    <img width="700" alt="Screenshot 2025-01-03 at 10 18 08 AM" src="https://github.com/user-attachments/assets/5c6ff933-8af7-48b2-b16e-90182a489fe0" />

    ---
    <img width="700" alt="Screenshot 2025-01-03 at 10 19 25 AM" src="https://github.com/user-attachments/assets/c5db50eb-8673-43dd-b5c1-ff80f78ea813" />

    ---
    <img width="700" alt="Screenshot 2025-01-03 at 10 19 42 AM" src="https://github.com/user-attachments/assets/f4ea4f17-50f7-4c1d-83ca-a84caeae44e1" />

    ---
    <img width="700" alt="Screenshot 2025-01-03 at 10 20 20 AM" src="https://github.com/user-attachments/assets/24f197f9-835b-4480-9deb-8c689c7968bc" />

    ---



5. **Modeling**: Develop and evaluate hybrid machine learning models.  
6. **Evaluation**: Assess model performance using metrics like accuracy, precision, recall, and F1-score.  
7. **Deployment**: Provide insights through interactive dashboards.  

---

## **4. Descriptive Analysis**

### **Key Insights**:
1. **Customer Demographics**:
   - Majority of customers are aged 40–45 and maintain low to moderate account balances.
   - 59% of customers are male, while 41% are female.  

   _Example Image_:  
   ![Age Distribution](./images/age_distribution.png)

2. **Financial Activity**:
   - Customers maintain stable average balances between 2,000–4,000 units.
   - Majority of debit and credit transactions are small (below 50 units).  

   _Example Image_:  
   ![Balance Distribution](./images/balance_distribution.png)

3. **Correlation Analysis**:
   - Higher account balances correlate with lower churn risk.
   - Temporal patterns in transactions are weakly correlated with churn.

   _Example Image_:  
   ![Correlation Heatmap](./images/correlation_heatmap.png)

---

## **5. Data Preparation**

### **Key Steps**:
1. **Data Cleaning**:
   - Removed missing values in critical columns like `last_transaction`.
   - Dropped redundant features like `customer_id`.  
   - Handled outliers in financial metrics using interquartile ranges.

   _Example Image_:  
   ![Outlier Removal](./images/outlier_removal.png)

2. **Feature Engineering**:
   - One-hot encoding for categorical variables (e.g., occupation, city).  
   - Binary encoding for gender.  
   - Temporal splitting of `last_transaction` into day, month, and year.

   _Example Image_:  
   ![Feature Engineering](./images/feature_engineering.png)

3. **Final Dataset**:
   - Prepared dataset with 20+ engineered features for model training.  

---

## **6. Modeling and Evaluation**

### **Machine Learning Models**:
1. **Random Forest**:
   - Evaluated financial activity patterns.  
   - Achieved **94% accuracy** for non-churn predictions but struggled with churn detection.

2. **Logistic Regression**:
   - Analyzed geographic and net worth factors.  
   - Highlighted customer segments at higher churn risk.  

3. **Decision Tree Classifier**:
   - Explored transaction patterns and their temporal effects.  

4. **XGBoost**:
   - Focused on demographic and tenure impact.  

### **Evaluation Metrics**:
- Accuracy, Precision, Recall, F1-Score.  
- Confusion matrices highlight areas for improvement in churn prediction.  

_Example Image_:  
![Confusion Matrix](./images/confusion_matrix.png)

---

## **7. Results and Insights**

- **High-Risk Segments**:
   - Customers with low balances and irregular transactions.  
   - Certain geographic regions show higher churn rates.

- **Model Performance**:
   - Random Forest excels in non-churn prediction; requires tuning for churn detection.
   - XGBoost provides robust insights into demographic factors.  

_Example Image_:  
![Model Performance](./images/model_performance.png)

---

## **8. Future Enhancements**
1. Integrate advanced deep learning models like LSTMs for time-series analysis.  
2. Incorporate real-time data for dynamic churn prediction.  
3. Explore unsupervised learning for deeper customer segmentation.  
4. Add multilingual dashboard support for global scalability.


