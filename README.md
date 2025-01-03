# Hybrid-Machine-Learning-Models-for-Churn-Prediction-and-Customer-Behavior-Analytics-Using-Big-Data
By leveraging customer financial activity data, geographic factors, and demographic attributes, the system aims to identify customers at risk of churn and help banks implement retention strategies. Using a structured approach following the CRISP-DM methodology, integrating descriptive analysis, predictive modeling, and interactive dashboards. 

**[Full Report: 135 Pages](./Full_Report.pdf)**

---



## **Table of Contents**
1. [Introduction](#introduction) 
2. [Aim-and-Objectives](#aim-and-objectives) 
3. [Big-Data-Analytics-Lifecycle](#big-data-analytics-lifecycle) 
4. [Modeling-and-Evaluation](#modeling-and-evaluation) 
5. [Results-and-Insights](#results-and-insights) 
6. [Future-Enhancements](#future-enhancements)

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

    <img width="500" alt="Screenshot 2025-01-03 at 10 10 47 AM" src="https://github.com/user-attachments/assets/c704cfeb-205b-4450-aa8c-214318223ffd" />

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

    ### Outliers:
   
    <img width="600" alt="Screenshot 2025-01-03 at 10 57 35 AM" src="https://github.com/user-attachments/assets/b58c9e58-1f9c-434c-894a-084b57eb185e" />

    ### **The rest of the Outlier is in the full Report**

    ---
    


# 4. **Modeling & Evaluation:**


# 4.1 **Random Forest**:
   - Evaluated financial activity patterns.  
   - Achieved **94% accuracy** for non-churn predictions but struggled with churn detection.


  
### A. Data Distribution 

   ![report](https://github.com/user-attachments/assets/ba05bb97-f1de-4df0-b3bc-c5e942d6e9ca)

   ---

    

### B. Potential Outliers 
  <img width="700" alt="image" src="https://github.com/user-attachments/assets/c8ffc76e-ffe4-47e6-8317-a9f79db670af" />

  ---

### C. Correlation Heatmap of Financial Activity Metrics in Relation to Churn Prediction
  <img width="700" alt="image" src="https://github.com/user-attachments/assets/b1dc4ce0-21ee-4ae0-bd7c-f037f4db826e" />

  ---

### D. Multivariate Regression Analysis of Financial Activity Metrics for Churn Prediction
  <img width="700" alt="Screenshot 2025-01-03 at 11 14 22 AM" src="https://github.com/user-attachments/assets/abcef536-4a6e-4897-94e2-0ae8e9303af7" />

  ---
    
### E. OLAP Dashboard for Financial Metrics Distribution
   <img width="350" alt="Screenshot 2025-01-03 at 11 15 29 AM" src="https://github.com/user-attachments/assets/5dc4a553-4c7d-4aa6-93be-de95a66500d0" />
   <img width="350" alt="Screenshot 2025-01-03 at 11 16 11 AM" src="https://github.com/user-attachments/assets/97cef3f3-ea08-4c4b-a57f-c9c9d572c8d8" />
   <img width="350" alt="Screenshot 2025-01-03 at 11 16 44 AM" src="https://github.com/user-attachments/assets/7ae692e6-85e3-4d7c-8e8e-15fa87d5b2d9" />
   <img width="350" alt="Screenshot 2025-01-03 at 11 16 44 AM" src="https://github.com/user-attachments/assets/7ae692e6-85e3-4d7c-8e8e-15fa87d5b2d9" />
   <img width="350" alt="Screenshot 2025-01-03 at 11 17 29 AM" src="https://github.com/user-attachments/assets/ae95aadb-ef5e-43a3-a648-119a02a73da1" />
   <img width="350" alt="Screenshot 2025-01-03 at 11 17 48 AM" src="https://github.com/user-attachments/assets/b660a5a8-2d67-4c5f-a48c-9c7b1a041b44" /> 
   <img width="350" alt="Screenshot 2025-01-03 at 11 18 21 AM" src="https://github.com/user-attachments/assets/d11faacf-c552-4c9a-901b-317e8ff2b9a3" />
   <img width="350" alt="Screenshot 2025-01-03 at 11 18 56 AM" src="https://github.com/user-attachments/assets/b32e63d8-f5b5-44bb-b101-52065b4ab0fd" />

   ---


### F. Predictive Analysis – Random Forest Classifier
   <img width="700" alt="Screenshot 2025-01-03 at 11 31 20 AM" src="https://github.com/user-attachments/assets/9b26bea2-61b0-4e03-b11f-fbb685a0b453" />

   ---


   <img width="700" alt="Screenshot 2025-01-03 at 11 31 47 AM" src="https://github.com/user-attachments/assets/e5e464b1-0082-4b21-8445-c579394c5a0d" />

   ---

   <img width="700" alt="Screenshot 2025-01-03 at 11 32 21 AM" src="https://github.com/user-attachments/assets/04cfc633-e876-4c46-8a0f-75cdf7918047" />

   ---

   <img width="700" alt="Screenshot 2025-01-03 at 11 33 04 AM" src="https://github.com/user-attachments/assets/f105dbd0-8e8b-45fc-abbc-98f93495a8f7" />

   ---


   <img width="700" alt="Screenshot 2025-01-03 at 11 34 08 AM" src="https://github.com/user-attachments/assets/2de8a14c-ceb7-4bf2-8390-64052bde0130" />

   ---


   <img width="700" alt="Screenshot 2025-01-03 at 11 35 19 AM" src="https://github.com/user-attachments/assets/8d62bdcb-f132-4c81-a37e-4c95a29bafd8" />

   ---


   <img width="700" alt="Screenshot 2025-01-03 at 11 35 36 AM" src="https://github.com/user-attachments/assets/7a4ffe33-953c-495a-a77d-1224faca5dc8" />


   ---

# 4.2 **Logistic Regression**:
   - Analyzed geographic and net worth factors.  
   - Highlighted customer segments at higher churn risk.
   - To evaluate the influence of customer net worth and geographic factors on churn
  
**<span style="color:red; font-size: 1.2em;">Details in the **[Full Report](./Full_Report.pdf)**</span>**

---

# 4.3 **Decision Tree Classifier**:
   - Explored transaction patterns and their temporal effects.
   - To examine the correlation between temporal transaction patterns and churn
  
**<span style="color:red; font-size: 1.2em;">Details in the **[Full Report](./Full_Report.pdf)**</span>**

  ---
# 4.4 **XGBoost**:
   - Focused on demographic and tenure impact. 
  

**<span style="color:red; font-size: 1.2em;">Details in the **[Full Report](./Full_Report.pdf)**</span>**

---

## **5. Results and Insights**

  The results indicate that **Random Forest** performed exceptionally well in predicting non-churn customers with an accuracy of **94%**, while models like **Logistic Regression** and         **XGBoost** offered valuable insights into the relationship between demographics, geographic factors, and customer tenure. However, a common challenge across the models was the lower        precision in detecting churn cases, highlighting the need for further refinement in feature engineering and algorithm selection.

Key findings include:  
- **High-Balance Customers**: These customers are significantly less likely to churn, suggesting that financial stability plays a critical role in retention.  
- **Geographic and Demographic Influence**: Certain regions and age groups exhibit higher churn rates, enabling banks to design targeted retention strategies.  
- **Transaction Patterns**: Customers with irregular or infrequent transactions showed higher churn risks, emphasizing the importance of monitoring financial activity.

---

## **6. Future Enhancements**
1. Integrate advanced deep learning models like LSTMs for time-series analysis.  
2. Incorporate real-time data for dynamic churn prediction.  
3. Explore unsupervised learning for deeper customer segmentation.  
4. Add multilingual dashboard support for global scalability.


