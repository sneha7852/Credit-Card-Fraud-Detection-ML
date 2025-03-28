# Credit-Card-Fraud-Detection-ML

**Overview**  
This project aims to detect fraudulent credit card transactions using machine learning algorithms. The objective is to accurately identify fraudulent transactions from a dataset of credit card transactions while minimizing false positives and false negatives.

**Introduction**  
Credit card fraud is a significant issue in the financial sector, leading to substantial losses each year. This project explores different machine learning techniques to build a model that can effectively identify fraudulent transactions. The project utilizes a dataset that contains transactions labeled as fraudulent or non-fraudulent, allowing for supervised learning.

**Dataset**  
The dataset used in this project is a publicly available credit card transaction dataset. It contains anonymized features due to confidentiality issues and is highly imbalanced, with a very small percentage of transactions being fraudulent.

- **Features:** The dataset includes various anonymized features such as V1, V2, ..., V28, and non-anonymized features like Amount and Time.
- **Target:** The target variable is Class , where 1  represents a fraudulent transaction and 0 represents a non-fraudulent transaction.

**Modeling Approach**  
The project uses several machine learning algorithms, including:

- **Logistic Regression:** A simple linear model to establish a baseline.
- **Random Forest:** An ensemble method to improve accuracy and robustness.
- **Gradient Boosting:** A powerful technique for handling imbalanced datasets.

**Conclusion**
The project successfully demonstrates the use of machine learning techniques to detect fraudulent credit card transactions. Future work may include improving the model's performance through advanced techniques such as deep learning, handling real-time data, and implementing the model in a production environment.
