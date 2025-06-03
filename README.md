🧠 Churn Prediction with Decision Tree – High Accuracy Model (93.83%)
This project aims to predict customer churn using a Decision Tree classification model trained on historical business data. The model identifies which clients are most likely to churn, helping companies proactively retain at-risk customers.

📊 Project Overview
Goal: Predict customer churn with high accuracy using structured business data.

Model Used: Decision Tree Classifier.

Training Data: Historical customer data.

Evaluation Data: Independent dataset from May.

Evaluation Accuracy: 99% overall, 93.83% churn identification accuracy.

🧱 Dataset Structure
The dataset contains 23 columns with customer and transaction attributes such as:

Categorical: Status, Segment, Channel, Sales Force, etc.

Numerical: TPV (Total Payment Volume), Gross Profit, Days Since Last Transaction, etc.

Total samples in evaluation (May) dataset: 1380

🧪 Model Evaluation Results
📈 Classification Report:


Metric	Class 0 (Not Churn)	Class 1 (Churn).

Precision	1.00	0.97

Recall	1.00	0.94

F1-score	1.00	0.96


Overall Accuracy: 99%


Churn Detection Recall: 93.83%


         
🛠️ Preprocessing & Feature Engineering
Preprocessing pipeline aligns categorical and numerical features using encoding and imputation.

Final model input: 252 engineered features derived from the original 23 columns.

Missing values handled via imputation based on training-time logic.

📁 Output Files
predicted_may_churners_dt_M0_excluded.csv: Contains 78 predicted churners.

Correct predictions: 76/81 churners matched to actual names.

Churn recall (name match): 93.83%

📌 Key Features Used by the Model
Top features (post-encoding and preprocessing) include:

TPV M-1, Lucro Bruto M-1, Dias_Desde_Ultima_Transacao, and other derived financial or time-based features.
