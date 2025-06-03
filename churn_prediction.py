import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib # Added joblib for saving models and preprocessor
import shap # Added SHAP
import matplotlib.pyplot as plt # Added for saving SHAP plots
import numpy as np # Added for array operations

# Function to clean currency strings and convert to float
def clean_currency(value):
    if isinstance(value, str):
        value = value.replace('R$', '').strip().replace('.', '').replace(',', '.')
    try:
        return float(value)
    except ValueError:
        return None

# Load the dataset
df = pd.read_csv('02_04.csv')

# --- Date Processing ---
date_cols = ['Data de Credenciamento', 'Data de Ativação', 'Data da Última Transação']
# print("\n--- Attempting Date Conversion (format='%d/%m/%y', errors='coerce') ---") # Less verbose now
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], format='%d/%m/%y', errors='coerce')

# ... (Currency conversion - applying to all relevant columns)
currency_columns = []
for col in df.columns:
    sample_values = df[col].dropna().unique()
    is_currency = False
    for sample in sample_values[:5]:
        if isinstance(sample, str) and 'R$' in sample:
            is_currency = True
            break
    if is_currency:
        currency_columns.append(col)

if currency_columns:
    # print(f"\nIdentified currency columns for conversion: {currency_columns}") # Less verbose
    for col in currency_columns:
        df[col] = df[col].apply(clean_currency)
# else:
    # print("\nNo currency columns automatically identified for conversion.")


# --- Define Churn Target Variable ---
# A client is churned if TPV M-1 != 0 and TPV M-0 == 0.
# Ensure TPV values are not NaN when applying this logic.
df['Churn'] = ((df['TPV M-1'].notna()) & (df['TPV M-1'] != 0) & \
               (df['TPV M-0'].notna()) & (df['TPV M-0'] == 0)).astype(int)

print("\nChurn definition applied. Value counts for 'Churn':")
print(df['Churn'].value_counts())

# --- Date Feature Engineering (Re-instated) ---
# (This should happen on df before X is defined, and after date columns are datetime)
if not df['Data de Credenciamento'].isnull().all():
    ref_credenciamento = df['Data de Credenciamento'].max() + pd.Timedelta(days=1)
    df['Dias_Credenciado'] = (ref_credenciamento - df['Data de Credenciamento']).dt.days
else:
    df['Dias_Credenciado'] = None

if not df['Data de Ativação'].isnull().all():
    ref_ativacao = df['Data de Ativação'].max() + pd.Timedelta(days=1)
    df['Dias_Desde_Ativacao'] = (ref_ativacao - df['Data de Ativação']).dt.days
else:
    df['Dias_Desde_Ativacao'] = None

if not df['Data da Última Transação'].isnull().all():
    ref_ultima_transacao = df['Data da Última Transação'].max() + pd.Timedelta(days=1)
    df['Dias_Desde_Ultima_Transacao'] = (ref_ultima_transacao - df['Data da Última Transação']).dt.days
else:
    df['Dias_Desde_Ultima_Transacao'] = None

df['Tempo_Entre_Credenciamento_Ativacao'] = (df['Data de Ativação'] - df['Data de Credenciamento']).dt.days


# Define new target variable y
y = df['Churn'].copy()

# Define columns to drop to create feature set X
imposto_custo_cols = [col for col in df.columns if 'imposto' in col.lower() or 'custo' in col.lower()]

# CRITICAL CHANGE: Identify and drop ALL M-0 columns from training features
m0_training_cols_to_drop = [col for col in df.columns if 'M-0' in col.upper()] # Using .upper() for safety, though M-0 is specific
print(f"\nIdentified M-0 columns to drop from TRAINING features: {m0_training_cols_to_drop}")

base_columns_to_drop_for_X = ['Afiliações Consideradas', 'Nome Fantasia', 'Churn'] 
# TPV M-0 is already in m0_training_cols_to_drop if present

columns_to_drop_for_X = list(set(base_columns_to_drop_for_X + imposto_custo_cols + date_cols + m0_training_cols_to_drop))

print(f"\nFinal columns to be dropped for X (Churn target, ALL M-0 columns excluded from features): {sorted(columns_to_drop_for_X)}")

X = df.drop(columns=columns_to_drop_for_X, errors='ignore').copy() # Added errors='ignore' for safety
print("\nInfo for feature set X (ALL M-0 columns EXCLUDED from features):")
print(X.info())

# --- Preprocessing ---
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# (Pipelines and ColumnTransformer setup as before)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save the training column order *before* processing, as preprocessor will be applied to these raw features.
# However, it's often better to save feature names *after* one-hot encoding if you reconstruct DataFrames later.
# For applying the preprocessor, the raw X_train.columns are what matter for ColumnTransformer lookup.
joblib.dump(X_train.columns.tolist(), 'X_train_columns.joblib')
print(f"\nTraining feature column names/order saved to X_train_columns.joblib. Count: {len(X_train.columns)}")

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

joblib.dump(preprocessor, 'preprocessor.joblib')
print("\nFitted preprocessor saved to preprocessor.joblib")

# Get feature names after preprocessing for SHAP plots and native importance
processed_feature_names = None
try:
    ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    processed_feature_names = list(numerical_features) + list(ohe_feature_names)
except Exception as e:
    print(f"Warning: Could not retrieve detailed feature names for SHAP/Importances. Error: {e}")

print(f"\nShape of X_train_processed (NumPy array): {X_train_processed.shape}")
print(f"Shape of X_test_processed (NumPy array): {X_test_processed.shape}")

# --- Model Training & Saving ---

# --- Logistic Regression ---
print("\n--- Training Logistic Regression Model ---")
log_reg_model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
log_reg_model.fit(X_train_processed, y_train)

y_pred_train_log_reg = log_reg_model.predict(X_train_processed)
y_pred_test_log_reg = log_reg_model.predict(X_test_processed)

print("\nLogistic Regression - Classification Report (Training Data):")
print(classification_report(y_train, y_pred_train_log_reg))
print("\nLogistic Regression - Classification Report (Test Data):")
print(classification_report(y_test, y_pred_test_log_reg))
cm_test_log_reg = confusion_matrix(y_test, y_pred_test_log_reg)
print("\nLogistic Regression - Confusion Matrix (Test Data):")
print(cm_test_log_reg)

# --- Decision Tree Classifier ---
print("\n--- Training Decision Tree Classifier Model ---")
dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
dt_model.fit(X_train_processed, y_train)
joblib.dump(dt_model, 'decision_tree_model.joblib')
print("Decision Tree model saved to decision_tree_model.joblib")

y_pred_train_dt = dt_model.predict(X_train_processed)
y_pred_test_dt = dt_model.predict(X_test_processed)

print("\nDecision Tree - Classification Report (Training Data):")
print(classification_report(y_train, y_pred_train_dt))
print("\nDecision Tree - Classification Report (Test Data):")
print(classification_report(y_test, y_pred_test_dt))
cm_test_dt = confusion_matrix(y_test, y_pred_test_dt)
print("\nDecision Tree - Confusion Matrix (Test Data):")
print(cm_test_dt)

# --- SHAP Analysis for Decision Tree Model (Modified from RF to DT) ---
print("\n--- Generating SHAP Analysis for Decision Tree Model ---")
try:
    # For DecisionTree, SHAP TreeExplainer works directly.
    explainer_dt = shap.TreeExplainer(dt_model)
    shap_values_dt_train = explainer_dt.shap_values(X_train_processed)

    # print(f"Shape of X_train_processed: {X_train_processed.shape}") # Less verbose
    if isinstance(shap_values_dt_train, list) and len(shap_values_dt_train) == 2:
        # print(f"SHAP values are a list of 2 arrays (binary classification).") # Less verbose
        # print(f"Shape of shap_values_dt_train[0] (class 0): {shap_values_dt_train[0].shape}") # Less verbose
        # print(f"Shape of shap_values_dt_train[1] (class 1): {shap_values_dt_train[1].shape}") # Less verbose
        shap_values_for_plot = shap_values_dt_train[1] # Use SHAP values for the positive class (Churn=1)
    else:
        # This case might occur if the output is not as expected for DT
        print(f"SHAP values for DT not in expected list-of-2-arrays. Shape: {np.array(shap_values_dt_train).shape}")
        shap_values_for_plot = shap_values_dt_train # Use as is, might need adjustment

    if processed_feature_names and shap_values_for_plot.shape[1] != len(processed_feature_names):
        print(f"Warning: Mismatch between SHAP values columns ({shap_values_for_plot.shape[1]}) and processed_feature_names ({len(processed_feature_names)}). SHAP plots might be misleading.")

    print("Generating SHAP summary plot (beeswarm) for Decision Tree...")
    plt.figure()
    shap.summary_plot(shap_values_for_plot, X_train_processed, feature_names=processed_feature_names, show=False)
    plt.savefig('shap_summary_plot_dt.png', bbox_inches='tight') # Changed filename to _dt
    plt.close()
    print("SHAP summary plot (beeswarm) for Decision Tree saved to shap_summary_plot_dt.png")

    print("Generating SHAP bar plot (global feature importance) for Decision Tree...")
    plt.figure()
    shap.summary_plot(shap_values_for_plot, X_train_processed, feature_names=processed_feature_names, plot_type="bar", show=False)
    plt.savefig('shap_bar_plot_dt.png', bbox_inches='tight') # Changed filename to _dt
    plt.close()
    print("SHAP bar plot for Decision Tree saved to shap_bar_plot_dt.png")

except Exception as e:
    print(f"Error during SHAP analysis for Decision Tree: {e}")
    import traceback
    print(traceback.format_exc())

# --- Native Feature Importance from Decision Tree Model (Modified from RF to DT) ---
print("\n--- Generating Native Feature Importance Plot for Decision Tree Model ---")
try:
    if processed_feature_names is not None and hasattr(dt_model, 'feature_importances_'):
        importances = dt_model.feature_importances_
        feature_importance_series = pd.Series(importances, index=processed_feature_names)
        sorted_importances = feature_importance_series.sort_values(ascending=False)
        top_n = 20
        top_n_importances = sorted_importances.head(top_n)
        
        plt.figure(figsize=(10, max(6, top_n * 0.3)))
        top_n_importances.sort_values(ascending=True).plot(kind='barh')
        plt.title(f'Top {top_n} Feature Importances (Decision Tree)') # Changed title
        plt.xlabel('Importance (Gini/Information Gain)') # Generic for DT
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('dt_native_feature_importance.png') # Changed filename to _dt
        plt.close()
        print(f"Decision Tree native feature importance plot saved to dt_native_feature_importance.png")
        print(f"\nTop {top_n} features (Decision Tree):")
        print(top_n_importances)
    else:
        print("Could not generate native DT feature importance: names or importances not available.")
except Exception as e:
    print(f"Error during native feature importance generation for Decision Tree: {e}")
    import traceback
    print(traceback.format_exc())

print("\nChurn prediction script complete (focused on Decision Tree).")
