import pandas as pd
import numpy as np # For np.nan if needed, though clean_currency returns None
import joblib # For loading models and preprocessor
from sklearn.metrics import classification_report, confusion_matrix # For evaluation
import traceback # Make sure it's imported for the except block above

# Function to clean currency strings (should be robust)
def clean_currency(value):
    if isinstance(value, str):
        value = value.replace('R$', '').strip().replace('.', '').replace(',', '.')
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def load_and_prepare_evaluation_data(file_path='31_05.csv'):
    """Loads, cleans, and prepares the May dataset for evaluation."""
    print(f"Loading dataset from '{file_path}'...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.")
        return None, None, None # Return three Nones if df fails to load

    print("Cleaning currency columns for May data (Robust Method)...")
    currency_columns_to_clean = []
    for col in df.columns:
        # Check a sample of unique non-null values for the 'R$' string
        sample_values = df[col].dropna().unique()
        is_currency_col = False
        # Iterate through a small sample (e.g., first 5 unique non-nulls)
        for sample in sample_values[:min(len(sample_values), 5)]:
            if isinstance(sample, str) and 'R$' in sample:
                is_currency_col = True
                break
        if is_currency_col:
            currency_columns_to_clean.append(col)
    
    if currency_columns_to_clean:
        print(f"Identified currency columns for conversion: {currency_columns_to_clean}")
        for col_to_clean in currency_columns_to_clean:
            df[col_to_clean] = df[col_to_clean].apply(clean_currency)
    else:
        print("No currency columns automatically identified for conversion based on 'R$' prefix.")
        # Fallback: Attempt to convert specific TPV columns if they are objects and weren't caught
        # This helps if R$ isn't present but they are known to be currency-like strings
        for col_name in ['TPV M-1', 'TPV M-0']:
            if col_name in df.columns and df[col_name].dtype == 'object':
                print(f"Attempting fallback currency conversion for '{col_name}'.")
                df[col_name] = df[col_name].apply(clean_currency)

    # --- Date Conversion and Feature Engineering (replicating from churn_prediction.py) ---
    print("Performing date conversion and feature engineering for May data...")
    date_cols = ['Data de Credenciamento', 'Data de Ativação', 'Data da Última Transação']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%d/%m/%y', errors='coerce')
        else:
            print(f"Warning: Date column '{col}' not found in May data for feature engineering.")

    if 'Data de Credenciamento' in df.columns and not df['Data de Credenciamento'].isnull().all():
        ref_credenciamento = df['Data de Credenciamento'].max() + pd.Timedelta(days=1)
        df['Dias_Credenciado'] = (ref_credenciamento - df['Data de Credenciamento']).dt.days
    else:
        df['Dias_Credenciado'] = np.nan

    if 'Data de Ativação' in df.columns and not df['Data de Ativação'].isnull().all():
        ref_ativacao = df['Data de Ativação'].max() + pd.Timedelta(days=1)
        df['Dias_Desde_Ativacao'] = (ref_ativacao - df['Data de Ativação']).dt.days
    else:
        df['Dias_Desde_Ativacao'] = np.nan
        
    if 'Data da Última Transação' in df.columns and not df['Data da Última Transação'].isnull().all():
        ref_ultima_transacao = df['Data da Última Transação'].max() + pd.Timedelta(days=1)
        df['Dias_Desde_Ultima_Transacao'] = (ref_ultima_transacao - df['Data da Última Transação']).dt.days
    else:
        df['Dias_Desde_Ultima_Transacao'] = np.nan

    if 'Data de Ativação' in df.columns and 'Data de Credenciamento' in df.columns:
        df['Tempo_Entre_Credenciamento_Ativacao'] = (df['Data de Ativação'] - df['Data de Credenciamento']).dt.days
    else:
        df['Tempo_Entre_Credenciamento_Ativacao'] = np.nan
        
    # --- Define True Churn for May (for evaluation) ---
    print("Defining actual churn for May data (y_may_true)...")
    if 'TPV M-1' in df.columns and 'TPV M-0' in df.columns:
        df['Actual_May_Churn'] = ((df['TPV M-1'].notna()) & (df['TPV M-1'] != 0) & \
                                   (df['TPV M-0'].notna()) & (df['TPV M-0'] == 0)).astype(int)
        y_may_true = df['Actual_May_Churn'].copy()
        print("Actual May Churn defined. Value counts:")
        print(y_may_true.value_counts())
    else:
        print("Error: 'TPV M-1' or 'TPV M-0' not found in May data. Cannot define actual churn.")
        y_may_true = None
        df['Actual_May_Churn'] = pd.Series(dtype='int') # Ensure column exists for consistent dropping

    # --- Prepare Feature Set X_may_eval ---
    print("Preparing feature set X_may_eval (ALL M-0 columns, including TPV M-0, will be dropped from initial features)...")
    imposto_custo_cols = [col for col in df.columns if 'imposto' in col.lower() or 'custo' in col.lower()]
    
    # Identify ALL M-0 columns to be dropped from features for this specific evaluation run
    m0_cols_to_drop = [col for col in df.columns if 'M-0' in col]
    print(f"Identified M-0 columns to drop from X_may_eval features: {m0_cols_to_drop}")

    cols_to_exclude_from_X_eval = ['Afiliações Consideradas', 'Nome Fantasia', 'Actual_May_Churn'] 
    # TPV M-0 is already in m0_cols_to_drop if present, so no need to add it again explicitly to cols_to_exclude_from_X_eval if m0_cols_to_drop is used

    columns_to_drop_for_X_eval = list(set(cols_to_exclude_from_X_eval + imposto_custo_cols + date_cols + m0_cols_to_drop))
    print(f"Final columns to be dropped to create initial X_may_eval: {sorted(columns_to_drop_for_X_eval)}")
    
    X_may_eval = df.drop(columns=columns_to_drop_for_X_eval, errors='ignore').copy()
    
    # It is CRITICAL that X_may_eval has the same columns in the same order as X_train
    # that the preprocessor was fit on. We may need to load X_train column names from the 
    # churn_prediction script or save them, then reindex X_may_eval.
    # For now, we assume the dropping logic results in compatible features. This is a common pitfall.

    print("\nInfo for X_may_eval (features for May data):")
    print(X_may_eval.info())
    return df, X_may_eval, y_may_true

if __name__ == '__main__':
    original_df_may, X_may_eval, y_may_true = load_and_prepare_evaluation_data()
    predicted_churner_names_dt_list = [] # For Decision Tree

    if X_may_eval is not None and y_may_true is not None:
        print("\nLoading preprocessor, Decision Tree model, and training feature list...")
        try:
            preprocessor = joblib.load('preprocessor.joblib')
            dt_model = joblib.load('decision_tree_model.joblib') # Load DT model
            expected_training_columns = joblib.load('X_train_columns.joblib')
            print("Preprocessor, DT model, and training column list loaded successfully.")

            # Display Top N Feature Importances from the loaded Decision Tree model
            print("\n--- Top Feature Importances (from TRAINED Decision Tree Model) ---")
            if hasattr(dt_model, 'feature_importances_') and expected_training_columns:
                importances_dt = dt_model.feature_importances_
                if len(importances_dt) == len(expected_training_columns):
                    feature_importance_series_dt = pd.Series(importances_dt, index=expected_training_columns)
                    sorted_importances_dt = feature_importance_series_dt.sort_values(ascending=False)
                    top_n = 20
                    print(f"Top {top_n} features (Decision Tree):")
                    print(sorted_importances_dt.head(top_n))
                else:
                    print(f"Mismatch in lengths: dt.feature_importances_ ({len(importances_dt)}) vs expected_training_columns ({len(expected_training_columns)})")     
            else:
                print("Could not display Decision Tree feature importances: model or training column list missing attributes.")
            
            print(f"Number of features expected by preprocessor (from training): {len(expected_training_columns)}")
            print(f"Number of features in X_may_eval before alignment: {X_may_eval.shape[1]}")

            X_may_eval = X_may_eval.reindex(columns=expected_training_columns, fill_value=np.nan)
            print(f"Number of features in X_may_eval after alignment (M-0 cols become NaN if not present): {X_may_eval.shape[1]}")
            
            if X_may_eval.isnull().all().any():
                all_nan_cols = X_may_eval.columns[X_may_eval.isnull().all()].tolist()
                print(f"Warning: Columns that are all NaN after reindexing in X_may_eval: {all_nan_cols}")

            print("\nApplying preprocessing to May evaluation data...")
            X_may_eval_processed = preprocessor.transform(X_may_eval)
            print(f"Shape of X_may_eval_processed: {X_may_eval_processed.shape}")

            # --- Decision Tree Predictions & Evaluation ---
            print("\n--- Decision Tree Evaluation on May Data (M-0 features excluded/imputed) ---")
            dt_may_predictions = dt_model.predict(X_may_eval_processed)
            print("Decision Tree - Classification Report (May Data):")
            print(classification_report(y_may_true, dt_may_predictions))
            print("Decision Tree - Confusion Matrix (May Data):")
            print(confusion_matrix(y_may_true, dt_may_predictions))

            # Save Predicted Churners' Nome Fantasia (Decision Tree)
            predicted_churn_mask_dt = (dt_may_predictions == 1)
            predicted_churn_indices_dt = X_may_eval.index[predicted_churn_mask_dt]
            if 'Nome Fantasia' in original_df_may.columns:
                predicted_churner_names_dt_series = original_df_may.loc[predicted_churn_indices_dt, 'Nome Fantasia']
                predicted_churner_names_dt_list = predicted_churner_names_dt_series.tolist()
                output_filename_dt = 'predicted_may_churners_dt_M0_excluded.csv'
                if not predicted_churner_names_dt_series.empty:
                    predicted_churner_names_dt_series.to_csv(output_filename_dt, index=False, header=['Nome Fantasia'])
                    print(f"Saving {len(predicted_churner_names_dt_series)} DT predicted churners to {output_filename_dt}")
                else:
                    print("Decision Tree model predicted no clients will churn.")
            else:
                print("Warning: 'Nome Fantasia' column not found. Cannot save DT predicted churner names.")

        except FileNotFoundError as e:
            print(f"Error loading files: {e}")
        except Exception as e:
            print(f"An error occurred: {e}\n{traceback.format_exc()}") # Added traceback
    else:
        print("Halting evaluation due to issues in data loading/preparation.")

    # --- Compare Predicted DT Churner Names with Actual Churner Names ---
    print("\n--- Comparing DT Predicted Churner Names with Actual Churners from may_churned.csv ---")
    actual_churners_file = 'may_churned.csv'
    try:
        df_actual_churners = pd.read_csv(actual_churners_file)
        if 'Nome Fantasia' in df_actual_churners.columns:
            actual_churner_names_set = set(df_actual_churners['Nome Fantasia'].dropna().unique())
            predicted_churner_names_set_dt = set(predicted_churner_names_dt_list)

            print(f"Number of actual churners (from {actual_churners_file}): {len(actual_churner_names_set)}")
            print(f"Number of predicted churners by DT model: {len(predicted_churner_names_set_dt)}")

            correctly_identified_churners_dt_set = actual_churner_names_set.intersection(predicted_churner_names_set_dt)
            num_correctly_identified_dt = len(correctly_identified_churners_dt_set)
            print(f"Number of churners correctly identified by DT model (name match): {num_correctly_identified_dt}")

            if len(actual_churner_names_set) > 0:
                percentage_correct_dt = (num_correctly_identified_dt / len(actual_churner_names_set)) * 100
                print(f"Percentage of actual churners correctly identified by DT model: {percentage_correct_dt:.2f}%")
            else:
                print("No actual churners found in may_churned.csv to compare against.")
            
            if num_correctly_identified_dt > 0:
                print("\nCorrectly identified churners (by Nome Fantasia):")
                for name in sorted(list(correctly_identified_churners_dt_set)):
                    print(name)
        else:
            print(f"Error: 'Nome Fantasia' column not found in '{actual_churners_file}'. Cannot perform name comparison.")

    except FileNotFoundError:
        print(f"Error: Actual churners file '{actual_churners_file}' not found. Please ensure it was generated by 'generate_may_churn_report.py'.")
    except Exception as e:
        print(f"An error occurred during the name comparison: {e}\n{traceback.format_exc()}") 