import pandas as pd

# Function to clean currency strings and convert to float
def clean_currency(value):
    if isinstance(value, str):
        # Remove R$, strip whitespace, replace . for thousands and , for decimal
        value = value.replace('R$', '').strip().replace('.', '').replace(',', '.')
    try:
        return float(value)
    except (ValueError, TypeError):
        return None # Or handle error as appropriate, e.g., np.nan

def main():
    print("Loading '31_05.csv' dataset...")
    try:
        df_may = pd.read_csv('31_05.csv')
    except FileNotFoundError:
        print("Error: '31_05.csv' not found. Please ensure the file is in the correct directory.")
        return

    print("Cleaning currency columns...")
    # Identify potential currency columns (heuristic)
    currency_columns_may = []
    for col in df_may.columns:
        sample_values = df_may[col].dropna().unique()
        is_currency = False
        for sample in sample_values[:5]: # Check first 5 unique non-null values
            if isinstance(sample, str) and 'R$' in sample:
                is_currency = True
                break
        if is_currency:
            currency_columns_may.append(col)
    
    if currency_columns_may:
        print(f"Identified currency columns for conversion in 31_05.csv: {currency_columns_may}")
        for col in currency_columns_may:
            df_may[col] = df_may[col].apply(clean_currency)
    else:
        print("No currency columns automatically identified for conversion in 31_05.csv based on 'R$' prefix.")
        # Attempt to convert TPV M-1 and TPV M-0 anyway if they exist and are object type
        for col_name in ['TPV M-1', 'TPV M-0']:
            if col_name in df_may.columns and df_may[col_name].dtype == 'object':
                print(f"Attempting direct currency conversion for '{col_name}'.")
                df_may[col_name] = df_may[col_name].apply(clean_currency)

    # Ensure TPV M-1 and TPV M-0 exist before applying churn logic
    if 'TPV M-1' not in df_may.columns or 'TPV M-0' not in df_may.columns:
        print("Error: 'TPV M-1' or 'TPV M-0' not found in the dataset. Cannot apply churn logic.")
        return

    print("Identifying churned clients for May...")
    # Churn logic: TPV M-1 != 0 and TPV M-0 == 0
    # Ensure TPV values are not NaN when applying this logic.
    churn_condition_may = (
        df_may['TPV M-1'].notna() & (df_may['TPV M-1'] != 0) & 
        df_may['TPV M-0'].notna() & (df_may['TPV M-0'] == 0)
    )
    may_churned_clients = df_may[churn_condition_may]

    num_churned = len(may_churned_clients)
    print(f"Found {num_churned} clients churned in May.")

    if num_churned > 0:
        output_filename = 'may_churned.csv'
        print(f"Saving churned clients to '{output_filename}'...")
        may_churned_clients.to_csv(output_filename, index=False)
        print(f"Successfully saved to '{output_filename}'.")
    else:
        print("No churned clients to save.")

if __name__ == '__main__':
    main() 