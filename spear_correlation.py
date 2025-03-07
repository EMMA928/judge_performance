import pandas as pd
from scipy.stats import spearmanr

# Define the function to compute Spearman correlations
def spearman_corr_by_group(group_df):
    leave_one_out = group_df['leave_one_out_avg_all_years']
    predicted = group_df['predicted_citations']
    actual = group_df['citation_count_std']  # Assuming this is the true citation count
    
    # Compute Spearman correlations
    spearman_corr_1, _ = spearmanr(leave_one_out, predicted)  # LOO vs predicted
    spearman_corr_2, _ = spearmanr(actual, predicted)  # Actual vs predicted
    
    return pd.Series({
        'spearman_corr_leave_out_vs_predicted': spearman_corr_1,
        'spearman_corr_actual_vs_predicted': spearman_corr_2
    })

# Function to load data, apply calculations, and return results
def process_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Apply the correlation function to each group based on both 'year' and 'court_normalized'
    spearman_results = data.groupby(['year', 'court_normalized']).apply(spearman_corr_by_group)
    
    # Reset index to include 'year' and 'court_normalized' in the result
    spearman_results.reset_index(inplace=True)
    
    return spearman_results

# List of file paths to all datasets
file_paths = [
    r"C:\Users\lixiang\.ssh\n-gram2\split based on judges' names\output_1970s.csv",
    r"C:\Users\lixiang\.ssh\n-gram2\split based on judges' names\output_1980s.csv",
    r"C:\Users\lixiang\.ssh\n-gram2\split based on judges' names\output_1990s.csv",
    r"C:\Users\lixiang\.ssh\n-gram2\split based on judges' names\output_2000s.csv"
]

# Process each dataset and concatenate results
all_results = pd.concat([process_data(file_path) for file_path in file_paths])

# Save the concatenated results to a new CSV file
all_results.to_csv(r"C:\Users\lixiang\spearman_results_all_decades_split_judges.csv", index=False)

# Print the concatenated results
print(all_results)
