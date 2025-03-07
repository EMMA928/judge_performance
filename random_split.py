import os
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score


# Directory to save CSV files
output_directory = r"C:/Users/lixiang/.ssh/n-gram"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Preprocessing functions
def preprocess_dataset(file_path):
    # Use 'on_bad_lines' to skip problematic rows)

    df = pd.read_csv(file_path)
    print(df.columns)


    # Drop rows with NaN in important columns
    df = df.dropna(subset=['opinion', 'opinion_type', 'citation_count', 'opinion_songernames'])
    df = df[df['opinion_type'] == 'majority']
    
    # Clean up opinion text
    df['opinion'] = df['opinion'].str.replace('\n', ' ')  
    df['opinion'] = df['opinion'].str.replace(r'^.*?Judge:', '', regex=True)  
    df['opinion'] = df['opinion'].str.replace(r'^.*?Judge.', '', regex=True)  
    df['opinion'] = df['opinion'].str.lstrip()  

    # Standardize citation count within each court-year group
    df["citation_count_std"] = df.groupby(["court_normalized", "year"])["citation_count"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
    )

    # Filter opinion_songernames with more than 1 occurrence
    opinion_songernames_counts = df['opinion_songernames'].value_counts()
    df = df[df['opinion_songernames'].isin(opinion_songernames_counts[opinion_songernames_counts > 1].index)]
    
    return df

def preprocess_text_v2(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s,.!?]', '', text)
    stop_words = set(stopwords.words('english'))
    cleaned_text = " ".join([word for word in text.lower().split() if word not in stop_words])
    return cleaned_text

def leave_one_out_avg_all_years(df):
    def calculate_leave_one_out_for_group(group):
        group = group.copy()
        group["leave_one_out_avg_all_years"] = np.nan
        for idx, row in group.iterrows():
            relevant_cases = df[(df["opinion_songernames"] == row["opinion_songernames"]) & 
                                (df["dc_identifier"] != row["dc_identifier"])]
            leave_out_avg = relevant_cases["citation_count_std"].mean() if not relevant_cases.empty else np.nan
            group.at[idx, "leave_one_out_avg_all_years"] = leave_out_avg
        return group
    return df.groupby("opinion_songernames", group_keys=False).apply(calculate_leave_one_out_for_group)

def train_and_evaluate_model(df, year_range, vectorizer, selected_features):
    # TF-IDF Feature Extraction
    X_tfidf = vectorizer.fit_transform(df['cleaned_opinion'])
    X_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

    # Supervised feature selection: Drop n-grams with low correlation
    selected_features = [col for col in selected_features if col in X_tfidf.columns]
    X_selected = X_tfidf[selected_features]
    y = df['leave_one_out_avg_all_years']
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Train Elastic Net
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
    elastic_net.fit(X_train, y_train)
    y_pred_en = elastic_net.predict(X_test)
    print(f"{year_range} - Elastic Net - MSE: {mean_squared_error(y_test, y_pred_en)}, R^2: {r2_score(y_test, y_pred_en)}")

    # Train Gradient Boosting
    xgb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    print(f"{year_range} - XGBoost - MSE: {mean_squared_error(y_test, y_pred_xgb)}, R^2: {r2_score(y_test, y_pred_xgb)}")

    # Extract cases & judges with highest predicted citations
    df['predicted_citations'] = xgb.predict(X_selected)
    df["predicted_quality"] = abs(df["predicted_LOO_average_citations"] - df["leave_one_out_avg_all_years"])

    # Save results for the decade
    top_cases = df[['opinion', 'predicted_LOO_average_citations']].sort_values(by='predicted_citations', ascending=False).head(10)
    top_judges = df.groupby('opinion_songernames')['predicted_LOO_average_citations'].mean().sort_values(ascending=False).head(10)
    
    # Save results to files
    top_cases.to_csv(os.path.join(output_directory, f"top_cases_{year_range}.csv"), index=False)
    top_judges.to_csv(os.path.join(output_directory, f"top_judges_{year_range}.csv"), index=False)
    df.to_csv(os.path.join(output_directory, f"output_{year_range}.csv"), index=False)

    return df

# Define file paths for all four decades
file_paths = {
    '1970s': "C:/Users/lixiang/.ssh/lexis_cases_opinions_circuit_1970s.csv",
    '1980s': "C:/Users/lixiang/.ssh/lexis_cases_opinions_circuit_1980s.csv",
    '1990s': "C:/Users/lixiang/.ssh/lexis_cases_opinions_circuit_1990s.csv",
    '2000s': "C:/Users/lixiang/.ssh/lexis_cases_opinions_circuit_2000s.csv"
}

# Read and preprocess datasets for each decade
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
selected_features = []

# Process each decade separately
for decade, file_path in file_paths.items():
    print(f"Processing data for the {decade}...")
    df = preprocess_dataset(file_path)
    df['cleaned_opinion'] = df['opinion'].apply(preprocess_text_v2)
    
    # Calculate leave-one-out averages
    df = leave_one_out_avg_all_years(df)

    # Perform TF-IDF and supervised feature selection
    X_tfidf = vectorizer.fit_transform(df['cleaned_opinion'])
    X_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Supervised feature selection: Drop n-grams with low correlation
    for col in X_tfidf.columns:
        correlation, _ = pearsonr(X_tfidf[col], df['leave_one_out_avg_all_years'])
        if abs(correlation) > 0.05:
            selected_features.append(col)
    
    df = train_and_evaluate_model(df, decade, vectorizer, selected_features)
