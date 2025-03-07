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

# Ensure output directory exists
output_directory = r"C:/Users/lixiang/.ssh/n-gram/paragraph_rankings"
os.makedirs(output_directory, exist_ok=True)

def preprocess_text_v2(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s,.!?]', '', text)
    stop_words = set(stopwords.words('english'))
    cleaned_text = " ".join([word for word in text.lower().split() if word not in stop_words])
    return cleaned_text

def split_opinion_into_paragraphs(opinion):
    """
    Split opinion text into paragraphs, filter out very short paragraphs
    """
    paragraphs = [p.strip() for p in opinion.split('\n\n') if len(p.strip()) > 50]
    return paragraphs

def rank_paragraphs_by_citations(file_path, decade, sample_size=500):
    """
    Rank paragraphs by predicted citations for a specific decade
    
    Parameters:
    - file_path: Path to the CSV file
    - decade: Decade being processed
    - sample_size: Number of cases to sample
    
    Returns:
    - DataFrame with paragraphs ranked by predicted citations
    """
    # Read and preprocess dataset
    df = pd.read_csv(file_path)
    
    # Drop rows with NaN in important columns
    df = df.dropna(subset=['opinion', 'opinion_type', 'citation_count', 'opinion_songernames'])
    df = df[df['opinion_type'] == 'majority']
    
    # Clean up opinion text
    df['opinion'] = df['opinion'].str.replace('\n', ' ')
    df['opinion'] = df['opinion'].str.replace(r'^.*?Judge:', '', regex=True)
    df['opinion'] = df['opinion'].str.replace(r'^.*?Judge.', '', regex=True)
    df['opinion'] = df['opinion'].str.lstrip()
    
    # Sample cases
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    # Extract paragraphs
    paragraph_data = []
    for _, row in df_sample.iterrows():
        paragraphs = split_opinion_into_paragraphs(row['opinion'])
        for para in paragraphs:
            paragraph_data.append({
                'opinion_songernames': row['opinion_songernames'],
                'paragraph': para,
                'original_case_citation_count': row['citation_count']
            })
    
    paragraph_df = pd.DataFrame(paragraph_data)
    
    # Preprocess paragraphs
    paragraph_df['cleaned_paragraph'] = paragraph_df['paragraph'].apply(preprocess_text_v2)
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_tfidf = vectorizer.fit_transform(paragraph_df['cleaned_paragraph'])
    X_tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Supervised feature selection
    selected_features = []
    for col in X_tfidf_df.columns:
        correlation, _ = pearsonr(X_tfidf_df[col], paragraph_df['original_case_citation_count'])
        if abs(correlation) > 0.05:
            selected_features.append(col)
    
    # Select features
    X_selected = X_tfidf_df[selected_features]
    y = paragraph_df['original_case_citation_count']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    # Train Gradient Boosting Regressor
    xgb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)
    
    # Predict citations for all paragraphs
    paragraph_df['predicted_citations'] = xgb.predict(X_selected)
    
    # Rank paragraphs
    ranked_paragraphs = paragraph_df.sort_values('predicted_citations', ascending=False)
    
    # Print model performance
    y_pred = xgb.predict(X_test)
    print(f"Model Performance for {decade}:")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R-squared Score: {r2_score(y_test, y_pred)}")
    
    # Save top and bottom paragraphs
    top_10 = ranked_paragraphs.head(10)[['paragraph', 'predicted_citations', 'opinion_songernames']]
    bottom_10 = ranked_paragraphs.tail(10)[['paragraph', 'predicted_citations', 'opinion_songernames']]
    
    top_10.to_csv(os.path.join(output_directory, f'top_10_paragraphs_{decade}.csv'), index=False)
    bottom_10.to_csv(os.path.join(output_directory, f'bottom_10_paragraphs_{decade}.csv'), index=False)
    
    return ranked_paragraphs

# Define file paths for all four decades
file_paths = {
    '1970s': "C:/Users/lixiang/.ssh/lexis_cases_opinions_circuit_1970s.csv",
    '1980s': "C:/Users/lixiang/.ssh/lexis_cases_opinions_circuit_1980s.csv",
    '1990s': "C:/Users/lixiang/.ssh/lexis_cases_opinions_circuit_1990s.csv",
    '2000s': "C:/Users/lixiang/.ssh/lexis_cases_opinions_circuit_2000s.csv"
}

# Process each decade
for decade, file_path in file_paths.items():
    print(f"\nProcessing {decade}...")
    rank_paragraphs_by_citations(file_path, decade)

print("\nParagraph ranking completed. Results saved in:", output_directory)