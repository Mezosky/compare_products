import re

import numpy as np
import nltk
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

nltk.download('stopwords')
nltk.download('punkt')

def get_processed_text(document):

    """Function to process the text of the documents."""

    return nltk.word_tokenize(document.lower())

def get_stopwords(df: pd.DataFrame, product_id: str = 'nombre', reg_pattern=r"\b\w*\d\w*\b"):

    """Function to get the stopwords from the dataframes."""

    # Get the stopwords in spanish
    spanish_stopwords = stopwords.words('spanish')

    # Simple Tokenize and count the words to get the stopwords
    tokens = [i.lower() for i in df[product_id].str.split(expand=True).stack()]
    df_count = pd.DataFrame.from_dict(
        Counter(tokens), 
        orient='index', 
        columns=['count']
    )
    my_stop_words = list(df_count[df_count['count']==1].index)
    my_stop_words = [i.lower() for i in my_stop_words]
    
    # Find trash symbols and count them
    stop_words_symbols = [
        word.lower() for token in tokens 
        for word in re.findall(reg_pattern, token) 
        if re.findall(reg_pattern, token) != []
    ]
    temp_count = pd.DataFrame.from_dict(Counter(stop_words_symbols), orient='index', columns=['count'])
    stop_words_symbols = list(temp_count[temp_count['count'] == 1].index)
    stop_words_symbols = [i.lower() for i in stop_words_symbols]
    
    # Combine the stopwords
    my_stop_words = list(set(my_stop_words + stop_words_symbols + spanish_stopwords))
    print(f"Number of stopwords: {len(my_stop_words)}")
    return my_stop_words

def vectorize_dataframes(df1, df2, stop_words, mode='hf_transformer'):

    """Function to vectorize the dataframes using the selected mode."""

    if mode == 'tfidf':
        # Define the vectorizer
        vectorizer = TfidfVectorizer(
            tokenizer=get_processed_text
            ,stop_words=stop_words
            ,lowercase=True
            ,strip_accents='ascii'
        )
        # Generate the embeddings
        embf_df_1 = pd.DataFrame(vectorizer.fit_transform(df1['nombre']).toarray()).set_index(df1['nombre'])
        embf_df_2 = pd.DataFrame(vectorizer.transform(df2['nombre']).toarray()).set_index(df2['nombre'])
    elif mode == 'hf_transformer':
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        #import pdb; pdb.set_trace()
        embf_df_1 = pd.DataFrame(model.encode(df1['nombre'])).set_index(df1['nombre'])
        embf_df_2 = pd.DataFrame(model.encode(df2['nombre'].reset_index(drop=True))).set_index(df2['nombre'])
    else:
        raise ValueError('Invalid mode. Unique method programmed: "tfidf"')
    
    return embf_df_1, embf_df_2

def compare_products(embf_df_1: pd.DataFrame, embf_df_2: pd.DataFrame, topn=5):

    """Function to compare the products using cosine similarity."""

    # Calculate cosine similarity for all rows at once
    cosine_similarities = cosine_similarity(embf_df_1, embf_df_2)
    # Convert to DataFrame for better visualization
    cosine_similarities_df = pd.DataFrame(cosine_similarities, columns=embf_df_2.index)
    # Get the top 10 indices for each row
    top_indices = np.argpartition(-cosine_similarities_df.values, topn, axis=1)[:, :topn]
    # Create a DataFrame to store the results
    final_df = pd.DataFrame()
    # Iterate over each row and get the top 10 column names
    for i in range(top_indices.shape[0]):
        print(f"Processing row {i+1}/{top_indices.shape[0]}", end='\r')
        top_cols = embf_df_2.index[top_indices[i]].to_list()
        #import pdb; pdb.set_trace()
        row_df = pd.DataFrame([top_cols], columns=[f"top{j}" for j in range(1, topn+1)], index=[embf_df_1.index[i]])
        final_df = pd.concat([final_df, row_df])

    return final_df

def run_similarity(path_data_1, path_data_2, output_file='final_df.csv'):
    # Load the data
    df1 = pd.read_excel(path_data_1)[['nombre', 'numero parte']].dropna(subset='nombre').iloc[:100,:] # change this line to get all the data
    df2 = pd.read_excel(path_data_2)[['nombre', 'numero parte']].dropna(subset='nombre')
    df2 = df2.drop_duplicates(subset='nombre')
    
    # Vectorize the dataframes
    embf_df_1, embf_df_2 = vectorize_dataframes(df1, df2, stop_words=get_stopwords(df1))
    final_df = compare_products(embf_df_1, embf_df_2)
    final_df.to_csv(output_file)
    