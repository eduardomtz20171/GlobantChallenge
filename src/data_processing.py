import os
import xml.etree.ElementTree as ET
import pandas as pd
import logging
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s'
)

def safe_strip(node):
    return node.text.strip() if (node is not None and node.text) else None

def parse_nsf_xml(file_path):
    file_name = os.path.basename(file_path)
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError:
        return None

    award_node = root.find('Award')
    if award_node is None:
        return None

    award_title = safe_strip(award_node.find('AwardTitle'))
    abstract = safe_strip(award_node.find('AbstractNarration'))
    award_id = safe_strip(award_node.find('AwardID'))
    if not award_id:
        award_id = file_name.replace('.xml', '')

    return {
        'award_id': award_id,
        'award_title': award_title,
        'abstract': abstract
    }

def read_all_xml(folder_path):
    records, failed_files, missing_abstract_files = [], [], []
    file_list = [f for f in os.listdir(folder_path) if f.lower().endswith('.xml')]

    if not file_list:
        logging.warning(f"No XML files found in folder: {folder_path}")
        return pd.DataFrame()

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        record = parse_nsf_xml(file_path)

        if record:
            if not record['abstract']:
                missing_abstract_files.append(file_name)
            records.append(record)
        else:
            failed_files.append(file_name)

    df = pd.DataFrame(records)

    logging.info(f"Parsed {len(records)} file(s) successfully, {len(failed_files)} failed.")
    if missing_abstract_files:
        logging.warning(f"Files with missing abstracts: {missing_abstract_files}")
    return df

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    return ' '.join(tokens)

def preprocess_df(df):
    df['clean_text'] = df['abstract'].astype(str).apply(preprocess_text)
    return df

# Main processing function
def process_data(raw_data_path):
    df_abstracts = read_all_xml(raw_data_path)
    df_abstracts.dropna(subset=['abstract'], inplace=True)
    df_clean = preprocess_df(df_abstracts)
    return df_clean

# Example usage:
# RAW_DATA_PATH = 'data/raw'
# df_clean = process_data(RAW_DATA_PATH)
# print(df_clean.head())