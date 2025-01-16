import os
import xml.etree.ElementTree as ET
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s'
)

def parse_nsf_xml(file_path):
    file_name = os.path.basename(file_path)
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        logging.warning(f"Failed to parse {file_name}: {e}")
        return None

    award_node = root.find('Award')
    if award_node is None:
        logging.warning(f"No <Award> node found in {file_name}. Skipping.")
        return None

    award_title_node = award_node.find('AwardTitle')
    abstract_node    = award_node.find('AbstractNarration')
    award_id_node    = award_node.find('AwardID')

    # If the node exists, check if text is not None
    def safe_strip(node):
        return node.text.strip() if (node is not None and node.text is not None) else None

    award_title = safe_strip(award_title_node)
    abstract    = safe_strip(abstract_node)
    award_id    = safe_strip(award_id_node)

    # Fallback if no award_id
    if not award_id:
        award_id = file_name.replace('.xml', '')
        logging.info(f"AwardID missing in {file_name}. Using file name {award_id} as ID.")

    # Log if abstract is empty or None
    if not abstract:
        logging.warning(f"Empty or missing <AbstractNarration> in {file_name} (AwardID={award_id}).")

    return {
        'award_id': award_id,
        'award_title': award_title,
        'abstract': abstract
    }


def read_all_xml(folder_path):
    """
    Reads all XML files in the folder_path, parses them,
    and returns a pandas DataFrame with relevant fields.
    """
    records = []
    file_list = [f for f in os.listdir(folder_path) if f.lower().endswith('.xml')]
    
    if not file_list:
        logging.warning(f"No XML files found in folder: {folder_path}")
        return pd.DataFrame()
    
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        record = parse_nsf_xml(file_path)
        if record:
            records.append(record)
        else:
            logging.info(f"Skipping file {file_name} due to parse errors or missing data.")
    
    if not records:
        logging.warning("No valid records were extracted from XML files.")
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    logging.info(f"Successfully parsed {len(df)} XML file(s) from {folder_path}.")
    return df

# Example usage:
# df_nsf = read_all_xml("data/")
# print(df_nsf.head())
