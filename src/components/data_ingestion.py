import logging 
import os

def reading_files(data_dir: str = r"src\data\raw_data" ) -> str:

    """
    ### This function is used to return the file location of the dataset
    - It accepts both csv or xlsx file
    """
    valid_data = ['csv', 'xlsx']

    files = []

    for i in os.listdir(data_dir):
        if i.split(".")[-1] in valid_data:
            files.append(i)
        
    if not files:
        logging.error("File with extension 'csv' and 'xlsx' not found")
        raise FileNotFoundError("File with extension 'csv' and 'xlsx' not found")
    
    if len(files) > 1:
        logging.info("Multiple files are present in the location")
        raise ValueError("Multiple files of csv or xlsx are present in the location")

    return os.path.join(data_dir,files[0])
