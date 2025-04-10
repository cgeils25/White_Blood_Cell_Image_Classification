import os
import shutil
import requests
import rarfile
import zipfile
from neattime import neattime
from typing import List

DATA_URLs = ["http://dl.raabindata.com/WBC/Cropped_double_labeled/TestA.rar",
"http://dl.raabindata.com/WBC/Cropped_double_labeled/TestB.zip",
"http://dl.raabindata.com/WBC/Cropped_double_labeled/Train.rar"]

# for holding the compressed data
TEMP_DIR = f'temp_{neattime()}/'

# for holding the final processed data 
DATA_DIR = f'data/'

CELL_TYPES = ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

def download_data(urls: List[str]):
    """Download raw data files from URLs.

    Args:
        urls (List[str]): List of URLs to download data from.

    Returns:
        List[str]: List of file paths of downloaded files.
    """
    downloaded_filepaths = []

    for url in urls:
        print(f"Downloading {url}...")
        local_filename = os.path.join(TEMP_DIR, url.split('/')[-1])

        with requests.get(url) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                f.write(r.content)
        
        downloaded_filepaths.append(local_filename)

        print(f"Downloaded {url} and saved to {local_filename}.")
    
    return downloaded_filepaths
    

def extract_data(filepath: str):
    """Extract data from compressed files (RAR or ZIP).

    Args:
        filepath (str): Path to the compressed file.
    """
    print(f"Extracting data from {filepath}...")
    if filepath.endswith('.rar'):
        with rarfile.RarFile(filepath) as rf:
            rf.extractall(path=DATA_DIR)

            print(f"Extracted data from {filepath}")
        
    elif filepath.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zf:
            zf.extractall(path=DATA_DIR)
            print(f"Extracted data from {filepath}")


def extract_all_data(filepaths: List[str]):
    """Extract data from all provided file paths.

    Args:
        filepaths (List[str]): List of file paths to extract data from.
    """
    for filepath in filepaths:
        extract_data(filepath)


def reorganize_data(data_dir: str = DATA_DIR):
    """Reorganize data into separate directories for each cell type. Ignore the included 'train' and 'test' splits provided
    by raabin.

    Args:
        data_dir (str, optional): directory containing extracted, uncompressed data. Defaults to DATA_DIR.
    """
    sub_data_dirs = os.listdir(data_dir)
    
    for cell_type in CELL_TYPES:
        os.makedirs(os.path.join(data_dir, cell_type), exist_ok=True)
    
    for sub_data_dir in sub_data_dirs:
        sub_data_dir_cell_types = os.listdir(os.path.join(data_dir, sub_data_dir))

        for cell_type in CELL_TYPES:
            if cell_type in sub_data_dir_cell_types:
                src_dir = os.path.join(data_dir, sub_data_dir, cell_type)
                dst_dir = os.path.join(data_dir, cell_type)
                for filename in os.listdir(src_dir):
                    shutil.move(os.path.join(src_dir, filename), os.path.join(dst_dir, filename))
                shutil.rmtree(src_dir)
        
        shutil.rmtree(os.path.join(data_dir, sub_data_dir))


def main():
    os.makedirs(TEMP_DIR, exist_ok = True)

    print("Downloading data...")
    downloaded_filepaths = download_data(DATA_URLs)

    print("Download complete. Extracting data...")

    extract_all_data(downloaded_filepaths)

    print("Extraction complete.")

    shutil.rmtree(path=TEMP_DIR)

    print("Reorganizing data...")
    reorganize_data()
    print("Data reorganization complete.")


if __name__ == "__main__":
    main()
