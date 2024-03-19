import glob
import yaml
import argparse
import logging
import os
import wget
import gzip
import shutil
import logging


logging.basicConfig(level = logging.INFO)
logger = logging.getLogger()

DATA_STORE = "./conf/data_stores/*.yaml"

def process_yaml_files():
    results = {}
    files = glob.glob(DATA_STORE)
    for file in files:
        with open(file, "r") as f:
            dataset_config = yaml.safe_load(f)
            if dataset_config is None:
                continue
            for key, value in dataset_config.items():
                if isinstance(value, dict) and 'download_link' in value and 'file' in value:
                    results[key] = {
                        "download_link": value['download_link'],
                        "file": value['file']
                    }
    return results

def download_data(url, save_file):
    # Ensure the save directory exists
    save_dir = os.path.dirname(save_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    original_file_name = url.split('/')[-1]
    temp_file_path = os.path.join(save_dir, original_file_name)
    
    # Download the file
    wget.download(url, out=temp_file_path)
    print("")
    logger.info(f"***** Successfully Downloaded '{url}' *****")

    # If the file is a gzip file, decompress it
    if original_file_name.endswith(".gz"):
        decompressed_file_path = temp_file_path[:-3]  # Remove '.gz' from the temp_file_path for the decompressed file
        logger.info(f"***** Decompressed gzip file saved as '{decompressed_file_path}'")
        with gzip.open(temp_file_path, 'rb') as f_in:
            with open(decompressed_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(temp_file_path)  # Remove the original gzip file after decompression
        temp_file_path = decompressed_file_path
    
    # Check if the destination file already exists
    if os.path.exists(save_file):
        os.remove(save_file)  # Delete the existing file
        logger.info(f"***** Existing file at '{save_file}' has been overwritten.")
    
    # Now move the file safely
    shutil.move(temp_file_path, save_file)
    logger.info(f"***** Successfully download data to '{save_file}'")


def main(download_targets):
    results = process_yaml_files()
    if 'all' in download_targets:
        download_targets = results.keys()
    
    for download_target in download_targets:
        if download_target in results:
            meta = results[download_target]
            logger.info(f"***** Downloading Dataset '{download_target}' *****")
            download_data(meta['download_link'], meta['file'])
            logger.info(f"***** Successfully download '{download_target}'")
        else:
            logger.error(f"Dataset '{download_target}' not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets based on YAML configuration.")
    parser.add_argument("dataset", nargs='*', default="all", help="The dataset(s) to download (default: all).")
    args = parser.parse_args()

    main(args.dataset)



