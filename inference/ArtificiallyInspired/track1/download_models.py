import os
import zipfile
import wget

DOWNLOAD_DIR = "./checkpoint"

if __name__=='__main__':
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    download_path = f'{DOWNLOAD_DIR}/track1.zip'
    wget.download('https://ntire-inspired.s3.us-west-2.amazonaws.com/test_submit/track1.zip', out=download_path)
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(DOWNLOAD_DIR)
