import os
import zipfile
import gdown

file_id = '11Irr25DLpUZ8mPNAaask_biHozvCV-tP'
url = f'https://drive.google.com/uc?id={file_id}'
output_file = 'dataset.zip'
extract_path = os.getcwd() 

def setup_data():
    target_folder = os.path.join(extract_path, 'Datos')
    
    if os.path.exists(target_folder):
        print(f"Target folder '{target_folder}' already exists. Skipping download.")
        return

    print("Downloading dataset from Drive...")
    gdown.download(url, output_file, quiet=False)

    # Extract
    if os.path.exists(output_file):
        print("Extracting files...")
        with zipfile.ZipFile(output_file, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Cleaning up zip file...")
        os.remove(output_file)
        print("Setup complete.")
    else:
        print("Error: Download failed.")

if __name__ == "__main__":
    setup_data()