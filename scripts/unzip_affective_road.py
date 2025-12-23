import os
import zipfile

def unzip_all_e4(e4_root):
    for drive_dir in os.listdir(e4_root):
        drive_path = os.path.join(e4_root, drive_dir)
        if os.path.isdir(drive_path):
            for zip_name in ['Left.zip', 'Right.zip']:
                zip_path = os.path.join(drive_path, zip_name)
                if os.path.exists(zip_path):
                    extract_path = os.path.join(drive_path, zip_name.replace('.zip', ''))
                    if not os.path.exists(extract_path):
                        print(f"Unzipping {zip_path} to {extract_path}...")
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(extract_path)
                    else:
                        print(f"Already unzipped: {extract_path}")

if __name__ == "__main__":
    E4_ROOT = r"C:\Users\Administrator\Desktop\Major_Project\data\AffectiveROAD_Data\Database\E4"
    unzip_all_e4(E4_ROOT)
