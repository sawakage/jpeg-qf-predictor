import os
import requests
import sys
import subprocess

def get_current_tag():
    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--exact-match"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return tag
    except subprocess.CalledProcessError:
        return None

def download_model(version):
    url = f"https://github.com/sawakage/jpeg-qf-predictor/releases/download/{version}/model.pth"
    local_dir = "checkpoints"
    local_path = os.path.join(local_dir, "model.pth")
    os.makedirs(local_dir, exist_ok=True)
    print(f"Downloading model for version {version}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Model saved to {local_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        version = sys.argv[1]
    else:
        version = get_current_tag()
        if not version:
            print("Error: Could not detect current Git tag. Please specify version manually.")
            sys.exit(1)
    download_model(version)