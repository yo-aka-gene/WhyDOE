import requests
from tqdm import tqdm
import os
import sys

try:
    save_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    save_dir = "."

os.makedirs(save_dir, exist_ok=True)
filename = "downloaded_data.zip" 
save_path = os.path.join(save_dir, filename)

url = "https://plus.figshare.com/ndownloader/articles/20029387/versions/1"
headers = {"User-Agent": "Mozilla/5.0"}

print(f"Save Directory: {save_dir}")
print("Checking file size...")
try:
    with requests.head(url, headers=headers, allow_redirects=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
except Exception as e:
    print(f"Failed to get file size: {e}")
    total_size = 0


resume_byte_pos = 0
if os.path.exists(save_path):
    resume_byte_pos = os.path.getsize(save_path)


if total_size > 0 and resume_byte_pos == total_size:
    print(f"\n✅ File already downloaded completely! ({total_size} bytes)")
    print(f"Path: {save_path}")

else:
    resume_header = headers.copy()
    if resume_byte_pos > 0:
        print(f"Resuming from {resume_byte_pos} bytes...")
        resume_header["Range"] = f"bytes={resume_byte_pos}-"
    else:
        print(f"Starting download from scratch...")

    print(f"Target: {save_path}")

    try:
        with requests.get(url, stream=True, headers=resume_header) as r:
            if r.status_code == 416: 
                print("\n✅ Server returned 416 (Range Not Satisfiable). Assuming download is complete.")
            else:
                r.raise_for_status()
                
                if "content-range" in r.headers:
                    total_size = int(r.headers.get("content-range").split("/")[-1])
                elif total_size == 0:
                    total_size = int(r.headers.get('content-length', 0))
                
                mode = 'ab' if resume_byte_pos > 0 else 'wb'
                
                with open(save_path, mode) as f, tqdm(
                    desc="Downloading",
                    total=total_size,
                    initial=resume_byte_pos,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1024*1024): 
                        if chunk:
                            size = f.write(chunk)
                            bar.update(size)
        print("\nDownload process finished.")

    except Exception as e:
        print(f"\n❌ Download failed: {e}")