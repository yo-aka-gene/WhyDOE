#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -l s_vmem=8G,mem_req=8G
#$ -j y
#$ -o prepare_log.txt

TARGET_DIR="./"
URL="https://plus.figshare.com/ndownloader/articles/20029387/versions/1"
ZIP_FILE="${TARGET_DIR}/downloaded_data.zip"

echo "=== 1. Start Downloading with User-Agent Masquerading ==="

wget -c --progress=dot:giga --tries=10 \
     --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" \
     -O "$ZIP_FILE" "$URL"

echo "=== 2. Check & Unzip ==="
file "$ZIP_FILE"

if [[ "$ZIP_FILE" == *.zip ]]; then
    echo "Unzipping..."
    unzip -o "$ZIP_FILE" -d "$TARGET_DIR"
    echo "Unzip finished."
    rm "$ZIP_FILE"
else
    echo "Error: Not a zip file or download failed."
fi

echo "=== 3. Final File List ==="
ls -lh "$TARGET_DIR"