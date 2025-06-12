#!/bin/bash

# Create temp directory
TEMP_DIR=$(mktemp -d)
ZIP_PATH="$TEMP_DIR/download.zip"

# Download ZIP file
wget -O "$ZIP_PATH" "https://drive.upm.es/s/PAotfPpXbLxHjA5/download"

# Unzip to temp dir
unzip -q "$ZIP_PATH" -d "$TEMP_DIR"

# Detect folder inside the ZIP
EXTRACTED_FOLDER=$(find "$TEMP_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)

# Check if folder was found
if [ -z "$EXTRACTED_FOLDER" ]; then
  echo "❌ No folder found in the zip archive."
  exit 1
fi

# Move and rename to ./data
mv "$EXTRACTED_FOLDER"/3_CAR-ARENA ./data

# Clean up
rm -rf "$TEMP_DIR"

echo "✅ Folder extracted and renamed to ./data"
