#!/bin/bash

# Color codes for pretty output
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
BLUE="\033[1;34m"
NC="\033[0m" # No Color

print_section() {
  echo -e "\n${BLUE} $1${NC}"
}

print_success() {
  echo -e "${GREEN} $1${NC}"
}

print_warning() {
  echo -e "${YELLOW}  $1${NC}"
}

print_error() {
  echo -e "${RED} $1${NC}"
}

# Step 1: Ensure data directory exists
print_section "Checking data directory"
if [ ! -d "./data" ]; then
  mkdir ./data
  print_success "Created 'data' directory."
else
  print_warning "'data' directory already exists."
fi

# Step 2: Check for required tools
print_section "Checking system dependencies"

for cmd in wget unzip conda; do
  if ! command -v $cmd &> /dev/null; then
    print_error "'$cmd' is not installed. Please install it to proceed."
    exit 1
  else
    print_success "'$cmd' is installed."
  fi
done

# Step 3: Download and extract dataset
print_section "Downloading and extracting CAR-ARENA dataset..."

TEMP_DIR=$(mktemp -d)
ZIP_PATH="$TEMP_DIR/car-arena.zip"

wget -q --show-progress -O "$ZIP_PATH" "https://drive.upm.es/s/PAotfPpXbLxHjA5/download"
if [ $? -ne 0 ]; then
  print_error "Failed to download the dataset. Please check your internet connection."
  exit 1
fi
if [ ! -f "$ZIP_PATH" ]; then
  print_error "Downloaded file not found: $ZIP_PATH"
  exit 1
fi
unzip -q "$ZIP_PATH" -d "$TEMP_DIR"

EXTRACTED_FOLDER=$(find "$TEMP_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)
if [ -z "$EXTRACTED_FOLDER" ]; then
  print_error "No folder found in the zip archive."
  exit 1
fi

mv "$EXTRACTED_FOLDER/3_CAR-ARENA" ./data
rm -rf "$TEMP_DIR"

print_success "Dataset extracted and moved to './data/3_CAR-ARENA'"

# Step 4: Create Conda environment
print_section "Creating Conda environment"

if conda env list | grep -qE '^\s*classic_vision\s'; then
  print_warning "Conda environment 'classic_vision' already exists. Skipping creation."
else
  if conda env create -f environment.yml; then
    print_success "Conda environment 'classic_vision' created successfully."
  else
    print_error "Failed to create Conda environment. Check 'environment.yml'."
    exit 1
  fi
fi

# Step 5: Final Instructions
print_section "Setup Complete!"
echo -e "To activate the environment:\n\t$ conda activate classic_vision"

print_section "Usage Examples"
echo -e "\t$ python edges.py --rgb input.jpg"
echo -e "\t$ python heights.py --depth depth.png --rgb rgb.png"
echo -e "\t$ python normals_angles.py --depth depth.png --rgb rgb.png"
echo -e "\t$ python normals_xyz.py --depth depth.png --rgb rgb.png"
echo -e "\t$ python sift_surf.py --rgb rgb.png$"
