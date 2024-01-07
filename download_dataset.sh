#!/bin/bash

# The URL where the dataset is located
DATASET_URL="https://files.de-1.osf.io/v1/resources/nrgx6/providers/osfstorage/623d9d9a938b480e3797af8f"

# The location where the dataset should be saved
DATASET_SAVE_PATH="data/"

# Create the directory where the dataset will be saved if it doesn't exist
mkdir -p "$DATASET_SAVE_PATH"

# The filename for the downloaded file
DATASET_FILENAME="$DATASET_SAVE_PATH/dataset.zip"

# Download the dataset
wget -O "$DATASET_FILENAME" "$DATASET_URL"

# Unzip the dataset if it is a zip file
unzip "$DATASET_FILENAME" -d "$DATASET_SAVE_PATH"

# Optionally, remove the zip file after extracting
#rm "$DATASET_FILENAME"
