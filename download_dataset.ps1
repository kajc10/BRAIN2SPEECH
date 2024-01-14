# Define the dataset URL and the save path
$DATASET_URL = "https://files.de-1.osf.io/v1/resources/nrgx6/providers/osfstorage/623d9d9a938b480e3797af8f"
$DATASET_SAVE_PATH = "data\"
$DATASET_FILENAME = "$DATASET_SAVE_PATH\SingleWordProductionDutch-iBIDS.zip"

# Create the directory if it doesn't exist
New-Item -ItemType Directory -Force -Path $DATASET_SAVE_PATH

# Download the dataset
Invoke-WebRequest -Uri $DATASET_URL -OutFile $DATASET_FILENAME

# Unzip the dataset
Expand-Archive -LiteralPath $DATASET_FILENAME -DestinationPath $DATASET_SAVE_PATH -Force

# Optionally, remove the zip file after extracting
# Remove-Item $DATASET_FILENAME
