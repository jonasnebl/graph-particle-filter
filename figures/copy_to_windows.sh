#!/bin/sh

# Run this shell file to copy all the generated figures to the windows folder with the latex project

SOURCE_DIR="./figures"
FILE_EXTENSION_1=".pdf"
FILE_EXTENSION_2=".svg"
TARGET_DIR="/mnt/c/users/jonas/repos/SA/figures"

# Find and copy all files with the specified extension
find "$SOURCE_DIR" -type f -name "*$FILE_EXTENSION_1" -exec cp {} "$TARGET_DIR" \;
find "$SOURCE_DIR" -type f -name "*$FILE_EXTENSION_2" -exec cp {} "$TARGET_DIR" \;

echo "All $FILE_EXTENSION_1 and $FILE_EXTENSION_2 files have been copied to $TARGET_DIR"