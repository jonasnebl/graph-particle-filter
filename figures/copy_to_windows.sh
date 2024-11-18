#!/bin/sh

# Run this shell file to copy all the generated figures to the windows folder with the latex project

SOURCE_DIR="./figures"
FILE_EXTENSION=".pdf"
TARGET_DIR="/mnt/c/users/jonas/repos/SA/figures"

# Find and copy all files with the specified extension
find "$SOURCE_DIR" -type f -name "*$FILE_EXTENSION" -exec cp {} "$TARGET_DIR" \;

echo "All $FILE_EXTENSION files have been copied to $TARGET_DIR"