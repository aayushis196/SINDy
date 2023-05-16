#!/bin/bash

# Script Name:  install.sh
# Author:       Aayushi Shrivastava ;   Pratik Shiveshwar
# Unique name:  aayushis            ;   spratik

# This script installs packages

# Packages to install
PACKAGES="pybullet numpy numpngw control opencv-python torch matplotlib tqdm gym"

# Loop through the list of packages and install each one using pip
for package in $PACKAGES; do
    echo "Installing $package..."
    pip install $package
done

# All packages installed successfully
echo "Please ensure python version is 3.9 or above"
echo "All packages have been installed successfully!"
