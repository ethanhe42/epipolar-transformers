#!/bin/bash
FILE=RHD_v1-1.zip

echo "Downloading..."
wget https://lmb.informatik.uni-freiburg.de/data/RenderedHandpose/$FILE

echo "Unzipping..."
unzip $FILE

echo "Done"
