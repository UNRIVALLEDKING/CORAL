#!/usr/bin/env bash
# Download trained weights to ./models/
set -e
mkdir -p models
echo "↓  Fetching marine weights ..."
wget -qO models/marine_model_retrained_weights.pth  "<PUBLIC_LINK_1>"
echo "↓  Fetching coral weights ..."
wget -qO models/coral_cnn.pth                       "<PUBLIC_LINK_2>"
echo "✅  Done"
