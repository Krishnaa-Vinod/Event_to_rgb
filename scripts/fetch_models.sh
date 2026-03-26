#!/bin/bash
# Fetch upstream model repositories and weights for E2VID and FireNet

set -e

echo "=== Fetching E2VID and FireNet Models ==="
echo "Target directory: $(pwd)"

# Create directories
mkdir -p third_party weights

echo ""
echo "1. Cloning E2VID repository..."
if [ ! -d "third_party/rpg_e2vid" ]; then
    cd third_party
    git clone https://github.com/uzh-rpg/rpg_e2vid.git
    cd rpg_e2vid
    echo "  ✓ E2VID cloned, commit: $(git rev-parse --short HEAD)"
    cd ../..
else
    echo "  ✓ E2VID already exists"
fi

echo ""
echo "2. Cloning FireNet repository..."
if [ ! -d "third_party/rpg_e2vid_firenet" ]; then
    cd third_party
    git clone https://github.com/cedric-scheerlinck/rpg_e2vid.git rpg_e2vid_firenet
    cd rpg_e2vid_firenet
    git checkout cedric/firenet
    echo "  ✓ FireNet cloned, branch: $(git branch --show-current), commit: $(git rev-parse --short HEAD)"
    cd ../..
else
    echo "  ✓ FireNet already exists"
fi

echo ""
echo "3. Downloading E2VID weights..."
if [ ! -f "weights/E2VID_lightweight.pth.tar" ]; then
    cd weights
    wget -O E2VID_lightweight.pth.tar "http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar"
    if [ -f "E2VID_lightweight.pth.tar" ]; then
        echo "  ✓ E2VID weights downloaded ($(du -h E2VID_lightweight.pth.tar | cut -f1))"
    else
        echo "  ✗ E2VID weights download failed"
    fi
    cd ..
else
    echo "  ✓ E2VID weights already exist ($(du -h weights/E2VID_lightweight.pth.tar | cut -f1))"
fi

echo ""
echo "4. FireNet weights information..."
echo "  Note: FireNet weights are hosted on Google Drive and require manual download"
echo "  URL: https://drive.google.com/file/d/1nBCeIF_Us-rGhCjdU5q1Ch-yrFckjZPa/view?usp=sharing"
echo "  Expected location: weights/firenet.pth.tar"
if [ -f "weights/firenet.pth.tar" ]; then
    echo "  ✓ FireNet weights found ($(du -h weights/firenet.pth.tar | cut -f1))"
else
    echo "  ⚠ FireNet weights not found - will need manual download or alternative"
fi

echo ""
echo "=== Model Repository Setup Complete ==="
echo "E2VID: third_party/rpg_e2vid"
echo "FireNet: third_party/rpg_e2vid_firenet"
echo "Weights: weights/"
echo ""
echo "Next steps:"
echo "- Set up Python dependencies for the model repositories"
echo "- Verify model loading and inference"
echo "- If FireNet weights are missing, download manually or use alternative weights"