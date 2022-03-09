#!/bin/bash

# executes the patches necessary for SocNavBench

set -e # fail on error

cd socnav
bash patches/apply_patches_1.sh
bash patches/apply_patches_3.sh
cd - # back to base dir

echo # newline
echo -e "Done!"