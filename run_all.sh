#!/bin/bash

# This script will execute multiple Python scripts with arguments in sequence

# List of Python scripts and their arguments to execute
python_scripts=(
    "main.py --yaml configs/clip_vit_coco.yaml"
    "main.py --yaml configs/clip_vit_flickr.yaml"
    "main.py --yaml configs/clip_rn_coco.yaml"
    "main.py --yaml configs/clip_rn_flickr.yaml"
    "main.py --yaml configs/align_coco.yaml"
    "main.py --yaml configs/align_flickr.yaml"
    "main.py --yaml configs/flava_coco.yaml"
    "main.py --yaml configs/flava_flickr.yaml"
    "main.py --yaml configs/cyclip_coco.yaml"
    "main.py --yaml configs/cyclip_flickr.yaml"
    "main.py --yaml configs/imagebind_coco.yaml"
    "main.py --yaml configs/imagebind_flickr.yaml"
)

# Loop through each Python script and execute
for script_with_args in "${python_scripts[@]}"
do
    # Split the script name and arguments
    IFS=' ' read -r -a script_args <<< "$script_with_args"
    script="${script_args[0]}"
    args="${script_args[@]:1}"
    
    # Check if the script file exists
    if [ -f "$script" ]; then
        echo "Executing script: $script $args"
        python3 "$script" $args
    else
        echo "Python script not found: $script"
    fi
done