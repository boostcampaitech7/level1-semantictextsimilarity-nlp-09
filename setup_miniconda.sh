#!/bin/bash

miniconda_dir="$HOME/miniconda3"
mkdir -p "$miniconda_dir"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$miniconda_dir/miniconda.sh"
bash "$miniconda_dir/miniconda.sh" -b -u -p "$miniconda_dir"
rm "$miniconda_dir/miniconda.sh"
"$miniconda_dir/bin/conda" init bash