#!/usr/bin/bash

conda activate spatial 
export PYTHONPATH="${PYTHONPATH}:/ltenas3/LatentMODIS"
# python /ltenas3/LatentMODIS/1c_prepare_training_data
python /ltenas3/LatentMODIS/2_train_PCA.py
python /ltenas3/LatentMODIS/2_train_UMAP.py 
python /ltenas3/LatentMODIS/2_train_ParametricUMAP.py
