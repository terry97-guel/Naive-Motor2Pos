#!/bin/bash

python TrainSorosim.py --epochs=100 --WANDB --nLayers=1 --wd=0.001 --saveDir='./output/temp/SIGbnwd1e-3' --activation='SIGMOID' --device='0'
python TrainSorosim.py --epochs=100 --WANDB --nLayers=1 --wd=0.005 --saveDir='./output/temp/SIGbnwd5e-3' --activation='SIGMOID' --device='0'
python TrainSorosim.py --epochs=100 --WANDB --nLayers=1 --wd=0.01 --saveDir='./output/temp/SIGbnwd1e-2' --activation='SIGMOID' --device='0'
python TrainSorosim.py --epochs=100 --WANDB --nLayers=1 --wd=0.05 --saveDir='./output/temp/SIGbnwd5e-2' --activation='SIGMOID' --device='0'
python TrainSorosim.py --epochs=100 --WANDB --nLayers=1 --wd=0.1 --saveDir='./output/temp/SIGbnwd1e-1' --activation='SIGMOID' --device='0'