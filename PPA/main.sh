#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00

for trial in 1 2 3 4 5 6 7 8 9 10
do
	# python main.py MODEL.TRIAL "$trial" MODEL.DEVICE "cuda:3" MODEL.DEVICE_ID "('3')" DATASETS.ROOT_DIR "data/regdb/" DATASETS.NAMES "('regdb')" INPUT.TYPE "mosaic" INPUT.RADIUS "24.0" OUTPUT_DIR "('./log/regdb/regdb-mosaic-full-distillation-anon-trial$trial')" MODEL.VAL_R1 "0.15"
	python main.py MODEL.TRIAL "$trial" MODEL.DEVICE "cuda:6" MODEL.DEVICE_ID "('6')" DATASETS.ROOT_DIR "/data0/yb_data/RegDB/" DATASETS.NAMES "('regdb')" INPUT.TYPE "mosaic" INPUT.RADIUS "24.0" OUTPUT_DIR "('/data1/yb_log/csreid/regdb/regdb-mosaic-full-trial$trial')" MODEL.VAL_R1 "0.15" MODEL.DCMA "on" MODEL.DCMA_LAMBDA "0.1" MODEL.POSE_GUIDE "on" MODEL.POSE_LAMBDA "1.0"

done
echo 'Done!'