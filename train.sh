#!/bin/bash

model=LNGRUText
echo "model:"$model
python main.py train --ngpu=1 --model=$model --id=$model"_seg1_layer4" --index=1 --seed=1 --num-layers=4 &
python main.py train --ngpu=1 --model=$model --id=$model"_seg2_layer4" --index=2 --seed=1 --num-layers=4 &
python main.py train --ngpu=1 --model=$model --id=$model"_seg3_layer4" --index=3 --seed=3 --num-layers=4 &
wait
python main.py train --ngpu=1 --model=$model --id=$model"_seg4_layer4" --index=4 --seed=4 --num-layers=4 &
python main.py train --ngpu=1 --model=$model --id=$model"_seg5_layer4" --index=5 --seed=5 --num-layers=4 &
python main.py train --ngpu=1 --model=$model --id=$model"_seg6_layer4" --index=6 --seed=6 --num-layers=4 &
wait
python main.py train --ngpu=1 --model=$model --id=$model"_seg7_layer4" --index=7 --seed=1 --num-layers=4 &
python main.py train --ngpu=1 --model=$model --id=$model"_seg8_layer4" --index=8 --seed=8 --num-layers=4  &
python main.py train --ngpu=1 --model=$model --id=$model"_seg9_layer4" --index=9 --seed=9 --num-layers=4  &
python main.py train --ngpu=1 --model=$model --id=$model"_seg10_layer4" --index=10 --seed=10 --num-layers=4  &
wait
echo finish