#!/bin/bash
model="LNGRUText"
models="LNGRUText_seg10_layer4/checkpoint_best"
for i in {1..9}
do
    models+=",LNGRUText_seg"$i"_layer4/checkpoint_best"
done
python main.py test --ngpu=8 --model=$model --models=$models --res_file="$model"_res.csv