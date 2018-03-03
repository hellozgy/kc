#!/bin/bash
model=LNGRUText

python main.py train --ngpu=9 --model=LNGRUText --id=$model"_seg1" --index=1 > logs/$model"_seg1" 2>&1 &
python main.py train --ngpu=8 --model=LNGRUText --id=$model"_seg2" --index=2 > logs/$model"_seg2" 2>&1 &
python main.py train --ngpu=7 --model=LNGRUText --id=$model"_seg3" --index=3 > logs/$model"_seg3" 2>&1 &
python main.py train --ngpu=6 --model=LNGRUText --id=$model"_seg4" --index=4 > logs/$model"_seg4" 2>&1 &
python main.py train --ngpu=5 --model=LNGRUText --id=$model"_seg5" --index=5 > logs/$model"_seg5" 2>&1 &
wait
python main.py train --ngpu=9 --model=LNGRUText --id=$model"_seg6" --index=6 > logs/$model"_seg6" 2>&1 &
python main.py train --ngpu=8 --model=LNGRUText --id=$model"_seg7" --index=7 > logs/$model"_seg7" 2>&1 &
python main.py train --ngpu=7 --model=LNGRUText --id=$model"_seg8" --index=8 > logs/$model"_seg8" 2>&1 &
python main.py train --ngpu=6 --model=LNGRUText --id=$model"_seg9" --index=9 > logs/$model"_seg9" 2>&1 &
python main.py train --ngpu=5 --model=LNGRUText --id=$model"_seg10" --index=10 > logs/$model"_seg10" 2>&1 &