#!/bin/bash

N=100
total_time=0

for ((i=1; i<=N; i++))
do
    start=$(date +%s%N)  # 取得開始時間 (奈秒)
    
    python main.py > /dev/null 2>&1   # 執行程式，輸出丟掉以免干擾
    
    end=$(date +%s%N)    # 取得結束時間 (奈秒)
    
    elapsed=$(( (end - start)/1000000 ))  # 轉換成毫秒
    total_time=$((total_time + elapsed))
    
    echo "EExecute $i: ${elapsed} ms"
    # python eval.py predicted_label.npy
done

avg_time=$((total_time / N))
echo "Average execution time over $N runs: ${avg_time} ms"
