#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_name=LTDNet

for pred_len in 96 192 336 720; do
    python -u run.py \
          --is_training 1 \
          --root_path ./dataset/electricity/ \
          --data_path electricity.csv \
          --model_id electricity_Trans_96_96_${pred_len} \
          --model ${model_name} \
          --data custom \
          --features M \
          --seq_len 96 \
          --label_len 48 \
          --pred_len ${pred_len} \
          --e_layers 2 \
          --enc_in 321 \
          --dec_in 321 \
          --c_out 321 \
          --des 'Exp' \
          --d_model 512 \
          --d_core 256 \
          --n_heads 1 \
          --batch_size 16 \
          --dropout 0.1 \
          --learning_rate 0.001 \
          --train_epochs 30 \
          --activation gelu \
          --kernel_size 3 \
          --cross_kernel_size 100 \
          --k_factor 0.1
done

for pred_len in 96 192 336 720; do
    python -u run.py \
          --is_training 1 \
          --root_path ./dataset/ETT-small/ \
          --data_path ETTh1.csv \
          --model_id ETTh1_Trans_96_96_${pred_len} \
          --model ${model_name} \
          --data ETTh1 \
          --features M \
          --seq_len 96 \
          --label_len 48 \
          --pred_len ${pred_len} \
          --e_layers 1 \
          --d_layers 1 \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --des 'Exp' \
          --d_model 128 \
          --d_core 64 \
          --n_heads 1 \
          --d_ff 32 \
          --factor 3 \
          --batch_size 32 \
          --dropout 0.1 \
          --learning_rate 0.001 \
          --train_epochs 10 \
          --k 2 \
          --patch_len 16 \
          --class_strategy projection \
          --embed timeF \
          --activation relu \
          --num_workers 8 \
          --kernel_size 3 \
          --cross_kernel_size 3 \
          --k_factor 5
done

for pred_len in 96 192 336 720; do
    python -u run.py \
          --is_training 1 \
          --root_path ./dataset/ETT-small/ \
          --data_path ETTh2.csv \
          --model_id ETTh2_Trans_96_96_${pred_len} \
          --model ${model_name} \
          --data ETTh2 \
          --features M \
          --seq_len 96 \
          --label_len 48 \
          --pred_len ${pred_len} \
          --e_layers 1 \
          --d_layers 1 \
          --enc_in 7 \
          --des 'Exp' \
          --d_model 128 \
          --d_core 64 \
          --batch_size 32 \
          --dropout 0.1 \
          --learning_rate 0.001 \
          --train_epochs 10 \
          --k 2 \
          --activation relu \
          --kernel_size 3 \
          --cross_kernel_size 3 \
          --k_factor 5
done

for pred_len in 96 192 336 720; do
    python -u run.py \
          --is_training 1 \
          --root_path ./dataset/ETT-small/ \
          --data_path ETTm1.csv \
          --model_id ETTm1_Trans_96_96_${pred_len} \
          --model ${model_name} \
          --data ETTm1 \
          --features M \
          --seq_len 96 \
          --label_len 48 \
          --pred_len ${pred_len} \
          --e_layers 1 \
          --d_layers 1 \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --des 'Exp' \
          --d_model 128 \
          --d_core 64 \
          --n_heads 1 \
          --d_ff 64 \
          --factor 3 \
          --batch_size 32 \
          --dropout 0.1 \
          --learning_rate 0.001 \
          --train_epochs 10 \
          --k 2 \
          --patch_len 16 \
          --class_strategy projection \
          --embed timeF \
          --activation relu \
          --num_workers 8 \
          --kernel_size 3 \
          --cross_kernel_size 3 \
          --k_factor 5
done

for pred_len in 96 192 336 720; do
    python -u run.py \
          --is_training 1 \
          --root_path ./dataset/ETT-small/ \
          --data_path ETTm2.csv \
          --model_id ETTm2_Trans_96_96_${pred_len} \
          --model ${model_name} \
          --data ETTm2 \
          --features M \
          --seq_len 96 \
          --label_len 48 \
          --pred_len ${pred_len} \
          --e_layers 1 \
          --d_layers 1 \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --des 'Exp' \
          --d_model 128 \
          --d_core 64 \
          --n_heads 1 \
          --d_ff 32 \
          --factor 3 \
          --batch_size 32 \
          --dropout 0.1 \
          --learning_rate 0.001 \
          --train_epochs 10 \
          --k 2 \
          --patch_len 16 \
          --class_strategy projection \
          --embed timeF \
          --activation relu \
          --num_workers 8 \
          --kernel_size 3 \
          --cross_kernel_size 3 \
          --k_factor 5
done

for pred_len in 12 24 48 96; do
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/PEMS \
        --data_path PEMS03.npz \
        --model_id PEMS03_96_96_${pred_len} \
        --model ${model_name} \
        --data PEMS \
        --features M \
        --seq_len 96 \
        --pred_len ${pred_len} \
        --e_layers 4 \
        --enc_in 358 \
        --des 'Exp' \
        --d_model 512 \
        --d_core 512 \
        --n_heads 1 \
        --batch_size 16 \
        --dropout 0.1 \
        --learning_rate 0.001 \
        --train_epochs 30 \
        --activation relu \
        --kernel_size 3 \
        --cross_kernel_size 50 \
        --k_factor 0.2 \
        --mark 0
done

for pred_len in 12 24 48 96; do
    python -u run.py \
          --is_training 1 \
          --root_path ./dataset/PEMS/ \
          --data_path PEMS04.npz \
          --model_id PEMS04_Trans_96_96_${pred_len} \
          --model ${model_name} \
          --data PEMS \
          --features M \
          --seq_len 96 \
          --label_len 48 \
          --pred_len ${pred_len} \
          --e_layers 3 \
          --enc_in 307 \
          --des 'Exp' \
          --d_model 512 \
          --d_core 512 \
          --n_heads 1 \
          --batch_size 16 \
          --dropout 0.1 \
          --learning_rate 0.001 \
          --train_epochs 10 \
          --activation relu \
          --kernel_size 3 \
          --cross_kernel_size 50 \
          --k_factor 0.2 \
          --mark 0
done

for pred_len in 12 24 48 96; do
    python -u run.py \
          --is_training 1 \
          --root_path ./dataset/PEMS/ \
          --data_path PEMS07.npz \
          --model_id PEMS07_Trans_96_96_${pred_len} \
          --model ${model_name} \
          --data PEMS \
          --features M \
          --seq_len 96 \
          --label_len 48 \
          --pred_len ${pred_len} \
          --e_layers 2 \
          --enc_in 883 \
          --des 'Exp' \
          --d_model 512 \
          --d_core 512 \
          --n_heads 1 \
          --batch_size 16 \
          --dropout 0.1 \
          --learning_rate 0.001 \
          --train_epochs 10 \
          --activation gelu \
          --kernel_size 3 \
          --cross_kernel_size 50 \
          --k_factor 0.2 \
          --mark 0
done

for pred_len in 12 24 48 96; do
    python -u run.py \
          --is_training 1 \
          --root_path ./dataset/PEMS/ \
          --data_path PEMS08.npz \
          --model_id PEMS08_Trans_96_96_${pred_len} \
          --model ${model_name} \
          --data PEMS \
          --features M \
          --seq_len 96 \
          --label_len 48 \
          --pred_len ${pred_len} \
          --e_layers 2 \
          --enc_in 170 \
          --des 'Exp' \
          --d_model 512 \
          --d_core 512 \
          --n_heads 1 \
          --batch_size 16 \
          --dropout 0.1 \
          --learning_rate 0.001 \
          --train_epochs 30 \
          --activation relu \
          --kernel_size 3 \
          --cross_kernel_size 30 \
          --k_factor 0.1 \
          --mark 0
done

for pred_len in 96 192 336 720; do
    python -u run.py \
          --is_training 1 \
          --root_path ./dataset/solar/ \
          --data_path solar_AL.txt \
          --model_id solar_96_96_${pred_len} \
          --model ${model_name} \
          --data Solar \
          --features M \
          --seq_len 96 \
          --label_len 48 \
          --pred_len ${pred_len} \
          --e_layers 2 \
          --enc_in 137 \
          --des 'Exp' \
          --d_model 512 \
          --d_core 128 \
          --n_heads 1 \
          --batch_size 16 \
          --dropout 0.1 \
          --learning_rate 0.001 \
          --train_epochs 10 \
          --activation relu \
          --kernel_size 3 \
          --cross_kernel_size 30 \
          --k_factor 0.1 \
          --mark 0
done

for pred_len in 96 192 336 720; do
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/weather \
        --data_path weather.csv \
        --model_id weather_96_96_${pred_len} \
        --model ${model_name} \
        --data custom \
        --features M \
        --seq_len 96 \
        --pred_len ${pred_len} \
        --e_layers 2 \
        --enc_in 21 \
        --des Exp \
        --d_model 512 \
        --d_core 128 \
        --batch_size 16 \
        --dropout 0.1 \
        --learning_rate 0.001 \
        --train_epochs 30 \
        --activation relu \
        --kernel_size 3 \
        --cross_kernel_size 3 \
        --k_factor 5
done

for pred_len in 96 192 336 720; do
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/exchange_rate \
        --data_path exchange_rate.csv \
        --model_id exchange_rate_96_96_${pred_len} \
        --model ${model_name} \
        --data custom \
        --features M \
        --seq_len 96 \
        --pred_len ${pred_len} \
        --e_layers 1 \
        --enc_in 8 \
        --des Exp \
        --d_model 256 \
        --d_core 128 \
        --batch_size 32 \
        --dropout 0.1 \
        --learning_rate 0.003 \
        --train_epochs 30 \
        --activation gelu \
        --kernel_size 3 \
        --cross_kernel_size 5 \
        --k_factor 3
done
