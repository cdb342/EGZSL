#!/bin/bash
nShot=100

CUDA_VISIBLE_DEVICES=0 python egzsl.py \
--dataset APY \
--class_embedding 'att' \
--attSize 64 \
--lr 5e-5 \
--BatchSize $nShot \
--nShot $nShot \
--tau 1.0 \
--m1 0.99 \
--m2 0.9 \
--weight_kl 1.0 \
--netC 'checkpoints/APY.pkl' \