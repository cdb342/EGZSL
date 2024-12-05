#!/bin/bash

nShot=100

CUDA_VISIBLE_DEVICES=0 python egzsl.py \
--dataset CUB \
--class_embedding 'att' \
--attSize 312 \
--lr 5e-5 \
--BatchSize $nShot \
--nShot $nShot \
--tau 0.5 \
--m1 0.99 \
--m2 0.9 \
--weight_kl 5 \
--netC 'checkpoints/CUB.pkl' \


