#!/bin/bash

nShot=100


CUDA_VISIBLE_DEVICES=0 python egzsl.py \
--dataset AWA2 \
--class_embedding 'att' \
--attSize 85 \
--lr 5e-5 \
--BatchSize $nShot \
--nShot $nShot \
--tau 0.5 \
--m1 0.99 \
--m2 0.9 \
--weight_kl 1 \
--netC 'checkpoints/AWA2.pkl' \

