#!/usr/bin/env bash

mkdir -p /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-2:4-0.5
CUDA_VISIBLE_DEVICES=0 python main.py --model meta-llama/Meta-Llama-3.1-8B --prune_method wanda --sparsity_ratio 0.5 --sparsity_type 2:4 --save_model --save /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-2:4-0.5 &

mkdir -p /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-4:8-0.5
CUDA_VISIBLE_DEVICES=1 python main.py --model meta-llama/Meta-Llama-3.1-8B --prune_method wanda --sparsity_ratio 0.5 --sparsity_type 4:8 --save_model --save /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-4:8-0.5


python analyze_sparsity.py --model_path /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-2:4-0.5
python analyze_sparsity.py --model_path /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-4:8-0.5
