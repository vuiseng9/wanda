#!/usr/bin/env bash

mkdir -p /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.0
CUDA_VISIBLE_DEVICES=0 python main.py --model meta-llama/Meta-Llama-3.1-8B --prune_method wanda --sparsity_ratio 0.0 --sparsity_type unstructured --save_model --save /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.0 &

mkdir -p /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.1
CUDA_VISIBLE_DEVICES=1 python main.py --model meta-llama/Meta-Llama-3.1-8B --prune_method wanda --sparsity_ratio 0.1 --sparsity_type unstructured --save_model --save /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.1 &

mkdir -p /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.2
CUDA_VISIBLE_DEVICES=2 python main.py --model meta-llama/Meta-Llama-3.1-8B --prune_method wanda --sparsity_ratio 0.2 --sparsity_type unstructured --save_model --save /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.2 &

mkdir -p /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.3
CUDA_VISIBLE_DEVICES=3 python main.py --model meta-llama/Meta-Llama-3.1-8B --prune_method wanda --sparsity_ratio 0.3 --sparsity_type unstructured --save_model --save /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.3 

mkdir -p /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.4
CUDA_VISIBLE_DEVICES=0 python main.py --model meta-llama/Meta-Llama-3.1-8B --prune_method wanda --sparsity_ratio 0.4 --sparsity_type unstructured --save_model --save /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.4 &

mkdir -p /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.5
CUDA_VISIBLE_DEVICES=1 python main.py --model meta-llama/Meta-Llama-3.1-8B --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save_model --save /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.5 &

mkdir -p /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.6
CUDA_VISIBLE_DEVICES=2 python main.py --model meta-llama/Meta-Llama-3.1-8B --prune_method wanda --sparsity_ratio 0.6 --sparsity_type unstructured --save_model --save /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.6 &

mkdir -p /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.7
CUDA_VISIBLE_DEVICES=3 python main.py --model meta-llama/Meta-Llama-3.1-8B --prune_method wanda --sparsity_ratio 0.7 --sparsity_type unstructured --save_model --save /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.7 

python analyze_sparsity.py --model_path /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.0
python analyze_sparsity.py --model_path /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.1
python analyze_sparsity.py --model_path /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.2
python analyze_sparsity.py --model_path /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.3
python analyze_sparsity.py --model_path /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.4
python analyze_sparsity.py --model_path /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.5
python analyze_sparsity.py --model_path /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.6
python analyze_sparsity.py --model_path /data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.7
