from pathlib import Path



model_list = ["meta-llama/Meta-Llama-3.1-8B"]

OUTROOT = "/data2/vchua/run/hgx1-240823-wanda/wanda-ed"
root = "/data/vchua/dev/hgx1-240823-wanda/wanda"

tasks = []
ngpu = 4

sparse_model_folder_list = []
for base_model in model_list:
    for run_id, sparsity in enumerate([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]):
        gpu_id = run_id % ngpu
        sparse_model_folder = Path(OUTROOT, f'{base_model}-wanda-unstructured-{sparsity}')
        sparse_model_folder_list.append(sparse_model_folder)

        cmd = f"mkdir -p {sparse_model_folder}\nCUDA_VISIBLE_DEVICES={gpu_id} python main.py --model {base_model} "\
              f"--prune_method wanda --sparsity_ratio {sparsity} --sparsity_type unstructured --save_model --save {sparse_model_folder} "
        
        if gpu_id != (ngpu-1):
            cmd += "&"
        
        cmd += "\n"
        print(cmd)

for sparse_model in sparse_model_folder_list:
    cmd = f"python analyze_sparsity.py --model_path {sparse_model}"
    print(cmd)

