This branch is developed for sweeping sparsity on Llama3.1-8B with detailed sparsity down to tile, tile_col, tile_row level.

gencli_sweep_sparsity.py
    generate commands to run wanda at different sparsity, as well as running analyze_sparsity.py on the saved wanda model.

run.sh
    a specific instance taken from above for actual run

analyze_sparsity.py
    analyze a wanda model, output are csv and a blob

sparsity_reporter.py
    imported by the above, layer-wise sparsity analysis

interactive_sparsity.sh
lookup_sparsity_blob.py
    for looking up a specific tile for thier sparsity stats

dev_shell_like.py
    for dev purpose of the behavior of the two above

branch.vsnotes.md
    This file, important documentation!


See this hf repo that captures only the reports and blobs for interactive use.
https://huggingface.co/vuiseng9/24-0830-wanda-llama3.1-8B