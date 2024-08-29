

import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sparsity_reporter import SparsityReporter


def main():
    parser = argparse.ArgumentParser(description="analyze sparsity of weight pruned CausalLM")

    parser.add_argument('--model_path', type=str, help="path to HF model")

    args = parser.parse_args()

    model_id = args.model_path
    
    # Dev use, dangerous
    # model_id = "/data2/vchua/run/hgx1-240823-wanda/wanda-ed/meta-llama/Meta-Llama-3.1-8B-wanda-unstructured-0.6"

    cfg = AutoConfig.from_pretrained(model_id)

    if not cfg.torch_dtype:
        raise ValueError("torch_dtype not in config.json")
        
        if not isinstance(cfg.torch_dtype, tuple(torch.bfloat16, torch.float16)):
            raise ValueError("cfg.torch_dtype is not 16 bit, pls review")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=cfg.torch_dtype, # we set according to config.json

        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    reporter = SparsityReporter(model=model)

    # llama3 specific processing!
    def replace_common_prefix(strings, new_prefix):
        # Find the common prefix
        common_prefix = os.path.commonprefix(strings)
        # Replace the common prefix with the new prefix
        return [new_prefix + s[len(common_prefix):] for s in strings]
 
    reporter.sparsity_df.insert(1, 'short_id', list(reporter.sparsity_df.layer_id.values))
    # reporter.sparsity_df.short_id[:-1] = \
        # replace_common_prefix(reporter.sparsity_df.short_id[:-1].tolist(), "txblk.")

    reporter.sparsity_df.short_id = \
        reporter.sparsity_df.short_id.str.replace("model.layers", "tx").\
            str.replace("self_attn", "attn").str.replace("_proj", "")

    rpt_path = os.path.join(model_id, f"sparsity_report_{os.path.basename(model_id)}.csv")
    print(f"Dumping to {rpt_path}")
    reporter.sparsity_df.to_csv(rpt_path, index_label='row')

    blob_path = os.path.join(model_id, f"blob.sparsity._{os.path.basename(model_id)}")
    torch.save(dict(df=reporter.sparsity_df, blob=reporter.sparsity_blob), blob_path)
    print("end.")

if __name__ == "__main__":
    main()