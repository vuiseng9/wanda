import torch
import pandas as pd


# Set options to display the full DataFrame
# pd.set_option('display.max_rows', None)   # Display all rows
# pd.set_option('display.max_columns', None) # Display all columns
pd.set_option('display.float_format', '{:.4f}'.format)  # Format floats to 3 decimal places

def get_stats(tensor, axis=None, label=None):
    if label is None:
        prefix = ""
    else:
        prefix = f"{label}_"

    if axis is None:
        return {
            f"{prefix}count": tensor.numel(),
            f"{prefix}avg": tensor.mean().item(),
            f"{prefix}min": tensor.min().item(),
            f"{prefix}med": tensor.median().item(),
            f"{prefix}max": tensor.max().item(),
        }
    else:
        return {
            f"{prefix}avg": tensor.mean(dim=axis),
            f"{prefix}min": tensor.min(dim=axis),
            f"{prefix}med": tensor.median(dim=axis),
            f"{prefix}max": tensor.max(dim=axis),
        }

class SparseBlob:
    def __init__(self, blob_path):
        self.blob_path = blob_path
        d = torch.load(blob_path)
        self._rpt = d['df']
        self._blob = d['blob']

    @classmethod
    def print_help(cls):
        print("\n- Help ------------------")
        print("\nh = SparseBlob(\"path to sparsity blob\")\n")
        
        methods = [
            'preview', 'ls_layers',
            'get_sparsity_by_short_id',
            'get_sparsity_by_row_id',
            'get_sparsity_of_tile',
            'show_help'
        ]
        for method in methods:
            print(f'{cls.__name__}.{method}:{cls.__dict__[method].__doc__}\n')
        print("- End of Help ------------------")

    def show_help(self):
        """
        print help for available function of SparseBlob
        eg. h.show_help()
        """
        self.print_help()

    def preview(self):
        """
        preview sparsity dataframe, intend to show row id, short id for look up
        eg. h.preview()
        """
        pd.set_option('display.max_rows', None)
        print(self._rpt)
        pd.reset_option('display.max_rows')

    def ls_layers(self):
        """
        list all available layer ids for look up
        eg. h.ls_layers()
        """
        pd.set_option('display.max_rows', None)
        print(self._rpt.short_id)
        pd.reset_option('display.max_rows')

    def get_sparsity_by_short_id(self, short_id):
        """
        return a sparsity stats of a layer via short_id lookup.
        eg. h.get_sparsity_by_short_id('tx.0.attn.v')
        """
        return self._rpt[self._rpt.short_id == short_id].iloc[0]

    def get_sparsity_by_row_id(self, id):
        """
        return a sparsity stats of a layer via row id lookup.
        eg. h.get_sparsity_by_row_id(36)
        """
        return self._rpt.iloc[id]

    def get_sparsity_of_tile(self, lut_id, tile_id):
        """
        zoom into a specific layer and a specific tile,
        return the sparsity stats of the tile down to col, row granularity
        eg. h.get_sparsity_by_row_id(36, (5, 6))
        """
        if isinstance(lut_id, int):
            row = self.get_sparsity_by_row_id(lut_id)
        elif  isinstance(lut_id, str):
            row = self.get_sparsity_by_layer_id(lut_id)
        else:
            print("Invalid lookup id, use row number or short_id; you can do .preview() to find out.")

        l = row['layer_id']
        max_nrow, max_ncol = self._blob[l]['tile_sparsity'].shape

        if not isinstance(tile_id, tuple) or len(tile_id) != 2:
            print(f"[Error] tile_id must be a tuple, eg .get_sparsity_of_tile({lut_id}, ( 5, 10))")
        else:
            if tile_id[0] >= max_nrow or tile_id[1] >= max_ncol:
                print(f"[Error] tile not exist: tile_id must be (0-{max_nrow-1}, 0-{max_ncol-1})")
            else:
                outdict={
                    "tile_id": tile_id,
                    "layer_id": l,
                    "tiled by": row['tile_shape'],
                    "tile_sparsity": self._blob[l]['tile_sparsity'][tile_id].item(),
                    **get_stats(self._blob[l]['tile_sparsity_per_col'][tile_id], label="col"),
                    **get_stats(self._blob[l]['tile_sparsity_per_row'][tile_id], label="row"),
                }
                self._print_one_tile_stats(outdict)

    def _print_one_tile_stats(self, d):
        for key, value in d.items():
            if isinstance(value, float):
                print(f'{value:>40.4f} : {key}')
            else:
                print(f'{str(value):>40} : {key}')


SparseBlob.print_help()