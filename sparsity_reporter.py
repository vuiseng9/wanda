
import numpy as np
import pandas as pd
from collections import OrderedDict
import torch
import torch.nn as nn
from copy import deepcopy

class SparsityReporter():
    def __init__(
        self,
        model: nn.Module) -> None:
        
        self.model = model
        self.sparsity_df, self.sparsity_blob = self._get_layer_wise_sparsity()

    @staticmethod
    def calc_sparsity(tensor):
        if isinstance(tensor, torch.Tensor):
            rate = 1-(tensor.count_nonzero()/tensor.numel())
            return rate.item()
        else:
            rate = 1-(np.count_nonzero(tensor)/tensor.size)
            return rate

    @staticmethod
    def per_item_sparsity(state_dict):
        dlist=[]
        for key, param in state_dict.items():
            l = OrderedDict()
            l['layer_id'] = key
            l['shape'] = list(param.shape)
            l['nparam'] = np.prod(l['shape'])
            if isinstance(param, torch.Tensor):
                l['nnz'] = param.count_nonzero().item()
            else:
                l['nnz'] = np.count_nonzero(param)
            l['sparsity'] = SparsityReporter.calc_sparsity(param)
            dlist.append(l)
        df = pd.DataFrame.from_dict(dlist)
        return df

    def _get_sparsity_by_tile(self, linear_weight, oc_stride=128, ic_stride=16):
        def get_stats(tensor, axis=None, label=None):
            if label is None:
                prefix = ""
            else:
                prefix = f"{label}_"

            if axis is None:
                return {
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
        
                
        tiled_weight = linear_weight.unfold(0, oc_stride, oc_stride).unfold(1, ic_stride, ic_stride)
        tile_nr, tile_nc, tile_dr, tile_dc = tuple(tiled_weight.shape)

        assert tile_dr == oc_stride, "pls debug"
        assert tile_dc == ic_stride, "pls debug"
        
        tile_numel = tile_dr * tile_dc
        tile_nz = torch.sum(tiled_weight == 0, dim=(-1, -2))

        sparsity_per_tiled_weight = tile_nz/tile_numel
        tile_sparsity_statdict = get_stats(sparsity_per_tiled_weight, label="tile")

        # per tile per col
        ax=2 # col
        tile_nz_by_col = (tiled_weight == 0).sum(dim=ax, keepdim=True)
        tile_sparsity_by_col = tile_nz_by_col/tiled_weight.shape[ax]
        globstat_tile_sparsity_by_col = get_stats(tile_sparsity_by_col, label="col")
        localstat_tile_sparsity_by_col = get_stats(tile_sparsity_by_col, axis=ax, label="col")

        # per tile per row
        ax=3 # row
        tile_nz_by_col = (tiled_weight == 0).sum(dim=ax, keepdim=True)
        tile_sparsity_by_row = tile_nz_by_col/tiled_weight.shape[ax]
        globstat_tile_sparsity_by_row = get_stats(tile_sparsity_by_row, label="row")
        localstat_tile_sparsity_by_row = get_stats(tile_sparsity_by_row, axis=ax, label="row")
        
        # ----------------------------------------------------------------
        # # alternate implementation for just global stat by col and row
        # # per col
        # per_col_weight = linear_weight.unfold(0, oc_stride, oc_stride).unfold(1, 1, 1)
        # nr, nc, dr, dc = tuple(per_col_weight.shape)
        # per_col_nz = torch.sum(per_col_weight == 0, dim=(-1, -2))
        # sparsity_per_col_weight = per_col_nz/dr
        # per_col_sparsity_statdict_global = get_stats(sparsity_per_col_weight, label="col")

        # # per row
        # per_row_weight = linear_weight.unfold(0, 1, 1).unfold(1, ic_stride, ic_stride)
        # nr, nc, dr, dc = tuple(per_row_weight.shape)
        # per_row_weight = torch.sum(per_row_weight == 0, dim=(-1, -2))
        # sparsity_per_col_weight = per_row_weight/dc
        # per_row_sparsity_statdict_global = get_stats(sparsity_per_col_weight, label="row")

        # summ_dict = {
        #     "tile_shape": (oc_stride, ic_stride),
        #     "n_tile": ' x '.join(map(str, sparsity_per_tiled_weight.shape)),
        #     "n_tile_total": sparsity_per_tiled_weight.numel(),
        #     **tile_sparsity_statdict,
        #     **per_col_sparsity_statdict_global,
        #     **per_row_sparsity_statdict_global,
        # }
        
        # ----------------------------------------------------------------

        summ_dict = {
            "tile_shape": (oc_stride, ic_stride),
            "n_tile": ' x '.join(map(str, sparsity_per_tiled_weight.shape)),
            "n_tile_total": sparsity_per_tiled_weight.numel(),
            **tile_sparsity_statdict,
            **globstat_tile_sparsity_by_col,
            **globstat_tile_sparsity_by_row,
        }

        sparsity_blob = {
            "tile_sparsity": sparsity_per_tiled_weight,
            "tile_sparsity_per_col": tile_sparsity_by_col,
            "tile_sparsity_per_row": tile_sparsity_by_row,
        }
        return summ_dict, sparsity_blob

    def _get_layer_wise_sparsity(self):
        dlist=[]
        blobdict = OrderedDict()
        for n, m in self.model.named_modules():
            if not isinstance(m, torch.nn.Linear):
                continue

            if hasattr(m, 'weight'):
                l = OrderedDict()
                l['layer_id'] = n
                l['layer_type'] = m.__class__.__name__
                l['param_type'] = 'weight'
                l['shape'] = list(m.weight.shape)
                l['nparam'] = np.prod(l['shape'])
                l['nnz'] = m.weight.count_nonzero().item()
                l['sparsity'] = self.calc_sparsity(m.weight)

                stat, blob = self._get_sparsity_by_tile(m.weight.data)
                l.update(stat)
                dlist.append(l)

                blobdict[n] = blob

            if False: # disabling this temporily
                if hasattr(m, 'bias'):
                    if m.bias is not None:
                        l = OrderedDict()
                        l['layer_id'] = n
                        l['layer_type'] = m.__class__.__name__
                        l['param_type'] = 'bias'
                        l['shape'] = list(m.bias.shape)
                        l['nparam'] = np.prod(l['shape'])
                        l['nnz'] = m.bias.count_nonzero().item()
                        l['sparsity'] = self.calc_sparsity(m.bias)
                        dlist.append(l)
                
        df = pd.DataFrame.from_dict(dlist)
        return df, blobdict
