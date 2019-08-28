import os
from collections import OrderedDict

import torch


def load_model_state(path:os.PathLike, device:str) -> OrderedDict:
    """
    When saving DataParallel model we have a nagging "module." prefix.
    This function removes it
    """
    original_state_dict = torch.load(path, map_location=device)
    was_parallel = all([k.startswith('module.') for k in original_state_dict.keys()])

    if was_parallel:
        new_state_dict = OrderedDict()

        for k, v in original_state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v

        return new_state_dict
    else:
        return original_state_dict
