import os
from pathlib import Path
import torch

from utils import Timing

WEIGHTS_DIR = Path(__file__).parent / "weights/"

WEIGHTS_7B_FILENAME = [WEIGHTS_DIR / "7B/consolidated.00.pth"]
WEIGHTS_13B_FILENAMES = [WEIGHTS_DIR / "13B/consolidated.00.pth", WEIGHTS_DIR / "13B/consolidated.01.pth"]
WEIGHTS_30B_FILENAMES = [WEIGHTS_DIR / "30B/consolidated.00.pth", WEIGHTS_DIR / "30B/consolidated.01.pth", WEIGHTS_DIR / "30B/consolidated.02.pth", WEIGHTS_DIR / "30B/consolidated.03.pth"]
WEIGHTS_65B_FILENAMES = [WEIGHTS_DIR / "65B/consolidated.00.pth", WEIGHTS_DIR / "65B/consolidated.01.pth", WEIGHTS_DIR / "65B/consolidated.02.pth", WEIGHTS_DIR / "65B/consolidated.03.pth", WEIGHTS_DIR / "65B/consolidated.04.pth", WEIGHTS_DIR / "65B/consolidated.05.pth", WEIGHTS_DIR / "65B/consolidated.06.pth", WEIGHTS_DIR / "65B/consolidated.07.pth"]

PARAM = '13B'
WEIGHT_FILES = {
    '7B': WEIGHTS_7B_FILENAME, 
    '13B': WEIGHTS_13B_FILENAMES, 
    '30B': WEIGHTS_30B_FILENAMES,
    '65B': WEIGHTS_65B_FILENAMES
}

device = 'cpu'

TMP_DIR = Path(__file__).parent / "tmp/"
TMP_DIR.mkdir(exist_ok=True)

print(f'Using LLaMA-{PARAM}')

state_dict_keys = None
for filename in WEIGHT_FILES[PARAM]:
    with Timing('Finished loading state dict in '):
        state_dict = torch.load(filename, map_location=device)

    state_dict_keys = state_dict.keys() if state_dict_keys is None else state_dict_keys
    for state in state_dict.items():
        disk_tensor_file = TMP_DIR / state[0]
        if disk_tensor_file.exists():
            disk_tensor = list(torch.load(disk_tensor_file, map_location=device))
            if state[0].startswith('tok_embeddings') \
                or state[0].endswith('.attention.wo.weight') \
                or state[0].endswith('.feed_forward.w2.weight'):
                axis = 1
            else:
                axis = 0
            disk_tensor[1] = torch.cat((disk_tensor[1], state[1]), dim=axis)
        torch.save(state, disk_tensor_file)
    del state_dict

output_dir = Path(f'serialized/{PARAM}/')
output_dir.mkdir(exist_ok=True)
layer_num = -1
weights = None

for filename in state_dict_keys:
    if 'layers' in filename and 'weight' in filename:
        v = torch.load(TMP_DIR / filename, map_location=device)[1]
        if str(layer_num) in filename:
            weights = torch.cat((weights, v.flatten()))
        else:
            print(filename)
            if layer_num != -1:
                print(weights.shape)
                weights = weights.numpy()
                weights.tofile(os.path.join(os.path.dirname(__file__), output_dir, f'layer{layer_num}.bin'))
            weights = v.flatten()
            layer_num += 1

if weights is not None:
    weights = weights.numpy()
    weights.tofile(os.path.join(os.path.dirname(__file__), output_dir, f'layer{layer_num}.bin'))
    weights = None

io_weight_files = [f for f in os.listdir(TMP_DIR) if 'layers' not in f and 'weight' in f]
io_weights = dict()

for f in io_weight_files:
    weights = torch.load(TMP_DIR / f)
    io_weights[weights[0]] = weights[1]

torch.save(io_weights, os.path.join(os.path.dirname(__file__), output_dir, 'io.pt'))

import shutil
shutil.rmtree(TMP_DIR)