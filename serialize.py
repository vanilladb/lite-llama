import os
import torch

model = torch.load('weights/7B/consolidated.00.pth')

layer_num = -1
weights = None
output_dir = 'serialized'

for k, v in model.items():
    if 'layers' in k and 'weight' in k:
        if str(layer_num) in k:
            weights = torch.cat((weights, v.flatten()))
            i += 1
        else:
            print(k)
            if layer_num != -1:
                print(weights.shape)
                weights = weights.numpy()
                weights.tofile(os.path.join(os.path.dirname(__file__), output_dir, f'layer{layer_num}.bin'))
            weights = v.flatten()
            layer_num += 1
            i = 1

if weights is not None:
    print(weights.shape)
    weights = weights.numpy()
    weights.tofile(os.path.join(os.path.dirname(__file__), output_dir, f'layer{layer_num}.bin'))
    weights = None


io_weight = [(k,v) for k, v in model.items()][:3]
torch.save({k:v for k, v in io_weight}, os.path.join(os.path.dirname(__file__), output_dir, 'io.pt'))