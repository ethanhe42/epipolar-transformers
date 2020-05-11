import torch
a = torch.load('outs/benchmark/keypoint_h36m/model_007.pth')
a.keys()
a['model'].keys()
a['model']['backbone.module.final_layer.bias'].shape
a['model']['backbone.module.final_layer.bias'] = a['model']['backbone.module.final_layer.bias'][[0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 9 , 11, 12, 14, 15, 16, 17, 18, 19]]
#a['model']['backbone.module.final_layer.weight'] = a['model']['backbone.module.final_layer.weight'][:, [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 9 , 11, 12, 14, 15, 16, 17, 18, 19]]
a['model']['backbone.module.final_layer.weight'].shape
a['model']['backbone.module.final_layer.weight'] = a['model']['backbone.module.final_layer.weight'][[0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 9 , 11, 12, 14, 15, 16, 17, 18, 19]]
torch.save?
torch.save(a, 'model_edited.pth')
%history -f 20to17.py
