import torch
import argparse

#resnet152: 60344232
#poseresnet152: 68792703
# fusion: 68,792,703 + 491,520,000 = 560,312,703, 80^4 * 17 * 12 = 8355840000 + 204236619776.0 = 212,592,459,776
# ransac: 204236619776.0 68,635,472.0
#algebraic: 79676546 209846091776.0 79,521,888.0
#volumetric: 80750339 4viewsFLOPS 359,566,409,728.0 
#epipolar resnet152: 68855408 4viewsFLOPS 408678498304 - 204236619776.0 = 204,441,878,528
#epipolar C=256: 66817

parser = argparse.ArgumentParser(description="PyTorch Keypoints Training")
parser.add_argument(
    "--src",
    default="",
    help="source model",
    type=str,
)

args = parser.parse_args()

model = torch.load(args.src)
if 'model' in model:
    model = model['model']
print(model.keys())
from IPython import embed ;embed()
print('total params:', sum(v.numel() for k, v in model.items()))

special = ['epipolar', 'aggre']
for s in special:
    epipolar = 0
    for k, v in model.items():
        if s in k:
            p = v.numel()      
            epipolar += p  
            print(k, p)
    print(s+' total:', epipolar)
