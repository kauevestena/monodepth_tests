# run it from the repo root

from lib import *


zerodepth_model = torch.hub.load("TRI-ML/vidar", "ZeroDepth", pretrained=True, trust_repo=True)

zerodepth_model.to('cuda:0')

intrinsics = torch.tensor(np.load('../vidar_release/examples/ddad_intrinsics.npy'), device='cuda:0').unsqueeze(0)
rgb = torch.tensor(imread('../vidar_release/examples/ddad_sample.png'), device='cuda:0').permute(2,0,1).unsqueeze(0)/255.

with torch.no_grad():

    depth_pred = zerodepth_model(rgb, intrinsics)
    colorize(depth_pred,outpath='tests/ddad_sample_zerodepth.png')
    
    for i in range(10):
        start = time.time()
        depth_pred = zerodepth_model(rgb, intrinsics)
        end = time.time()
        print(end - start)

    
