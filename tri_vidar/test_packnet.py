# run it from the repo root
from lib import *

packnet_model = torch.hub.load("TRI-ML/vidar", "PackNet", pretrained=True, trust_repo=True)

packnet_model.to('cuda:0')

rgb = torch.tensor(imread('../vidar_release/examples/ddad_sample.png'),device='cuda:0').permute(2,0,1).unsqueeze(0)/255.

with torch.no_grad():

    depth_pred = packnet_model(rgb)
    
    colorize(depth_pred[0],outpath='tests/ddad_sample_packnet.png')

    for i in range(10):
        start = time.time()
        depth_pred = packnet_model(rgb)
        end = time.time()
        print(end - start)