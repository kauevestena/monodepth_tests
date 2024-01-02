from tri_vidar.lib import *

inputfolderpath = 'tests/smmt_samples'


# packnet:
packnet_model = torch.hub.load("TRI-ML/vidar", "PackNet", pretrained=True, trust_repo=True)
packnet_model.to('cuda:0')
outfolder_packnet = 'tests/packnet'

# zerodepth:
zerodepth_model = torch.hub.load("TRI-ML/vidar", "ZeroDepth", pretrained=True, trust_repo=True)
zerodepth_model.to('cuda:0')
outfolder_zerodepth = 'tests/zerodepth'




for imgname in tqdm(os.listdir(inputfolderpath)):
    img = imread(os.path.join(inputfolderpath, imgname))
    
    print(img.shape)
    
    scale_percent = 25 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    img = resize(img, dim)
    
    rgb = torch.tensor(img,device='cuda:0').permute(2,0,1).unsqueeze(0)/255.

    with torch.no_grad():

        # packnet:
        # depth_pred = packnet_model(rgb)
        # colorize(depth_pred[0],outpath=os.path.join(outfolder_packnet, imgname))

        # zerodepth:
        intrinsics = torch.tensor(intrinsics_matrix(sample_camera_quarter), device='cuda:0').unsqueeze(0)
        depth_pred = zerodepth_model(rgb.float(), intrinsics.float())
        colorize(depth_pred,outpath=os.path.join(outfolder_zerodepth, imgname))
        zerodepth_pcloud_path = os.path.join(outfolder_zerodepth, imgname.replace('.jpg','.txt'))

        save_pcloud(color_img=img, 
                    depth_data=depth_pred.cpu().numpy()[0][0], 
                    pcloud_path=zerodepth_pcloud_path, 
                    f_pix=sample_camera_quarter['f_norm'])