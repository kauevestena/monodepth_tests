from lib import *

# packnet:
packnet_model = torch.hub.load("TRI-ML/vidar", "PackNet", pretrained=True, trust_repo=True)
packnet_model.to('cuda:0')

# zerodepth:
zerodepth_model = torch.hub.load("TRI-ML/vidar", "ZeroDepth", pretrained=True, trust_repo=True)
zerodepth_model.to('cuda:0')

# using the sample camera:
camera = camera_params()

packnet_width = 640
camera_packnet = camera.resized_by_width(packnet_width)
camera_1280 = camera.resized_by_width(1280)

for imgname in tqdm(os.listdir(samples_folderpath)):
    img = imread(os.path.join(samples_folderpath, imgname))

    # img = reduce_img_size(img,25)

    img_packnet = reduce_img_fixed_width(img,packnet_width)
    img_packnet = cut_image_aspect_ratio(img_packnet, packnet_ratio)

    camera = sample_camera

    # convert to tensor  
    rgb = torch.tensor(img,device='cuda:0').permute(2,0,1).unsqueeze(0)/255.

    rgb_packnet = torch.tensor(img_packnet,device='cuda:0').permute(2,0,1).unsqueeze(0)/255.

    with torch.no_grad():

        # packnet:
        t1 = time.time()
        depth_pred = packnet_model(rgb_packnet)
        depth_pred_numpy = packnet_as_numpy(depth_pred)
        colorize(depth_pred_numpy,outpath=os.path.join(packnet_tests_folderpath, imgname))
        simple_tooktime(t1,'packnet')

        
        # img_resize_test = reduce_img_fixed_width(img,1280)
        # img_resize_test = cut_image_aspect_ratio(img_resize_test, packnet_ratio)
        # upsampled_depth = resize(depth_pred_numpy, (img_resize_test.shape[1], img_resize_test.shape[0]))


        save_pcloud(color_img=img_packnet,
                    depth_data=depth_pred_numpy,
                    pcloud_path=os.path.join(packnet_pcloud_tests_folderpath, 
                                             imgname.replace('.jpg','.txt')),f_pix=camera_1280['f_pix']),

        # zerodepth:
        # t2 = time.time()
        # intrinsics = torch.tensor(intrinsics_matrix(camera), device='cuda:0').unsqueeze(0)
        # depth_pred = zerodepth_model(rgb.float(), intrinsics.float())
        # colorize(depth_pred,outpath=os.path.join(zerodepth_tests_folderpath, imgname))
        # simple_tooktime(t2,'zerodepth')

        # zerodepth_pcloud_path = os.path.join(zerodepth_tests_folderpath, imgname.replace('.jpg','.txt'))

        # save_pcloud(color_img=img, 
        #             depth_data=depth_pred.cpu().numpy()[0][0], 
        #             pcloud_path=zerodepth_pcloud_path, 
        #             f_pix=sample_camera_quarter['f_norm'])