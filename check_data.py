# import argparse
# import json
# import time

# import test  
# import test_metrics
# from models import *
# from utils.datasets import JointDataset, collate_fn
# from utils.utils import *
# from utils.log import logger
# from torchvision.transforms import transforms as T


# def train(
#         cfg,
#         data_cfg,
#         resume=False,
#         epochs=100,
#         batch_size=16,
#         accumulated_batches=1,
#         freeze_backbone=False,
#         opt=None,
# ):
#     # weights = 'weights' 
#     # mkdir_if_missing(weights)
#     # latest = osp.join(weights, 'latest.pt')

#     torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

#     # Configure run
#     f = open(data_cfg)
#     data_config = json.load(f)
#     trainset_paths = data_config['train']
#     dataset_root = data_config['root']
#     f.close()
#     cfg_dict = parse_model_cfg(cfg) 
#     img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]

#     # Get dataloader
#     transforms = T.Compose([T.ToTensor()])
#     dataset = JointDataset(dataset_root, trainset_paths, img_size, augment=True, transforms=transforms)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
#                                              num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn) 
#     # for i, (imgs0, targets, imgs0_path, imgs0_size, targets_len) in enumerate(dataloader):
#     for i, (_, _, imgs0_path, imgs0_size, _) in enumerate(dataloader):
#         print(imgs0_path)
    
#     print("All rigth! check done!")
#     # i 84 
#     # imgs tensor([[[[0.5020, 0.5020, 0.5020,  ..., 0.5020, 0.5020, 0.5020],
#     #           [0.5020, 0.5020, 0.5020,  ..., 0.5020, 0.5020, 0.5020],
#     #           ...,
#     #           [0.5020, 0.5020, 0.5020,  ..., 0.5020, 0.5020, 0.5020]]]]) 
#     # targets tensor([[[ 0.0000e+00,  6.8300e+02,  6.0382e-01,  3.5323e-01,  1.3714e-02,
#     #           5.0201e-02],
#     #         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
#     #           0.0000e+00],
#     #         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
#     #           0.0000e+00]]]) 
#     # one ['/home/master/kuanzi/data/Caltech/images/set04_V010_790.png', '/home/master/kuanzi/data/PRW/images/c3s1_010576.jpg', '/home/master/kuanzi/data/PRW/images/c1s1_019276.jpg', '/home/master/kuanzi/data/Caltech/images/set05_V003_1378.png', '/home/master/kuanzi/data/Caltech/images/set00_V001_1469.png', '/home/master/kuanzi/data/MOT17/images/train/MOT17-05-SDP/img1/000432.jpg', '/home/master/kuanzi/data/Caltech/images/set02_V009_1105.png', '/home/master/kuanzi/data/CUHKSYSU/images/s5678.jpg'] 
#     # two [[480, 640], [1080, 1920], [1080, 1920], [480, 640], [480, 640], [480, 640], [480, 640], [800, 600]] 
#     # target_len tensor([[1.],
#     #         [5.],
#     #         [4.],
#     #         [8.]])

#     # i 85 
#     # imgs tensor([[[[0.5020, 0.5020, 0.5020,  ..., 0.5020, 0.5020, 0.5020],
        
   

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
#     # parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
#     parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
#     parser.add_argument('--accumulated-batches', type=int, default=1, help='number of batches before optimizer step')
#     parser.add_argument('--cfg', type=str, default='cfg/yolov3_1088x608.cfg', help='cfg file path')
#     # parser.add_argument('--data-cfg', type=str, default='cfg/ccmcpe.json', help='coco.data file path')
#     parser.add_argument('--data-cfg', type=str, default='cfg/ccmcpe_check.json', help='coco.data file path')
#     parser.add_argument('--resume', action='store_true', help='resume training flag')
#     parser.add_argument('--print-interval', type=int, default=40, help='print interval')
#     parser.add_argument('--test-interval', type=int, default=9, help='test interval')
#     parser.add_argument('--lr', type=float, default=1e-2, help='init lr')
#     parser.add_argument('--unfreeze-bn', action='store_true', help='unfreeze bn')
#     opt = parser.parse_args()

#     init_seeds()

#     train(
#         opt.cfg,
#         opt.data_cfg,
#         resume=opt.resume,
#         epochs=opt.epochs,
#         batch_size=opt.batch_size,
#         accumulated_batches=opt.accumulated_batches,
#         opt=opt,
#     )



from PIL import Image
import os
import sys

if __name__ == '__main__':
    badFilesList = []
    image_path = sys.argv[1]
    for root, dirs, files in os.walk(image_path):
        for each in files:
            if each.endswith('.png') or each.endswith('.jpg') or each.endswith('.gif') \
            or each.endswith('.JPG') or each.endswith('.PNG') or each.endswith('.GIF') \
            or each.endswith('.jpeg') or each.endswith('.JPEG'):
                try:
                    im = Image.open(os.path.join(root, each))
                    # im.show()
                except Exception as e:
                    print('Bad file:', os.path.join(root, each))
                    badFilesList.append(os.path.join(root, each))

    print("badFilesList:", badFilesList)