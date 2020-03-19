import torch 
import sys
import test_mapgiou

pretrain = "/home/master/kuanzi/weights/72_epoch_arcface.pt" #576x320
print("Loading finetune weight...", pretrain)
sys.stdout.flush()
checkpoint = torch.load(pretrain, map_location='cpu')
print("epoch:", checkpoint['epoch'])
print("optimizer:", checkpoint['optimizer'])


# with torch.no_grad():
#     # mAP, R, P = test.test(cfg, data_cfg, weights=latest, batch_size=batch_size, print_interval=40)
#     # print ("test.test:\t", mAP, "\t", R, "\t", P)
#     test_mapgiou.test_giou(cfg, data_cfg, weights=latest, batch_size=batch_size, print_interval=40)
#     test_mapgiou.test_emb(cfg, data_cfg, weights=latest, batch_size=batch_size, print_interval=40)