import argparse
import json
import time
import sys

# import test  
import test_mapgiou
from models_diou_arcface import *
# from models_diou import *
# from models import *
from utils.datasets import JointDataset, collate_fn
from utils.utils import *
from utils.log import logger
from torchvision.transforms import transforms as T

# import multiprocessing
# multiprocessing.set_start_method('spawn',True)

def train(
        cfg,
        data_cfg,
        resume=False,
        epochs=100,
        batch_size=16,
        accumulated_batches=1,
        freeze_backbone=False,
        opt=None,
):
    weights = '../weights'   # 改到上一层， 这样方便文件夹复制
    mkdir_if_missing(weights)
    latest = osp.join(weights, 'latest.pt') # 这个是为了resume上次存好的checkpoint，注意不要覆盖！

    torch.backends.cudnn.benchmark = True  # unsuitable for multiscale

    # Configure run
    print("loading data")
    sys.stdout.flush()
    f = open(data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    f.close()
    cfg_dict = parse_model_cfg(cfg) 
    img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]

    # Get dataloader
    transforms = T.Compose([T.ToTensor()])
    dataset = JointDataset(dataset_root, trainset_paths, img_size, augment=True, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=16, pin_memory=True, drop_last=True, collate_fn=collate_fn) 

    # Initialize model
    print("building model")
    sys.stdout.flush()
    model = Darknet(cfg_dict, dataset.nID)

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    if resume:
        if opt.latest:
            latest_resume = "/home/master/kuanzi/weights/66_epoch_diou_arcface.pt"
            print("Loading the latest weight...", latest_resume)
            checkpoint = torch.load(latest_resume, map_location='cpu')

            # Load weights to resume from
            model.load_state_dict(checkpoint['model'])
            model.cuda().train()

            # Set optimizer
            classifer_param_value = list(map(id, model.classifier.parameters()))
            classifer_param = model.classifier.parameters()
            base_params = filter(lambda p: id(p) not in classifer_param_value, model.parameters())
            print("classifer_param\n", classifer_param)  #  [2218660649072]
            print("classifer_param_value\n", classifer_param_value)  #  [2218660649072]
            print("base_params\n", base_params)  # <filter object at 0x0000020493D95048>
            sys.stdout.flush()
            # optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr * 0.1, momentum=.9)
            optimizer = torch.optim.SGD([
                        {'params': filter(lambda x: x.requires_grad, base_params), 'lr': opt.lr * 0.01},
                        {'params': classifer_param, 'lr': opt.lr}], 
                        momentum=.9)
            # optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr, momentum=.9)

            start_epoch = checkpoint['epoch'] + 1
            if checkpoint['optimizer'] is not None:
                # Anyway, if you’re “freezing” any part of your network, and your optimizer is only passed “unfrozen” model parameters 
                # (i.e. your optimizer filters out model parameters whose requires_grad is False), 
                # then when resuming, you’ll need to unfreeze the network again and re-instantiate the optimizer afterwards. 
                optimizer.load_state_dict(checkpoint['optimizer'])

            del checkpoint  # current, saved

        else:
            # pretrain = "/home/master/kuanzi/weights/jde_1088x608_uncertainty.pt"
            pretrain = "/home/master/kuanzi/weights/jde_864x480_uncertainty.pt" #576x320
            print("Loading jde finetune weight...", pretrain)
            sys.stdout.flush()
            checkpoint = torch.load(pretrain, map_location='cpu')
            
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model'].items() if not k.startswith("classifier")}  # 去掉全连接层
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            model.cuda().train()
            print ("model weight loaded")
            sys.stdout.flush()

            classifer_param_value = list(map(id, model.classifier.parameters()))
            classifer_param = model.classifier.parameters()
            base_params = filter(lambda p: id(p) not in classifer_param_value, model.parameters())
            print("classifer_param\n", classifer_param)  #  [2218660649072]
            print("classifer_param_value\n", classifer_param_value)  #  [2218660649072]
            print("base_params\n", base_params)  # <filter object at 0x0000020493D95048>
            sys.stdout.flush()
            # optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr * 0.1, momentum=.9)
            optimizer = torch.optim.SGD([
                        {'params': filter(lambda x: x.requires_grad, base_params), 'lr': opt.lr * 0.01},
                        {'params': classifer_param, 'lr': opt.lr}], 
                        momentum=.9)

            print("chk epoch:\n", checkpoint['epoch'])
            sys.stdout.flush()
            start_epoch = checkpoint['epoch'] + 1

    else:
        # Initialize model with backbone (optional)
        print("Loading backbone...")
        sys.stdout.flush()
        if cfg.endswith('yolov3.cfg'):
            load_darknet_weights(model, osp.join(weights ,'darknet53.conv.74'))
            cutoff = 75
        elif cfg.endswith('yolov3-tiny.cfg'):
            load_darknet_weights(model, osp.join(weights , 'yolov3-tiny.conv.15'))
            cutoff = 15

        model.cuda().train()

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr, momentum=.9, weight_decay=1e-4)

    model = torch.nn.DataParallel(model)
    # Set scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
            milestones=[int(0.5*opt.epochs), int(0.75*opt.epochs)], gamma=0.1)
    
    # An important trick for detection: freeze bn during fine-tuning 
    if not opt.unfreeze_bn:
        for i, (name, p) in enumerate(model.named_parameters()):
            p.requires_grad = False if 'batch_norm' in name else True

    model_info(model)
       
    t0 = time.time()
    print("begin training...")
    sys.stdout.flush()
    for epoch in range(epochs):
        epoch += start_epoch

        logger.info(('%8s%12s' + '%10s' * 6) % (
            'Epoch', 'Batch', 'box', 'conf', 'id', 'total', 'nTargets', 'time'))
        
        # Freeze darknet53.conv.74 for first epoch
        if freeze_backbone and (epoch < 2):
            for i, (name, p) in enumerate(model.named_parameters()):
                if int(name.split('.')[2]) < cutoff:  # if layer < 75
                    p.requires_grad = False if (epoch == 0) else True

        ui = -1
        rloss = defaultdict(float)  # running loss
        optimizer.zero_grad()
        for i, (imgs, targets, _, _, targets_len) in enumerate(dataloader):
            if sum([len(x) for x in targets]) < 1:  # if no targets continue
                continue
            
            # SGD burn-in
            burnin = min(1000, len(dataloader))
            if (epoch == 0) & (i <= burnin):
                lr = opt.lr * (i / burnin) **4 
                for g in optimizer.param_groups:
                    g['lr'] = lr
            
            # Compute loss, compute gradient, update parameters
            loss, components = model(imgs.cuda(), targets.cuda(), targets_len.cuda())
            components = torch.mean(components.view(-1, 5),dim=0)

            loss = torch.mean(loss)
            loss.backward()

            # accumulate gradient for x batches before optimizing
            if ((i + 1) % accumulated_batches == 0) or (i == len(dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()

            # Running epoch-means of tracked metrics
            ui += 1
            
            for ii, key in enumerate(model.module.loss_names):
                rloss[key] = (rloss[key] * ui + components[ii]) / (ui + 1)

            s = ('%8s%12s' + '%10.3g' * 6) % (
                '%g/%g' % (epoch, epochs - 1),
                '%g/%g' % (i, len(dataloader) - 1),
                rloss['box'], rloss['conf'],
                rloss['id'],rloss['loss'],
                rloss['nT'], time.time() - t0)
            t0 = time.time()
            if i % opt.print_interval == 0:
                logger.info(s)
        
        # # Save latest checkpoint
        # checkpoint = {'epoch': epoch,
        #               'model': model.module.state_dict(),
        #               'optimizer': optimizer.state_dict()}
        # torch.save(checkpoint, latest)

        # Calculate mAP
        if epoch % opt.test_interval ==0 and epoch != 0:
            epoch_chk = osp.join(weights, str(epoch) + '_epoch_diou_arcface.pt')
            checkpoint = {'epoch': epoch,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, epoch_chk)
            # """ 训练与测试解耦，以下工作单独进行 """
            # with torch.no_grad():
            #     # mAP, R, P = test.test(cfg, data_cfg, weights=latest, batch_size=batch_size, print_interval=40)
            #     # print ("test.test:\t", mAP, "\t", R, "\t", P)
            #     test_mapgiou.test_giou(cfg, data_cfg, weights=latest, batch_size=batch_size, print_interval=40)
            #     test_mapgiou.test_emb(cfg, data_cfg, weights=latest, batch_size=batch_size, print_interval=40)


        # Call scheduler.step() after opimizer.step() with pytorch > 1.1.0 
        scheduler.step()

if __name__ == '__main__':
    # 576x320 可以batch=8单卡
    # 864x480 可以batch=4单卡
    # 1088x608 可以batch=4单卡
    # CUDA_VISIBLE_DEVICES=0,1 python train_exp_diou_arcface.py --data-cfg cfg/ccmcpe.json --batch-size 8 > train_exp_diou_arcface_dataall.log 2>&1 &
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--accumulated-batches', type=int, default=1, help='number of batches before optimizer step')
    
    # parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
    # parser.add_argument('--cfg', type=str, default='cfg/yolov3_1088x608.cfg', help='cfg file path')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3_864x480.cfg', help='cfg file path')  # 864x480  576x320
    parser.add_argument('--data-cfg', type=str, default='cfg/ccmcpe.json', help='coco.data file path')
    # parser.add_argument('--data-cfg', type=str, default='cfg/ccmcpe_easy.json', help='coco.data file path')
    parser.add_argument('--test-interval', type=int, default=3, help='test interval')
    # parser.add_argument('--test-interval', type=int, default=1, help='test interval')

    parser.add_argument('--resume', action='store_false', help='resume training flag')
    parser.add_argument('--latest', action='store_true', help='default resume from jde') # 默认从jde模型开始训练， 如果要从上一次的权重中恢复，则加上--not-jde
    parser.add_argument('--print-interval', type=int, default=40, help='print interval')
    parser.add_argument('--lr', type=float, default=1e-2, help='init lr')
    parser.add_argument('--unfreeze-bn', action='store_true', help='unfreeze bn')
    opt = parser.parse_args()
    print("opt\n", opt)
    sys.stdout.flush()
    init_seeds()

    train(
        opt.cfg,
        opt.data_cfg,
        resume=opt.resume,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulated_batches=opt.accumulated_batches,
        opt=opt,
    )
