import argparse
import json
import time
import sys
from pathlib import Path

from sklearn import metrics
from scipy import interpolate
import torch.nn.functional as F
from models import *
from utils.utils import *
from torchvision.transforms import transforms as T
from utils.datasets import LoadImagesAndLabels, JointDataset, collate_fn

import multiprocessing
multiprocessing.set_start_method('spawn',True)


def test(
        cfg,
        data_cfg,
        weights,
        batch_size=16,
        iou_thres=0.5,
        conf_thres=0.3,
        nms_thres=0.45,
        print_interval=40,
):

    # Configure run
    f = open(data_cfg)
    data_cfg_dict = json.load(f)
    f.close()
    #nC = int(data_cfg_dict['classes'])  # number of classes (80 for COCO)
    nC = 1
    test_path = data_cfg_dict['test']
    dataset_root = data_cfg_dict['root']
    cfg_dict = parse_model_cfg(cfg)
    img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]

    # Initialize model
    model = Darknet(cfg_dict, test_emb=False)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'], strict=False)
    else:  # darknet format
        load_darknet_weights(model, weights)

    model = torch.nn.DataParallel(model)
    model.cuda().eval()

    # Get dataloader
    transforms = T.Compose([T.ToTensor()])
    dataset = JointDataset(dataset_root, test_path, img_size, augment=False, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                             num_workers=8, drop_last=False, collate_fn=collate_fn) 

    mean_mAP, mean_R, mean_P, seen = 0.0, 0.0, 0.0, 0
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    sys.stdout.flush()
    outputs, mAPs, mR, mP, TP, confidence, pred_class, target_class, jdict = \
        [], [], [], [], [], [], [], [], []
    AP_accum, AP_accum_count = np.zeros(nC), np.zeros(nC)
    for batch_i, (imgs, targets, paths, shapes, targets_len) in enumerate(dataloader):
        t = time.time()
        output = model(imgs.cuda())
        output = non_max_suppression(output, conf_thres=conf_thres, nms_thres=nms_thres)
        for i, o in enumerate(output):
            if o is not None:
                output[i] = o[:, :6]

        # Compute average precision for each sample
        targets = [targets[i][:int(l)] for i,l in enumerate(targets_len)]
        for si, (labels, detections) in enumerate(zip(targets, output)):
            seen += 1

            if detections is None:
                # If there are labels but no detections mark as zero AP
                if labels.size(0) != 0:
                    mAPs.append(0), mR.append(0), mP.append(0)
                continue

            # Get detections sorted by decreasing confidence scores
            detections = detections.cpu().numpy()
            detections = detections[np.argsort(-detections[:, 4])]


            # If no labels add number of detections as incorrect
            correct = []
            if labels.size(0) == 0:
                # correct.extend([0 for _ in range(len(detections))])
                mAPs.append(0), mR.append(0), mP.append(0)
                continue
            else:
                target_cls = labels[:, 0]

                # Extract target boxes as (x1, y1, x2, y2)
                target_boxes = xywh2xyxy(labels[:, 2:6]) 
                target_boxes[:, 0] *= img_size[0]
                target_boxes[:, 2] *= img_size[0]
                target_boxes[:, 1] *= img_size[1]
                target_boxes[:, 3] *= img_size[1]

                detected = []
                for *pred_bbox, conf, obj_conf  in detections:
                    obj_pred = 0
                    pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                    # Compute iou with target boxes
                    iou = bbox_iou(pred_bbox, target_boxes, x1y1x2y2=True)[0]
                    # Extract index of largest overlap
                    best_i = np.argmax(iou)
                    # If overlap exceeds threshold and classification is correct mark as correct
                    if iou[best_i] > iou_thres and obj_pred == labels[best_i, 0] and best_i not in detected:
                        correct.append(1)
                        detected.append(best_i)
                    else:
                        correct.append(0)

            # Compute Average Precision (AP) per class
            AP, AP_class, R, P = ap_per_class(tp=correct,
                                              conf=detections[:, 4],
                                              pred_cls=np.zeros_like(detections[:, 5]), # detections[:, 6]
                                              target_cls=target_cls)

            # Accumulate AP per class
            AP_accum_count += np.bincount(AP_class, minlength=nC)
            AP_accum += np.bincount(AP_class, minlength=nC, weights=AP)

            # Compute mean AP across all classes in this image, and append to image list
            mAPs.append(AP.mean())
            mR.append(R.mean())
            mP.append(P.mean())

            # Means of all images
            mean_mAP = np.sum(mAPs) / ( AP_accum_count + 1E-16)
            mean_R = np.sum(mR) / ( AP_accum_count + 1E-16)
            mean_P = np.sum(mP) / (AP_accum_count + 1E-16)

        if batch_i % print_interval==0:
            # Print image mAP and running mean mAP
            print(('%11s%11s' + '%11.3g' * 4 + 's') % (seen, dataloader.dataset.nF, mean_P, mean_R, mean_mAP, time.time() - t))
            sys.stdout.flush()
    # Print mAP per class
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    sys.stdout.flush()
    print('AP: %-.4f\n\n' % (AP_accum[0] / (AP_accum_count[0] + 1E-16)))
    sys.stdout.flush()

    # Return mAP
    return mean_mAP, mean_R, mean_P


def test_emb(
            cfg,
            data_cfg,
            weights,
            batch_size=16,
            iou_thres=0.5,
            conf_thres=0.3,
            nms_thres=0.45,
            print_interval=40,
):

    # Configure run
    f = open(data_cfg)
    data_cfg_dict = json.load(f)
    f.close()
    test_paths = data_cfg_dict['test_emb']
    dataset_root = data_cfg_dict['root']
    cfg_dict = parse_model_cfg(cfg)
    img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]

    # Initialize model
    model = Darknet(cfg_dict, test_emb=True)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'], strict=False)
    else:  # darknet format
        load_darknet_weights(model, weights)

    model = torch.nn.DataParallel(model)
    model.cuda().eval()

    # Get dataloader
    transforms = T.Compose([T.ToTensor()])
    dataset = JointDataset(dataset_root, test_paths, img_size, augment=False, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                             num_workers=8, drop_last=False, collate_fn=collate_fn) 
    embedding, id_labels = [], []
    print('Extracting pedestrain features...')
    sys.stdout.flush()
    for batch_i, (imgs, targets, paths, shapes, targets_len) in enumerate(dataloader):
        t = time.time()
        # output = model(imgs.cuda(), targets.cuda(), targets_len.cuda()).squeeze()
        output = model(imgs.cuda(), targets.cuda(), targets_len.cuda())

        for out in output:
            feat, label = out[:-1], out[-1].long()
            if label != -1:
                embedding.append(feat)
                id_labels.append(label)

        if batch_i % print_interval==0:
            print('Extracting {}/{}, # of instances {}, time {:.2f} sec.'.format(batch_i, len(dataloader), len(id_labels), time.time() - t))
            sys.stdout.flush()

    print('Computing pairwise similairity...')
    sys.stdout.flush()
    if len(embedding) <1 :
        return None
    embedding = torch.stack(embedding, dim=0).cuda()
    id_labels = torch.LongTensor(id_labels)
    n = len(id_labels)
    print(n, len(embedding))
    sys.stdout.flush()
    assert len(embedding) == n

    embedding = F.normalize(embedding, dim=1)
    pdist = torch.mm(embedding, embedding.t()).cpu().numpy()
    gt = id_labels.expand(n,n).eq(id_labels.expand(n,n).t()).numpy()
    
    up_triangle = np.where(np.triu(pdist)- np.eye(n)*pdist !=0)
    pdist = pdist[up_triangle]
    gt = gt[up_triangle]

    far_levels = [ 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    far,tar,threshold = metrics.roc_curve(gt, pdist)
    interp = interpolate.interp1d(far, tar)
    tar_at_far = [interp(x) for x in far_levels]
    for f,fa in enumerate(far_levels):
        print('TPR@FAR={:.7f}: {:.4f}'.format(fa, tar_at_far[f]))
        sys.stdout.flush()
    return tar_at_far


def test_giou(
        cfg,
        data_cfg,
        weights,
        batch_size=16,
        iou_thres=0.5,
        conf_thres=0.3,
        nms_thres=0.45,
        print_interval=40,
):
    print("test giou...")
    sys.stdout.flush()
    # Configure run
    f = open(data_cfg)
    data_cfg_dict = json.load(f)
    f.close()
    #nC = int(data_cfg_dict['classes'])  # number of classes (80 for COCO)
    nC = 1
    test_path = data_cfg_dict['test']
    dataset_root = data_cfg_dict['root']
    cfg_dict = parse_model_cfg(cfg)
    img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]  
    print("loading data...")
    sys.stdout.flush()
    # Get dataloader
    transforms = T.Compose([T.ToTensor()])
    dataset = JointDataset(dataset_root, test_path, img_size, augment=False, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                             num_workers=8, drop_last=False, collate_fn=collate_fn) 

    print("building model...")
    sys.stdout.flush()
    # Initialize model
    model = Darknet(cfg_dict, test_emb=False)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'], strict=False)
    else:  # darknet format
        load_darknet_weights(model, weights)

    model = torch.nn.DataParallel(model)
    model.cuda().eval()

    mean_mAP, mean_R, mean_P, seen = 0.0, 0.0, 0.0, 0
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    sys.stdout.flush()
    mAPs, mR, mP, TP = [], [], [], []
    AP_accum, AP_accum_count = np.zeros(nC), np.zeros(nC)
    # outputs, mAPs, mR, mP, TP, confidence, pred_class, target_class, jdict = \
    #     [], [], [], [], [], [], [], [], []

    giou_mean_mAP, giou_mean_R, giou_mean_P, giou_seen = 0.0, 0.0, 0.0, 0
    sys.stdout.flush()
    giou_mAPs, giou_mR, giou_mP, giou_TP = [], [], [], []
    giou_AP_accum, giou_AP_accum_count = np.zeros(nC), np.zeros(nC)
    
    print("Computing mAP...")
    sys.stdout.flush()
    for batch_i, (imgs, targets, paths, shapes, targets_len) in enumerate(dataloader):
        t = time.time()
        output = model(imgs.cuda())
        output = non_max_suppression(output, conf_thres=conf_thres, nms_thres=nms_thres)
        for i, o in enumerate(output):
            if o is not None:
                output[i] = o[:, :6]

        # Compute average precision for each sample
        targets = [targets[i][:int(l)] for i,l in enumerate(targets_len)]
        for si, (labels, detections) in enumerate(zip(targets, output)):
            seen += 1

            if detections is None:
                # If there are labels but no detections mark as zero AP
                if labels.size(0) != 0:
                    mAPs.append(0), mR.append(0), mP.append(0)
                continue

            # Get detections sorted by decreasing confidence scores
            detections = detections.cpu().numpy()
            detections = detections[np.argsort(-detections[:, 4])]

            # If no labels add number of detections as incorrect
            correct = []
            giou_correct = []
            if labels.size(0) == 0:
                # correct.extend([0 for _ in range(len(detections))])
                mAPs.append(0), mR.append(0), mP.append(0)
                giou_mAPs.append(0), giou_mR.append(0), giou_mP.append(0)
                continue
            else:
                target_cls = labels[:, 0]

                # Extract target boxes as (x1, y1, x2, y2)
                target_boxes = xywh2xyxy(labels[:, 2:6]) 
                target_boxes[:, 0] *= img_size[0]
                target_boxes[:, 2] *= img_size[0]
                target_boxes[:, 1] *= img_size[1]
                target_boxes[:, 3] *= img_size[1]

                detected = []
                giou_detected = []
                for *pred_bbox, conf, obj_conf  in detections:
                    obj_pred = 0
                    pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)

                    """" mAP@IoU """
                    # Compute iou with target boxes
                    iou = bbox_iou(pred_bbox, target_boxes, x1y1x2y2=True)[0]
                    # Extract index of largest overlap
                    best_i = np.argmax(iou)
                    # If overlap exceeds threshold and classification is correct mark as correct
                    if iou[best_i] > iou_thres and obj_pred == labels[best_i, 0] and best_i not in detected:
                        correct.append(1)
                        detected.append(best_i)
                    else:
                        correct.append(0)
                    
                    """ mAP@GIoU """
                    giou = bbox_giou(pred_bbox, target_boxes, x1y1x2y2=True)[0]  # 为了简便，现在还是用iou，之后要改为giou
                    best_i = np.argmax(giou)
                    if giou[best_i] > iou_thres and obj_pred == labels[best_i, 0] and best_i not in giou_detected:
                        giou_correct.append(1)
                        giou_detected.append(best_i)
                    else:
                        giou_correct.append(0)

            """" mAP@IoU """
            # Compute Average Precision (AP) per class
            AP, AP_class, R, P = ap_per_class(tp=correct,
                                              conf=detections[:, 4],
                                              pred_cls=np.zeros_like(detections[:, 5]), # detections[:, 6]
                                              target_cls=target_cls)

            # Accumulate AP per class
            AP_accum_count += np.bincount(AP_class, minlength=nC)
            AP_accum += np.bincount(AP_class, minlength=nC, weights=AP)

            # Compute mean AP across all classes in this image, and append to image list
            mAPs.append(AP.mean())
            mR.append(R.mean())
            mP.append(P.mean())

            # Means of all images
            mean_mAP = np.sum(mAPs) / ( AP_accum_count + 1E-16)
            mean_R = np.sum(mR) / ( AP_accum_count + 1E-16)
            mean_P = np.sum(mP) / (AP_accum_count + 1E-16)

            """ mAP@GIoU """
            # Compute Average Precision (AP) per class
            giou_AP, giou_AP_class, giou_R, giou_P = ap_per_class(tp=giou_correct,
                                              conf=detections[:, 4],
                                              pred_cls=np.zeros_like(detections[:, 5]), # detections[:, 6]
                                              target_cls=target_cls)

            # Accumulate AP per class
            giou_AP_accum_count += np.bincount(giou_AP_class, minlength=nC)
            giou_AP_accum += np.bincount(giou_AP_class, minlength=nC, weights=giou_AP)

            # Compute mean AP across all classes in this image, and append to image list
            giou_mAPs.append(giou_AP.mean())
            giou_mR.append(giou_R.mean())
            giou_mP.append(giou_P.mean())

            # Means of all images
            giou_mean_mAP = np.sum(giou_mAPs) / ( giou_AP_accum_count + 1E-16)
            giou_mean_R = np.sum(giou_mR) / ( giou_AP_accum_count + 1E-16)
            giou_mean_P = np.sum(giou_mP) / (giou_AP_accum_count + 1E-16)

        if batch_i % print_interval==0:
            # Print image mAP and running mean mAP
            print(('%11s%11s' + '%11.3g' * 4 + 's') % (seen, dataloader.dataset.nF, mean_P, mean_R, mean_mAP, time.time() - t))
            sys.stdout.flush()
            print(('GIOU:%6s%11s' + '%11.3g' * 4 + 's') % (seen, dataloader.dataset.nF, giou_mean_P, giou_mean_R, giou_mean_mAP, time.time() - t))
            sys.stdout.flush()

    # Print mAP per class
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    sys.stdout.flush()
    print('AP: %-.4f\n\n' % (AP_accum[0] / (AP_accum_count[0] + 1E-16)))
    sys.stdout.flush()
    print('giou_AP: %-.4f\n\n' % (giou_AP_accum[0] / (giou_AP_accum_count[0] + 1E-16)))
    sys.stdout.flush()

    # Return mAP
    return mean_mAP, mean_R, mean_P



if __name__ == '__main__':
    # 576x320 test可以batch=16单卡 train可以batch=8单卡
    # 864x480 test可以batch=8单卡 train可以batch=4单卡
    # 1088x608 test可以batch=8单卡 train可以batch=4单卡
    parser = argparse.ArgumentParser(prog='test.py')
    # parser.add_argument('--weights', type=str, default='/home/master/kuanzi/weights/diou_65_epoch.pt', help='path to weights file')
    parser.add_argument('--weights', type=str, default='/home/master/kuanzi/weights/60_epoch_diou_arcface.pt', help='path to weights file')
    # parser.add_argument('--batch-size', type=int, default=40, help='size of each image batch')
    parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
    # parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3_864x480.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='cfg/ccmcpe.json', help='data config')
    # parser.add_argument('--data-cfg', type=str, default='cfg/ccmcpe_easy.json', help='data config')
    
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--print-interval', type=int, default=40, help='size of each image dimension')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    with torch.no_grad():
        mAP, R, P = test_giou( opt.cfg,
                opt.data_cfg,
                opt.weights,
                opt.batch_size,
                opt.iou_thres,
                opt.conf_thres,
                opt.nms_thres,
                opt.print_interval)
        print ("test.test:\t", mAP, "\t", R, "\t", P)
        sys.stdout.flush()

        # test_mapgiou.test_emb(cfg, data_cfg, weights=latest, batch_size=batch_size, print_interval=40)
        res = test_emb(
            opt.cfg,
            opt.data_cfg,
            opt.weights,
            opt.batch_size,
            opt.iou_thres,
            opt.conf_thres,
            opt.nms_thres,
            opt.print_interval,
        )

