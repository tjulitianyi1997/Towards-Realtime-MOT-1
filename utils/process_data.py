import os.path as osp
import numpy as np

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

out_labels = 'mot.train'
seq_root = 'images/train'
label_root = 'labels/train'
seqs = [s for s in os.listdir(seq_root) if s.endswith('SDP')]

tid_curr = 0
tid_last = -1
for seq in seqs:
    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=')+8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=')+9:seq_info.find('\nimExt')])
    
    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

    seq_label_root = osp.join(label_root, seq, 'img1')
    mkdirs(seq_label_root)
    
    for fid, tid, x, y, w, h, mark, label, _ in gt:
        if mark==0 or not label==1:
            continue
        fid = int(fid)
        tid = int(tid)
        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid
        x += w/2
        y += h/2
        label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid)) 
        label_str ='0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                tid_curr, x/seq_width, y/seq_height, w/seq_width, h/seq_height)
        with open(label_fpath, 'a') as f:
            f.write(label_str)

tmp_list ='mot.train.tmp'
os.system('find {} -type f>{}'.format(label_root, tmp_list))
with open(tmp_list, 'r') as f:
    s = f.read().replace('.txt', '.jpg').replace(
            'labels', '/home/wangzd/MOT/yolov3.pedestrain.ae/data/mot17/images')
with open(out_labels, 'w') as f:
    f.write(s)

os.system('rm {}'.format(tmp_list))