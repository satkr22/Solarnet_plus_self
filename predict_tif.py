import numpy as np
import torch
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from skimage import io
import itertools
from SolarNet import SolarNet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def sliding_window(img, step=128, window_size=(256, 256)):
    for x in range(0, img.shape[0], step):
        if x + window_size[0] > img.shape[0]:
            x = img.shape[0] - window_size[0]
        for y in range(0, img.shape[1], step):
            if y + window_size[1] > img.shape[1]:
                y = img.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]

def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def count_sliding_window(img, step=128, window_size=(256, 256)):
    nSW = 0
    for x in range(0, img.shape[0], step):
        if x + window_size[0] > img.shape[0]:
            x = img.shape[0] - window_size[0]
        for y in range(0, img.shape[1], step):
            if y + window_size[1] > img.shape[1]:
                y = img.shape[1] - window_size[1]
            nSW += 1
    return nSW

def main(args):
    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
    model = SolarNet(in_channels=args.colordim, n_class=args.num_class)
    if args.cuda:
        model = model.cuda()
    
    model.load_state_dict(torch.load(args.pretrain_net))
    model.eval()
    
    with torch.no_grad():
        print('Inferencing begin')
        img = io.imread(args.pred_rgb_file).astype('float32') / 255.0
        img = img[:, :, 0:3]
        
        pred = np.zeros(img.shape[:2] + (6,))  # 6 roof orientation classes
        pred2 = np.zeros(img.shape[:2] + (args.num_class,))
        
        batch_total = count_sliding_window(img, step=args.step, window_size=(args.img_size, args.img_size)) // args.predictbatchsize
        print(f'Total Batch: {batch_total}')

        for coords in grouper(args.predictbatchsize, sliding_window(img, step=args.step, window_size=(args.img_size, args.img_size))):
            image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
            image_patches = np.asarray(image_patches)
            image_patches = torch.from_numpy(image_patches)
            
            if args.cuda:
                image_patches = image_patches.cuda()
            
            outs = model(image_patches)
            outs_np = outs[1].cpu().numpy()
            outs_np2 = outs[2].cpu().numpy()

            for out, (x, y, w, h) in zip(outs_np, coords):
                out = out.transpose((1,2,0))
                pred[x:x+w, y:y+h] += out
            
            for out2, (x, y, w, h) in zip(outs_np2, coords):
                out2 = out2.transpose((1,2,0))
                pred2[x:x+w, y:y+h] += out2
                       
        pred = np.argmax(pred, axis=-1)
        io.imsave(args.pre_result, pred.astype(np.uint8))
        pred2 = np.argmax(pred2, axis=-1)
        io.imsave(args.pre_result2, pred2.astype(np.uint8))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--predictbatchsize', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--colordim', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--step', type=int, default=128)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--pretrain_net', type=str, default='./checkpoint-batchsize4-learning_rate0.001-optimizersgd/best_model.pth')
    parser.add_argument('--pred_rgb_file', type=str, default='large_img.tif')
    parser.add_argument('--num_class', type=int, default=9)
    parser.add_argument('--pre_result', type=str, default='large_rf.png')
    parser.add_argument('--pre_result2', type=str, default='large_su.png')
    
    args = parser.parse_args()
    main(args)