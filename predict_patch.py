import torch
import numpy as np
import os
import argparse
from torch.utils.data import Dataset, DataLoader
import cv2
from skimage import io
from SolarNet import SolarNet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class generateDataset(Dataset):
    def __init__(self, dirFiles, img_size, colordim, isTrain=True):
        self.isTrain = isTrain
        self.dirFiles = dirFiles
        self.nameFiles = [name for name in os.listdir(dirFiles) if os.path.isfile(os.path.join(dirFiles, name))]
        self.numFiles = len(self.nameFiles)
        self.img_size = img_size
        self.colordim = colordim
        print(f'number of files: {self.numFiles}')
        
    def __getitem__(self, index):
        filename = os.path.join(self.dirFiles, self.nameFiles[index])
        img = io.imread(filename)
        img = img[:, :, 0:3] / 255.0
        img = torch.from_numpy(img).float()
        img = img.permute(2, 0, 1)  # More efficient than multiple transposes
        imgName = os.path.splitext(self.nameFiles[index])[0]
        return img, imgName
        
    def __len__(self):
        return self.numFiles

def main(args):
    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
    model = SolarNet(in_channels=args.colordim, n_class=args.num_class)
    if args.cuda:
        model = model.cuda()
    
    model.load_state_dict(torch.load(args.pretrain_net))
    model.eval()
    
    predDataset = generateDataset(args.pre_root_dir, args.img_size, args.colordim, isTrain=False)
    predLoader = DataLoader(dataset=predDataset, batch_size=args.predictbatchsize, num_workers=args.threads)
    
    with torch.no_grad():
        for batch_idx, (batch_x, batch_name) in enumerate(predLoader):
            if args.cuda:
                batch_x = batch_x.cuda()
            
            out = model(batch_x)
            out2, out3 = out[1], out[2]
            
            _, pred_label2 = torch.max(out2, 1)
            pred_label_np2 = pred_label2.cpu().numpy()
            
            _, pred_label3 = torch.max(out3, 1)
            pred_label_np3 = pred_label3.cpu().numpy()
            
            for id in range(len(batch_name)):
                pred_label_single2 = pred_label_np2[id, :, :]
                predLabel_filename2 = os.path.join(args.preDir2, batch_name[id] + '.png')
                cv2.imwrite(predLabel_filename2, pred_label_single2.astype(np.uint8))
                
                pred_label_single3 = pred_label_np3[id, :, :]
                predLabel_filename3 = os.path.join(args.preDir3, batch_name[id] + '.png')
                cv2.imwrite(predLabel_filename3, pred_label_single3.astype(np.uint8))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--predictbatchsize', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--colordim', type=int, default=3)
    parser.add_argument('--pretrain_net', type=str, default='./checkpoint-batchsize4-learning_rate0.001-optimizersgd/best_model.pth')
    parser.add_argument('--pre_root_dir', type=str, default='./wbf_data/n/')
    parser.add_argument('--num_class', type=int, default=9)
    parser.add_argument('--preDir2', type=str, default='./roof/')
    parser.add_argument('--preDir3', type=str, default='./sroof/')
    # parser.add_argument('--preDir2', type=str, default='./predictionroofsegment/')
    # parser.add_argument('--preDir3', type=str, default='./predictionsuperstructure/')
    
    args = parser.parse_args()
    os.makedirs(args.preDir2, exist_ok=True)
    os.makedirs(args.preDir3, exist_ok=True)
    main(args)