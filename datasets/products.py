import os 
import csv
from PIL import Image 
import torch 
from torch.utils import data 
from torchvision import transforms as tvtf
import numpy as np 

class Product(data.Dataset):
    def __init__(self, root_dir, csv_path):
        self.root_dir = root_dir 
        data = list(csv.reader(open(csv_path))) 
        self.dirs = [
            f'{root_dir}/{str(lbl).zfill(2)}/{_id}'
                for _id, lbl in data
        ]
        _, self.labels = zip(*data)
        self.labels = list(map(int, self.labels))
        
    def __getitem__(self, i):
        lbl = self.labels[i] 
        
        img = Image.open(self.dirs[i])
        transform_ = tvtf.Compose([tvtf.Resize((224,224)), tvtf.ToTensor(),
                              tvtf.Normalize((0.5,), (0.5,)),
                              ])
        img = transform_(img)
        return img, lbl
        
    def __len__(self):
        return len(self.dirs)

if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    import pandas as pd 
    from pathlib import Path 
    path = 'data/trainingSet/trainingSet'
    # _id = []
    # _lbl = []
    # for lbl_p in sorted(Path(path).iterdir()):
    #     try:
    #         lbl = str(lbl_p).replace(path + '/','')
    #         print('Label ', lbl, lbl_p)
    #         for img_p in sorted(lbl_p.iterdir()):
    #             img = str(img_p).replace(str(lbl_p) + '/','')
    #             print(img)
    #             _id.append(img)
    #             _lbl.append(lbl)
    #     except:
    #         pass
    # train_csv = pd.DataFrame()
    # train_csv['id'] = _id 
    # train_csv['label'] = _lbl 
    # train_csv.to_csv('train.csv', index = False)
    dataset = Product(path, csv_path = 'train.csv')
    
    for i, (inp, lbl) in enumerate(dataset):
        if (i == 10):
            break
        print(lbl)
        plt.subplot(2, 5, i + 1)
        plt.imshow(inp)
    
    plt.show()
    plt.close()
