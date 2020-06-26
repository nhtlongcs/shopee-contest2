import csv
import os
from tqdm import tqdm

import pandas
import torch
import yaml
from PIL import Image
from torchvision import transforms as tvtf

from utils.getter import get_instance

config_path = '/content/shopee-contest2/configs/train/baseline_colab.yaml'
test_dir = '/content/data/test/test/'
csv_test_dir = '/content/data/test.csv'

config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

dev_id = 'cuda:{}'.format(config['gpus']) \
    if torch.cuda.is_available() and config.get('gpus', None) is not None \
    else 'cpu'
device = torch.device(dev_id)

print('load weights-----------------------')
cp_model_dir = ['/content/cp/best_loss 3.693.pth',
                '/content/cp/best_loss 3.693.pth',
                '/content/cp/best_loss 3.693.pth', ]

model = [get_instance(config['model']).to(device)
         for i in range(len(cp_model_dir))]

for i in range(len(cp_model_dir)):
    model[i].load_state_dict(torch.load(cp_model_dir[i])['model_state_dict'])

# Classify
print('generate submission----------------')
model.eval()
with torch.no_grad():
    data = list(csv.reader(open(csv_test_dir)))
    data.pop(0)
    data, _ = zip(*data)
    result = []
    cnt = 0
    for item in tqdm(data):
        item_path = os.path.join(test_dir, item)
        image = Image.open(item_path).convert('RGB')

        tf = tvtf.Compose([tvtf.Resize(224),
                           tvtf.CenterCrop(224),
                           tvtf.ToTensor(),
                           tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
                           ])
        image = tf(image).unsqueeze(0)
        image = image.to(device)
        preds = []

        for i in range(len(cp_model_dir)):
            outputs = model(image)
            probs = torch.softmax(outputs, dim=1).to('cpu')
            pred = probs.argmax(dim=1).numpy()
            preds.append(pred)

        pred = int(max(preds, key=preds.count))
        result.append(pred)

result = list(map(int, result))
result = ["{0:0=2d}".format(x) for x in result]

df = pandas.DataFrame(data={"filename": data, "category": result})
df.to_csv("./submission.csv", sep=',', index=False)
