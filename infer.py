import torch
import yaml
from utils.getter import get_instance

config_path = ''

config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

dev_id = 'cuda:{}'.format(config['gpus']) \
    if torch.cuda.is_available() and config.get('gpus', None) is not None \
    else 'cpu'
device = torch.device(dev_id)
model = get_instance(config['model']).to(device)

# Classify
model.eval()
with torch.no_grad():
    outputs = model(img)

# Print predictions
print('-----')
for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))
