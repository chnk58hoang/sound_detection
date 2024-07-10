from asteroid.models import SuDORMRFImprovedNet
import torch
import yaml

conf = yaml.safe_load(open('conf.yml'))
print(type(conf['filterbank']))
state_dict = torch.load('epoch=1-step=4900.ckpt', map_location='cpu')
state_dict['model_name'] = 'cc'
conf['filterbank'].update(conf['masknet'])
state_dict['model_args'] = conf['filterbank']
model = SuDORMRFImprovedNet.from_pretrained(state_dict)