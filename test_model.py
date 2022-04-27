from stcn import VOSMODEL
from mmcv import Config
import pickle
cfg = Config.fromfile('stcn/config/model/stcn.py')
with open('batch.pkl','rb') as fp:
    batch = pickle.load(fp)
model = VOSMODEL.build(cfg.model)
model.train_step(batch,None)