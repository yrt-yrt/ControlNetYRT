from share import *

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from tutorial_valdataset import MyValDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint

# Configs
resume_path = './models/control_sd15_inis1.ckpt'
batch_size = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
valdataset = MyValDataset()
valdataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = TensorBoardLogger(save_dir="logs", name="my_model")
#logger = ImageLogger(batch_frequency=logger_freq)
#trainer = pl.Trainer(devices="auto", precision=32, callbacks=[logger])
#trainer = pl.Trainer(devices="auto", precision=32, logger=logger)
checkpoint_callback = ModelCheckpoint(dirpath='./models', monitor='train/loss_vlb_epoch', mode='min')
trainer = pl.Trainer(gpus=1, max_epochs=300, precision=32, logger=logger, check_val_every_n_epoch=1, callbacks=[checkpoint_callback])
# Train!
trainer.fit(model, dataloader, valdataloader)
best_model_path = checkpoint_callback.best_model_path
print(best_model_path)
