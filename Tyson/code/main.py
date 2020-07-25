import os

import torch.optim as optim

from functools import partial
from argparse import ArgumentParser

from unet import UNet2D
from model import Model
from utils import MetricList
from metrics import LogNLLLoss, classwise_iou, classwise_f1, make_weighted_metric
from dataset import JointTransform2D, ImageToImage2D, Image2D

parser = ArgumentParser()
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--metrics', default='f1', type=str)
parser.add_argument('--in_channels', default=3, type=int)
parser.add_argument('--out_channels', default=2, type=int)
parser.add_argument('--depth', default=5, type=int)
parser.add_argument('--width', default=32, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--save_freq', default=0, type=int)
parser.add_argument('--save_model', default=1, type=int)
parser.add_argument('--model_name', type=str, default='model')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--crop', type=int, default=None)
args = parser.parse_args()

if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None

train_path = os.path.join('../data/', args.dataset, 'train')
val_path = os.path.join('../data/', args.dataset, 'val')
output_path = os.path.join('../output/', args.dataset)
model_path = os.path.join('../model/', args.dataset)

tf_train = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=None, long_mask=True)
tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)

train_dataset = ImageToImage2D(train_path, tf_train)
val_dataset = ImageToImage2D(val_path, tf_val)

predict_val_dataset = Image2D(val_path)
predict_output_dataset = Image2D(output_path)

conv_depths = [int(args.width*(2**k)) for k in range(args.depth)] # standard UNet uses [32, 64, ..., 1024]
unet = UNet2D(args.in_channels, args.out_channels, conv_depths)
loss = LogNLLLoss()
optimizer = optim.Adam(unet.parameters(), lr=args.lr)

if args.metrics == 'both':
	f1_score = make_weighted_metric(classwise_f1)
	jaccard_index = make_weighted_metric(classwise_iou)

	metric_list = MetricList({'jaccard': partial(jaccard_index), 'f1': partial(f1_score)}, 
							 device = args.device)
elif args.metrics == 'f1':
	f1_score = make_weighted_metric(classwise_f1)

	metric_list = MetricList({'f1': partial(f1_score)},
							 device = args.device)
elif args.metrics == 'jaccard':
	jaccard_index = make_weighted_metric(classwise_iou)

	metric_list = MetricList({'jaccard': partial(jaccard_index)},
							 device = args.device)

model = Model(unet, loss, optimizer, model_path, device=args.device)

model.fit_dataset(train_dataset, n_epochs=args.epochs, n_batch=args.batch_size,
                  shuffle=True, val_dataset=val_dataset, save_freq=args.save_freq,
                  save_model=args.save_model, predict_dataset=predict_val_dataset,
                  metric_list=metric_list, verbose=True)

model.predict_dataset(predict_output_dataset, os.path.join(output_path,'masks'))