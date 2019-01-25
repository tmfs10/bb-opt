import pickle
import torch
from torch.nn import Sequential, Linear
from torch.utils.data import TensorDataset, DataLoader
from itertools import cycle
import os.path
from os.path import exists,join
import sys,argparse
#import pyro
sys.path.append('/cluster/geliu/bayesian/')
sys.path.append('/cluster/geliu/bayesian/bb_opt/src/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from bb_opt.src.deep_ensemble_saber import (
    NNEnsemble,
    RandomNN,
)
from bb_opt.src.dna_bopt import knn_density
from bb_opt.src.utils import get_path, load_checkpoint, save_checkpoint, jointplot, load_data_saber, load_data_saber2,load_data_gc,load_data_uci
from gpu_utils.utils import gpu_init
from collections import namedtuple

_Input_Labels = namedtuple("Input_Labels", ["inputs", "labels"])
_Dataset = namedtuple("Dataset", ["train", "val", "test","top"])

def parse_args():
    print(1)
    parser = argparse.ArgumentParser(description="Launch a list of commands on EC2.")
    parser.add_argument("-g", "--gpu",dest="gpu",type=int,default=0,help="specify which gpu to use")
    parser.add_argument("-w","--loss",dest="loss_type",type=str,default='maxvar',help="specify which loss function to use")
    parser.add_argument("-s","--sample",dest="sample_size",type=float,default=1,help="specify how many uniformly sampled samples needed")
    parser.add_argument("-i", "--hyper", dest="hyper",type=float, default=1.0,help="hyperparam for extra loss")
    parser.add_argument("-m", "--modelpath",dest='model_dir',help="File to save model")
    parser.add_argument("-ex", "--extra",dest='extra_random',type=bool,default=False,help="Extra randomness")
    parser.add_argument("-e", "--ensize",dest="ensem_size",type=int,default=4,help="size of ensemble")
    parser.add_argument("-l", "--learnrate",dest="lr",type=float,default=0.001,help="learning rate")
    parser.add_argument("-hd", "--hidden",dest="hidden",type=int,default=100,help="num hidden neurons")
    parser.add_argument("-l2", "--l2",dest="l2",type=float,default=1e-4,help="l2 weight")
    parser.add_argument("-d","--data",dest="data_type",type=str,default='rand',help="specify which train/val/test")
    parser.add_argument("-ep", "--epoch",dest="epoch",type=int,default=700,help="num training epoch")
    parser.add_argument("-sg", "--single",dest='single_layer',type=str,default='Y',help="single layer?")
    parser.add_argument("-ds", "--dataset",dest='dataset',type=str,default='toy',help="which dataset to use")
    parser.add_argument("-p", "--top",dest="percent",type=float,default=0.05,help="percentage of top data heldout")
    parser.add_argument("-sd", "--seed",dest="seed",type=int,default=0,help="random seed")
    parser.add_argument("-k", "--knn",dest="knn",type=str,default='mean',help="using max or mean for knn density")
    return parser.parse_args()

def sample_uniform(out_size,min_val,max_val):
    #low=np.tile(min_val,(out_size[0],1))
    #hi=np.tile(max_val,(out_size[0],1))
    z=np.random.uniform(low=min_val,high=max_val,size=out_size)
    out_data = torch.from_numpy(z).float().cuda()
    return out_data

def func(x):
    noise=np.random.randn(len(x))*0.02
    y=0.3*x+0.3*np.sin(2*np.pi*(x))+0.3*np.sin(4*np.pi*(x))+noise
    return y

if __name__ == "__main__":
	args = parse_args()
	#gpu_id = gpu_init()
	torch.cuda.set_device(args.gpu)
	model_path=os.path.join(args.model_dir,'{}_{}_{}_{}_{}_{}_{}_{}.pth'.format(args.dataset,args.loss_type,args.sample_size,args.hyper,args.lr,args.l2,args.ensem_size,args.hidden))
	print(model_path)
	if exists(model_path.replace(".pth", ".txt")):
		a=pd.read_csv(model_path.replace(".pth", ".txt"))
		if not len(a)%100==0:
			print("file exists!")
			sys.exit(0)
	device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
	np.random.seed(args.seed)
	band=np.concatenate([np.arange(-0.8,-0.3,0.00001),np.arange(0.5,1,0.00001)],axis=0)
	x_idx=np.random.randint(len(band),size=50)
	train_inputs=band[x_idx]
	train_labels=func(train_inputs)
	train_inputs=np.reshape(train_inputs,(len(x_idx),1))
	x_idx2=np.random.randint(len(band),size=50)
	val_inputs=band[x_idx2]
	val_labels=func(val_inputs)
	val_inputs=np.reshape(val_inputs,(len(x_idx2),1))
	test_inputs=val_inputs
	test_labels=val_labels
	top_inputs=np.concatenate([np.arange(-1.5,-0.8,0.001),np.arange(-0.3,0.5,0.001),np.arange(1.0,1.5,0.001)],axis=0)
	top_labels=func(top_inputs)
	top_inputs=np.reshape(top_inputs,(len(top_inputs),1))
	train_inputs = torch.tensor(train_inputs).float().to(device)
	val_inputs = torch.tensor(val_inputs).float().to(device)
	test_inputs = torch.tensor(test_inputs).float().to(device)
	train_labels = torch.tensor(train_labels).float().to(device)
	val_labels = torch.tensor(val_labels).float().to(device)
	test_labels = torch.tensor(test_labels).float().to(device)
	top_inputs = torch.tensor(top_inputs).float().to(device)
	top_labels = torch.tensor(top_labels).float().to(device)
	data = _Dataset(
	*[  
	    _Input_Labels(inputs, labels)
	    for inputs, labels in zip(
		[train_inputs, val_inputs, test_inputs,top_inputs],
		[train_labels, val_labels, test_labels,top_labels],
	    )
	]
	)
	n_hidden = args.hidden
	print(data.train.inputs.shape)
	n_inputs = data.train.inputs.shape[1]
	batch_size=50
	train_loader = DataLoader(TensorDataset(data.train.inputs, data.train.labels),batch_size=batch_size,shuffle=True)
	n_models = args.ensem_size
	adversarial_epsilon = None
	model = NNEnsemble.get_model(n_inputs, batch_size, n_models, n_hidden, adversarial_epsilon, device,extra_random=args.extra_random,single_layer=(args.single_layer=='Y'))
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
	myhist={'train_nll':[],'train_nll_mix':[],'train_mse':[],'train_loss':[],'train_var':[],'test_nll':[],'test_nll_mix':[],'test_mse':[],'test_var':[],'val_nll':[],'val_nll_mix':[],'val_mse':[],'top_nll':[],'top_nll_mix':[],'top_mse':[],'top_var':[]}
	min_loss= float("inf")
	early_buff=0
	train_newloss=[]
	epoch = 0
	min_val=-1.5
	max_val=1.5
	print(args.loss_type)
	for epoch in range(epoch, epoch + args.epoch):
		model.train()
		for batch in train_loader:
			inputs, labels = batch
			#print(labels)
			out_size=(int(inputs.shape[0]*args.sample_size),n_inputs)
			optimizer.zero_grad()
			means, variances = model(inputs)
			negative_log_likelihood, mse = NNEnsemble.compute_negative_log_likelihood(labels, means, variances, return_mse=True)
			if args.loss_type=="maxvar":
				out_data = sample_uniform(out_size,min_val,max_val)
				means_o, variances_o = model(out_data)
				var=(means_o.var(dim=0).mean())
				loss=negative_log_likelihood-args.hyper*var
				train_newloss.append(var.item())
			elif args.loss_type=="defmean":
				out_data = sample_uniform(out_size,min_val,max_val)
				means_o, variances_o = model(out_data)
				nll=NNEnsemble.compute_negative_log_likelihood(default_mean,means_o,variances_o)
				loss=negative_log_likelihood+args.hyper*nll
				train_newloss.append(nll.item())
			elif args.loss_type=="normal":
				loss=negative_log_likelihood
			elif args.loss_type=="invar":
				var=(means.var(dim=0).mean())
				loss=negative_log_likelihood-args.hyper*var
				train_newloss.append(var.item())
			elif args.loss_type=="2var":
				out_data = sample_uniform(out_size)
				means_o, variances_o = model(out_data)
				var=(0.5*means_o.var(dim=0).mean())+(0.5*means.var(dim=0).mean())
				loss=negative_log_likelihood-args.hyper*var
				train_newloss.append(var.item())
			elif args.loss_type=='idvar':
				out_data = sample_uniform(out_size,min_val,max_val)
				means_o, variances_o = model(out_data)
				var=((means_o.var(dim=0)*knn_density(inputs,out_data,2)).mean())
				loss=negative_log_likelihood-args.hyper*var
				train_newloss.append(var.item())
			elif args.loss_type=='idvar-max':
				out_data = sample_uniform(out_size,min_val,max_val)
				means_o, variances_o = model(out_data)
				var=((means_o.var(dim=0)*knn_density_max(inputs,out_data,5)).mean())
				loss=negative_log_likelihood-args.hyper*var
				train_newloss.append(var.item())
			loss.backward()
			optimizer.step()
		model.eval()
		with torch.no_grad():
			means, variances = model(data.train.inputs)
			negative_log_likelihood1,negative_log_likelihood2, mse = NNEnsemble.report_metric(data.train.labels, means,variances, return_mse=True)
			loss=negative_log_likelihood1
			var=(means.var(dim=0).mean())
			myhist['train_var'].append(var.item())
			myhist['train_nll_mix'].append(negative_log_likelihood1.item())
			myhist['train_nll'].append(negative_log_likelihood2.item())
			myhist['train_mse'].append(mse.item())
			myhist['train_loss'].append(loss.item())
			means, variances = model(data.test.inputs)
			var=(means.var(dim=0).mean())
			myhist['test_var'].append(var.item())
			negative_log_likelihood1,negative_log_likelihood2,mse = NNEnsemble.report_metric(data.test.labels, means, variances, return_mse=True)
			myhist['test_nll_mix'].append(negative_log_likelihood1.item())
			myhist['test_nll'].append(negative_log_likelihood2.item())
			myhist['test_mse'].append(mse.item())
			means, variances = model(data.top.inputs)
			var=(means.var(dim=0).mean())
			myhist['top_var'].append(var.item())
			negative_log_likelihood1,negative_log_likelihood2,mse = NNEnsemble.report_metric(data.top.labels, means, variances, return_mse=True)
			myhist['top_nll_mix'].append(negative_log_likelihood1.item())
			myhist['top_nll'].append(negative_log_likelihood2.item())
			myhist['top_mse'].append(mse.item())
			means, variances = model(data.val.inputs)
			negative_log_likelihood1,negative_log_likelihood2, mse = NNEnsemble.report_metric(data.val.labels, means, variances, return_mse=True)
			myhist['val_nll_mix'].append(negative_log_likelihood1.item())
			myhist['val_nll'].append(negative_log_likelihood2.item())
			myhist['val_mse'].append(mse.item())
			if negative_log_likelihood2<=min_loss:
				min_loss=negative_log_likelihood2.item()
				model.save_model(model_path, optimizer)
				means, variances = model(data.top.inputs)
				with open(model_path.replace(".pth", "_pred.pkl"), 'wb') as f:
					pickle.dump({'means':means,'var':variances}, f)
				tnll=myhist['train_nll'][-1]
				vnll=myhist['val_nll'][-1]
				tenll=myhist['test_nll'][-1]
				print('new minimum found, saving model to file system')
				early_buff=0
				print(f'[E{epoch}] Train NLL = {tnll:,.0f}. Valid NLL = {vnll:,.0f}. Test NLL = {tenll:,.0f}')
			elif early_buff>10:
				break
			else:
				early_buff=early_buff+1
		if epoch % 100 == 0:
			tnll2=myhist['train_nll'][-1]
			tnll=myhist['train_nll_mix'][-1]
			tenll2=myhist['test_nll'][-1]
			tenll=myhist['test_nll_mix'][-1]
			tmse=myhist['train_mse'][-1]
			tloss=myhist['train_loss'][-1]
			tvar=myhist['train_var'][-1]
			temse=myhist['test_mse'][-1]
			tevar=myhist['test_var'][-1]
			vnll2=myhist['val_nll'][-1]
			vnll=myhist['val_nll_mix'][-1]
			vmse=myhist['val_mse'][-1]
			tonll2=myhist['top_nll'][-1]
			tonll=myhist['top_nll_mix'][-1]
			tomse=myhist['top_mse'][-1]
			tovar=myhist['top_var'][-1]
			print(f'[E{epoch}] Train NLL1 = {tnll:,.0f}. Train NLL2 = {tnll2:,.0f}.Train MSE = {tmse:,.0f}. Train LOSS = {tloss:,.0f}. Train VAR = {tvar:,.0f}. Val NLL1={vnll:,.0f}.Val NLL2={vnll2:,.0f}. Val MSE={vmse:,.0f}. Test NLL2 = {tenll:,.0f}.Test NLL2 = {tenll2:,.0f}. Test MSE = {temse:,.0f}. Test VAR = {tevar:,.0f}.')
			#pd.DataFrame(myhist).to_csv(model_path.replace(".pth", ".txt"))
	pd.DataFrame(myhist).to_csv(model_path.replace(".pth", ".txt"))
