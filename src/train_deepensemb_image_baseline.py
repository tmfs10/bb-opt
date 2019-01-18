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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from bb_opt.src.deep_ensemble_saber import (
    NNEnsemble,
    RandomNN,
)
from bb_opt.src.utils import get_path, load_checkpoint, save_checkpoint, jointplot, load_data_wiki
from gpu_utils.utils import gpu_init

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
    parser.add_argument("-dp", "--depth",dest="depth",type=int,default=16,help="depth of resnet")
    parser.add_argument("-wd", "--widen",dest="widen",type=int,default=8,help="widen factor of resnet")
    parser.add_argument("-dr", "--dropout",dest="dropout",type=float,default=0.3,help="dropout rate of resnet")
    parser.add_argument("-hd", "--hidden",dest="hidden",type=int,default=100,help="num hidden neurons")
    parser.add_argument("-l2", "--l2",dest="l2",type=float,default=1e-4,help="l2 weight")
    parser.add_argument("-d","--split",dest="split_type",type=str,default='gender',help="specify which train/val/test")
    parser.add_argument("-ep", "--epoch",dest="epoch",type=int,default=200,help="num training epoch")
    parser.add_argument("-sg", "--single",dest='single_layer',type=str,default='Y',help="single layer?")
    parser.add_argument("-ds", "--dataset",dest='dataset',type=str,default='imdb',help="which dataset to use")
    parser.add_argument("-p", "--top",dest="percent",type=float,default=0.1,help="percentage of top data heldout")
    parser.add_argument("-sd", "--seed",dest="seed",type=int,default=0,help="random seed")
    return parser.parse_args()

def sample_uniform(out_size):
    z=np.random.randint(256,size=(out_size,3,32,32))
    out_data=torch.from_numpy(z).float().cuda()
    return out_data

if __name__ == "__main__":
	args = parse_args()
	#gpu_id = gpu_init()
	torch.cuda.set_device(args.gpu)
	model_path=os.path.join(args.model_dir,'{}_{}_{}_{}_{}_{}_{}_{}.pth'.format(args.dataset.split('_8mers')[0],args.loss_type,args.sample_size,args.hyper,args.lr,args.l2,args.ensem_size,args.hidden))
	print(model_path)
	if exists(model_path.replace(".pth", ".txt")):
		print("file exists!")
		sys.exit(0)
	device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
	n_train = 6000#1000
	n_val = 1000#500
	batch_size = 50
	n_hidden = args.hidden
	non_linearity = 'ReLU'
	data_root = "/cluster/geliu/bayesian/age_gender_estimation/data"
	project = "imdb"
	dataset = args.dataset#"crx_ref_r1"
	if args.split_type =='gender':
		data = load_data_wiki(data_root, project, dataset, n_train, n_val,top_percent=None, standardize_labels=True,random_state=args.seed,device=device)
	elif args.split_type =='top':
		data = load_data_wiki(data_root, project, dataset, n_train, n_val, top_percent=args.percent, standardize_labels=True,random_state=args.seed, device=device)
	n_inputs = data.train.inputs.shape[1]
	n_models = args.ensem_size
	adversarial_epsilon = None
	#model = NNEnsemble.get_model_resnet(n_inputs, batch_size, n_models, args.depth,args.widen,args.dropout,device)
	#optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
	myhist={'train_nll':[],'train_nll_mix':[],'train_mse':[],'train_loss':[],'train_var':[],'test_nll':[],'test_nll_mix':[],'test_mse':[],'test_var':[],'val_nll':[],'val_nll_mix':[],'val_mse':[],'top_nll':[],'top_nll_mix':[],'top_mse':[],'top_var':[]}
	min_loss= float("inf")
	early_buff=0
	train_newloss=[]
	epoch = 0
	default_mean=data.train.labels.mean().item()
	default_var=data.train.labels.var().item()
	print(default_mean)
	#model.eval()
	with torch.no_grad():
		m=torch.from_numpy(np.array([default_mean]*len(data.train.labels))).float().cuda()
		v=torch.from_numpy(np.array([default_var]*len(data.train.labels))).float().cuda()
		negative_log_likelihood1,negative_log_likelihood2, mse = NNEnsemble.report_metric(data.train.labels,m,v,return_mse=True)
		var=(m.var(dim=0).mean())
		myhist['train_var'].append(var.item())
		myhist['train_nll_mix'].append(negative_log_likelihood1.item())
		myhist['train_nll'].append(negative_log_likelihood2.item())
		myhist['train_mse'].append(mse.item())
		myhist['train_loss'].append(negative_log_likelihood2.item())
		m=torch.from_numpy(np.array([default_mean]*len(data.test.labels))).float().cuda()
		v=torch.from_numpy(np.array([default_var]*len(data.test.labels))).float().cuda()
		negative_log_likelihood1,negative_log_likelihood2, mse = NNEnsemble.report_metric(data.test.labels,m,v,return_mse=True)
		var=(m.var(dim=0).mean())
		myhist['test_var'].append(var.item())
		myhist['test_nll_mix'].append(negative_log_likelihood1.item())
		myhist['test_nll'].append(negative_log_likelihood2.item())
		myhist['test_mse'].append(mse.item())
		m=torch.from_numpy(np.array([default_mean]*len(data.top.labels))).float().cuda()
		v=torch.from_numpy(np.array([default_var]*len(data.top.labels))).float().cuda()
		negative_log_likelihood1,negative_log_likelihood2, mse = NNEnsemble.report_metric(data.top.labels,m,v,return_mse=True)
		var=(m.var(dim=0).mean())
		myhist['top_var'].append(var.item())
		myhist['top_nll_mix'].append(negative_log_likelihood1.item())
		myhist['top_nll'].append(negative_log_likelihood2.item())
		myhist['top_mse'].append(mse.item())
		m=torch.from_numpy(np.array([default_mean]*len(data.val.labels))).float().cuda()
		v=torch.from_numpy(np.array([default_var]*len(data.val.labels))).float().cuda()
		negative_log_likelihood1,negative_log_likelihood2, mse = NNEnsemble.report_metric(data.val.labels,m,v,return_mse=True)
		myhist['val_nll_mix'].append(negative_log_likelihood1.item())
		myhist['val_nll'].append(negative_log_likelihood2.item())
		myhist['val_mse'].append(mse.item())
		tnll2=myhist['train_nll'][-1]
		tnll=myhist['train_nll_mix'][-1]
		tenll2=myhist['test_nll'][-1]
		tenll=myhist['test_nll_mix'][-1]
		tmse=myhist['train_mse'][-1]
		tloss=-1#myhist['train_loss'][-1]
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
		print(f'[E{epoch}] Train NLL1 = {tnll:f}. Train NLL2 = {tnll2:f}.Train MSE = {tmse:f}. Train LOSS = {tloss:f}. Train VAR = {tvar:f}. Val NLL1={vnll:f}.Val NLL2={vnll2:f}. Val MSE={vmse:f}. Test NLL2 = {tenll:f}.Test NLL2 = {tenll2:f}. Test MSE = {temse:f}. Test VAR = {tevar:f}.')
		pd.DataFrame(myhist).to_csv(model_path.replace(".pth", ".txt"))
pd.DataFrame(myhist).to_csv(model_path.replace(".pth", ".txt"))
