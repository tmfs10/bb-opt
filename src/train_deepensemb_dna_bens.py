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
from bb_opt.src.utils import get_path, load_checkpoint, save_checkpoint, jointplot, load_data_saber, load_data_saber2,load_data_gc
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
    parser.add_argument("-hd", "--hidden",dest="hidden",type=int,default=100,help="num hidden neurons")
    parser.add_argument("-l2", "--l2",dest="l2",type=float,default=1e-4,help="l2 weight")
    parser.add_argument("-d","--data",dest="data_type",type=str,default='rand',help="specify which train/val/test")
    parser.add_argument("-ep", "--epoch",dest="epoch",type=int,default=700,help="num training epoch")
    parser.add_argument("-sg", "--single",dest='single_layer',type=str,default='Y',help="single layer?")
    parser.add_argument("-ds", "--dataset",dest='dataset',type=str,default='ARX_L343Q_R1_8mers.txt',help="which dataset to use")
    parser.add_argument("-p", "--top",dest="percent",type=float,default=0.1,help="percentage of top data heldout")
    parser.add_argument("-sd", "--seed",dest="seed",type=int,default=0,help="random seed")
    parser.add_argument("-ad", "--adverse",dest="adverse_eps",type=float,default=None,help="epsilon for adverserial training")
    parser.add_argument("-v", "--std", dest="std",type=float, default=1.0,help="customized aleatoric std")
    parser.add_argument("-a", "--aleatoric",dest='aleatoric',type=str,default='hetero',help="heteroskedastic?")
    return parser.parse_args()

def sample_uniform(out_size):
    z = np.zeros((8*out_size,4))
    z[range(8*out_size),np.random.randint(4,size=8*out_size)]=1
    out_data = torch.from_numpy(z).view((-1,32)).float().cuda()
    return out_data

if __name__ == "__main__":
	args = parse_args()
	#gpu_id = gpu_init()
	torch.cuda.set_device(args.gpu)
	if args.adverse_eps:
		model_path=os.path.join(args.model_dir,'{}_{}_{}_{}_{}_{}_{}_{}_{}-{}.pth'.format(args.dataset.split('_8mers')[0],args.loss_type,args.sample_size,args.hyper,args.lr,args.l2,args.ensem_size,args.hidden,args.adverse_eps,args.std))
	else:
		model_path=os.path.join(args.model_dir,'{}_{}_{}_{}_{}_{}_{}_{}-{}.pth'.format(args.dataset.split('_8mers')[0],args.loss_type,args.sample_size,args.hyper,args.lr,args.l2,args.ensem_size,args.hidden,args.std))
	print(model_path)
	if exists(model_path.replace(".pth", ".txt")):
		a=pd.read_csv(model_path.replace(".pth", ".txt"))
		if not len(a)%100==0:
			print("file exists!")
			sys.exit(0)
		#print("file exists!")
		#sys.exit(0)
	device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
	n_train = 600#1000
	n_val = 300#500
	batch_size = 128
	n_hidden = args.hidden
	non_linearity = 'ReLU'
	#top_k_percent = 1
	data_root = "/cluster/sj1/bb_opt/data"
	project = "paper_data2"
	dataset = args.dataset#"crx_ref_r1"
	if args.data_type =='rand':
		data = load_data_saber(data_root, project, dataset, n_train, n_val, standardize_labels=True, device=device)
	elif args.data_type =='top':
		data = load_data_saber2(data_root, project, dataset, n_train, n_val, args.percent, standardize_labels=True,random_state=args.seed, device=device)
	elif args.data_type =='gc':
		data = load_data_gc(data_root, project, dataset, n_train, n_val, args.percent, standardize_labels=True,random_state=args.seed, device=device)
	n_inputs = data.train.inputs.shape[1]
	train_loader = DataLoader(TensorDataset(data.train.inputs, data.train.labels),batch_size=batch_size,shuffle=True)
	n_models = args.ensem_size
	adversarial_epsilon = args.adverse_eps
	if args.loss_type=='bayes':
		model = NNEnsemble.get_model(n_inputs, batch_size, n_models, n_hidden, adversarial_epsilon, device=device,extra_random=args.extra_random,single_layer=(args.single_layer=='Y'),mu_prior=0,std_prior=1)
	else:
		model = NNEnsemble.get_model(n_inputs, batch_size, n_models, n_hidden, adversarial_epsilon, device=device,extra_random=args.extra_random,single_layer=(args.single_layer=='Y'))
	model=model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
	myhist={'train_nll':[],'train_nll_mix':[],'train_mse':[],'train_loss':[],'train_var':[],'test_nll':[],'test_nll_mix':[],'test_mse':[],'test_var':[],'val_nll':[],'val_nll_mix':[],'val_mse':[],'top_nll':[],'top_nll_mix':[],'top_mse':[],'top_var':[]}
	min_loss= float("inf")
	early_buff=0
	train_newloss=[]
	epoch = 0
	default_mean=data.test.labels.mean().item()
	for epoch in range(epoch, epoch + args.epoch):
		model.train()
		for batch in train_loader:
			inputs, labels = batch
			#print(labels)
			out_size=int(inputs.shape[0]*args.sample_size)
			optimizer.zero_grad()
			means, variances = model(inputs)
			if args.aleatoric=='homo':
				negative_log_likelihood, mse = NNEnsemble.compute_negative_log_likelihood(labels, means, variances,custom_std=args.std,return_mse=True)
				lossterm=mse
			else:
				negative_log_likelihood, mse = NNEnsemble.compute_negative_log_likelihood(labels, means, variances,return_mse=True)
				lossterm=negative_log_likelihood
			if args.loss_type=="maxvar":
				out_data = sample_uniform(out_size)
				means_o, variances_o = model(out_data)
				var=(means_o.var(dim=0).mean())
				loss=lossterm-args.hyper*var
				train_newloss.append(var.item())
			elif args.loss_type=="defmean":
				out_data = sample_uniform(out_size)
				means_o, variances_o = model(out_data)
				nll=NNEnsemble.compute_negative_log_likelihood(default_mean,means_o,variances_o,custom_std=args.std if args.aleatoric=='homo' else None)
				loss=lossterm+args.hyper*nll
				train_newloss.append(nll.item())
			elif args.loss_type=="normal":
				loss=lossterm#negative_log_likelihood
			elif args.loss_type=="normal+at":
				means_adv, variances_adv = model(inputs,labels)
				negative_log_likelihood_adv, mse_adv = NNEnsemble.compute_negative_log_likelihood(labels, means_adv, variances_adv,custom_std=args.std if args.aleatoric=='homo' else None, return_mse=True)
				if args.aleatoric=='homo':
					lossterm2=mse_adv
				else:
					lossterm2=negative_log_likelihood_adv                                 
				loss=lossterm+lossterm2#negative_log_likelihood+negative_log_likelihood_adv
			elif args.loss_type=="invar":
				var=(means.var(dim=0).mean())
				loss=lossterm-args.hyper*var
				train_newloss.append(var.item())
			elif args.loss_type=="2var":
				out_data = sample_uniform(out_size)
				means_o, variances_o = model(out_data)
				var=(0.5*means_o.var(dim=0).mean())+(0.5*means.var(dim=0).mean())
				loss=lossterm-args.hyper*var
				train_newloss.append(var.item())
			elif args.loss_type=='idvar':
				out_data = sample_uniform(out_size)
				means_o, variances_o = model(out_data)
				var=((means_o.var(dim=0)*knn_density(inputs,out_data,5)).mean())
				loss=lossterm-args.hyper*var
				train_newloss.append(var.item())
			elif args.loss_type=='idvar-max':
				out_data = sample_uniform(out_size)
				means_o, variances_o = model(out_data)
				var=((means_o.var(dim=0)*knn_density_max(inputs,out_data,5)).mean())
				loss=lossterm-args.hyper*var
				train_newloss.append(var.item())
			elif args.loss_type=='nc':
				nc= NNEnsemble.compute_negative_correlation(means)
				loss=lossterm+args.hyper*nc
				train_newloss.append(var.item())
			elif args.loss_type=='bayes':
				#loss=negative_log_likelihood+model.bayesian_ensemble_loss(data_noise=torch.tensor(args.std).to(device))/inputs.shape[0]
				loss=lossterm+model.bayesian_ensemble_loss(data_noise=torch.tensor(args.std).to(device))/inputs.shape[0]
				train_newloss.append(loss.item())
			loss.backward()
			optimizer.step()
		model.eval()
		with torch.no_grad():
			means, variances = model(data.train.inputs)
			#out_size=int(data.train.inputs.shape[0]*args.sample_size)
			negative_log_likelihood1,negative_log_likelihood2, mse = NNEnsemble.report_metric_sb(data.train.labels, means,variances,custom_std=args.std if args.aleatoric=='homo' else None,return_mse=True)
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
			negative_log_likelihood1,negative_log_likelihood2,mse = NNEnsemble.report_metric_sb(data.test.labels, means, variances,custom_std=args.std if args.aleatoric=='homo' else None, return_mse=True)
			myhist['test_nll_mix'].append(negative_log_likelihood1.item())
			myhist['test_nll'].append(negative_log_likelihood2.item())
			myhist['test_mse'].append(mse.item())
			means, variances = model(data.top.inputs)
			var=(means.var(dim=0).mean())
			myhist['top_var'].append(var.item())
			negative_log_likelihood1,negative_log_likelihood2,mse = NNEnsemble.report_metric_sb(data.top.labels, means, variances,custom_std=args.std if args.aleatoric=='homo' else None, return_mse=True)
			myhist['top_nll_mix'].append(negative_log_likelihood1.item())
			myhist['top_nll'].append(negative_log_likelihood2.item())
			myhist['top_mse'].append(mse.item())
			means, variances = model(data.val.inputs)
			negative_log_likelihood1,negative_log_likelihood2, mse = NNEnsemble.report_metric_sb(data.val.labels, means, variances,custom_std=args.std if args.aleatoric=='homo' else None, return_mse=True)
			myhist['val_nll_mix'].append(negative_log_likelihood1.item())
			myhist['val_nll'].append(negative_log_likelihood2.item())
			myhist['val_mse'].append(mse.item())
			if negative_log_likelihood2<=min_loss:
				min_loss=negative_log_likelihood2.item()
				model.save_model(model_path, optimizer)
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
			pd.DataFrame(myhist).to_csv(model_path.replace(".pth", ".txt"))
	pd.DataFrame(myhist).to_csv(model_path.replace(".pth", ".txt"))
