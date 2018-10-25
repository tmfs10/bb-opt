import pickle
import torch
from torch.nn import Sequential, Linear
from torch.utils.data import TensorDataset, DataLoader
from itertools import cycle
import os.path
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
from bb_opt.src.utils import get_path, load_checkpoint, save_checkpoint, jointplot, load_data_saber
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
    parser.add_argument("-e", "--ensize",dest="ensem_size",type=int,default=20,help="size of ensemble")
    parser.add_argument("-l", "--learnrate",dest="lr",type=float,default=0.001,help="learning rate")
    parser.add_argument("-hd", "--hidden",dest="hidden",type=int,default=100,help="num hidden neurons")
    parser.add_argument("-l2", "--l2",dest="l2",type=float,default=1e-4,help="l2 weight")
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
	model_path=os.path.join(args.model_dir,'{}_{}_{}_{}_{}_{}.pth'.format(args.loss_type,args.sample_size,args.hyper,args.lr,args.ensem_size,args.hidden))
	print(model_path)
	device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
	n_train = 1000
	n_val = 100
	batch_size = 128
	n_hidden = args.hidden
	non_linearity = 'ReLU'
	top_k_percent = 1
	data_root = "/cluster/nhunt/github/bb_opt/data"
	project = "dna_binding"
	dataset = "crx_ref_r1"
	data = load_data_saber(data_root, project, dataset, n_train, n_val, standardize_labels=True, device=device)
	n_inputs = data.train.inputs.shape[1]
	train_loader = DataLoader(TensorDataset(data.train.inputs, data.train.labels),batch_size=batch_size,shuffle=True)
	n_models = args.ensem_size
	mins = data.train.inputs.min(dim=0)[0]
	maxes = data.train.inputs.max(dim=0)[0]
	adversarial_epsilon = None
	model = NNEnsemble.get_model(n_inputs, batch_size, n_models, n_hidden, adversarial_epsilon, device,extra_random=args.extra_random)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
	myhist={'train_nll':[],'train_nll_mix':[],'train_mse':[],'train_loss':[],'train_var':[],'test_nll':[],'test_nll_mix':[],'test_mse':[],'test_var':[],'val_nll':[],'val_nll_mix':[],'val_mse':[]}
	min_loss= float("inf")
	train_newloss=[]
	epoch = 0
	default_mean=data.train.labels.mean().item()
	for epoch in range(epoch, epoch + 15_000):
		model.train()
		for batch in train_loader:
			inputs, labels = batch
			out_size=int(inputs.shape[0]*args.sample_size)
			optimizer.zero_grad()
			means, variances = model(inputs)
			negative_log_likelihood, mse = NNEnsemble.compute_negative_log_likelihood(labels, means, variances, return_mse=True)
			out_data = sample_uniform(out_size)
			means_o, variances_o = model(out_data)
			if args.loss_type=="maxvar":
				var=(means_o.var(dim=0).mean())
				loss=negative_log_likelihood-args.hyper*var
				train_newloss.append(var.item())
			elif args.loss_type=="defmean":
				nll=NNEnsemble.compute_negative_log_likelihood(default_mean,means_o,variances_o)
				loss=negative_log_likelihood+args.hyper*nll
				train_newloss.append(nll.item())
			elif args.loss_type=="normal":
				loss=negative_log_likelihood
			loss.backward()
			optimizer.step()
		model.eval()
		with torch.no_grad():
			means, variances = model(data.train.inputs)
			out_size=int(data.train.inputs.shape[0]*args.sample_size)
			negative_log_likelihood1,negative_log_likelihood2, mse = NNEnsemble.report_metric(data.train.labels, means,variances, return_mse=True)
			out_data = sample_uniform(out_size)
			means_o, variances_o = model(out_data)
			if args.loss_type=="maxvar":
				var=(means_o.var(dim=0).mean())
				loss=negative_log_likelihood2-args.hyper*var
			elif args.loss_type=="defmean":
				nll1,nll2=NNEnsemble.report_metric(default_mean,means_o,variances_o)
				loss=negative_log_likelihood2+args.hyper*nll1
			elif args.loss_type=="normal":
                                loss=negative_log_likelihood
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
			means, variances = model(data.val.inputs)
			negative_log_likelihood1,negative_log_likelihood2, mse = NNEnsemble.report_metric(data.val.labels, means, variances, return_mse=True)
			myhist['val_nll_mix'].append(negative_log_likelihood1.item())
			myhist['val_nll'].append(negative_log_likelihood2.item())
			myhist['val_mse'].append(mse.item())
			if negative_log_likelihood2<min_loss:
				min_loss=negative_log_likelihood2.item()
				model.save_model(model_path, optimizer)
				tnll=myhist['train_nll'][-1]
				vnll=myhist['val_nll'][-1]
				tenll=myhist['test_nll'][-1]
				print('new minimum found, saving model to file system')
				print(f'[E{epoch}] Train NLL = {tnll:,.0f}. Valid NLL = {vnll:,.0f}. Test NLL = {tenll:,.0f}')
		if epoch % 30 == 0:
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
			print(f'[E{epoch}] Train NLL1 = {tnll:,.0f}. Train NLL2 = {tnll2:,.0f}.Train MSE = {tmse:,.0f}. Train LOSS = {tloss:,.0f}. Train VAR = {tvar:,.0f}. Val NLL1={vnll:,.0f}.Val NLL2={vnll2:,.0f}. Val MSE={vmse:,.0f}. Test NLL2 = {tenll:,.0f}.Test NLL2 = {tenll2:,.0f}. Test MSE = {temse:,.0f}. Test VAR = {tevar:,.0f}.')
			pd.DataFrame(myhist).to_csv(model_path.replace(".pth", ".txt"))
