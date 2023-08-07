import os, sys, time, argparse
from pickle import TRUE
import shutil
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt


import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from utils import * 
from Learning.Modules.AdvDiffPDE import PIANO_FlowV



##########################################################################
######################       Utility Function      #######################
##########################################################################

def avg_grad(params):
	''''
	params: Torch learnable parameters with gradients
	'''
	grad = 0.
	for p_name, p in params: 
		grad += p.grad.sum()
	return grad

def save_params(named_params, fld, origin, spacing, direction, device):
	for p_name, p in named_params: 
		p = p.detach()
		if device != 'cpu':
			p = p.cpu()
		p = p.numpy()
		file_name = os.path.join(fld, "%s.nii" % p_name)
		#nda2img(p, origin, spacing, direction, isVector = False, save_path = file_name) 
		nda2img(p, isVector = False, save_path = file_name) 

class SmoothnessLoss(nn.Module):
	def __init__(self, weight, data_spacing):
		super(SmoothnessLoss, self).__init__()
		self.weight = weight
		self.data_spacing = data_spacing

	def forward(self, scalar):
		g = gradient_f(scalar, batched = True)
		#g = gradient_f(scalar, batched = True, delta_lst = self.data_spacing) # TODO: correct version
		return (g[..., 0] ** 2 + g[..., 1] ** 2).mean()


##########################################################################


main_fld = '/playpen-raid2/peirong/Data/Allen/2d-fit'

data_fld = '/playpen-raid2/peirong/Data/Allen/2d-sim'
init_P_path = os.path.join(data_fld, 'P0.nii')
P_path = os.path.join(data_fld, 'P.nii')
D_path = os.path.join(data_fld, 'D.nii')

CTC_nda, origin, spacing, direction = img2nda(os.path.join(data_fld, 'C.nii')) # (nT, r, c)
Vx_nda, _, _, _ = img2nda(os.path.join(data_fld, 'Vx.nii'))
Vy_nda, _, _, _ = img2nda(os.path.join(data_fld, 'Vy.nii'))
P0_nda, _, _, _ = img2nda(os.path.join(data_fld, 'P0.nii'))
P_nda, _, _, _ = img2nda(os.path.join(data_fld, 'P.nii'))
D_nda, _, _, _ = img2nda(os.path.join(data_fld, 'D.nii'))

data_dim = [CTC_nda.shape[1], CTC_nda.shape[0]]
data_spacing = [spacing[1], spacing[0]]

n_batch = 1

dt = 0.05 # 0.01, 0.05

#x0, y0 = [-6., -4.] # TODO
data_dim = [64, 64] #, 64]
data_spacing = [0.2, 0.2] #, 1.]

#dt = 0.05
#data_spacing = [1., 1.] #, 1.]

##########################################################################
parser = argparse.ArgumentParser('2D Flow - Fitting, V non-curl-free by construction (Optimization)')

parser.add_argument('--V_time', type = bool, default = True)

####################
# NOTE: UNDER TEST #
####################
parser.add_argument('--fix_D', type = bool, default = False)
parser.add_argument('--param_loss', type = bool, default = False) 
parser.add_argument('--smoothness', type = float, default = 1.)
parser.add_argument('--curl_loss', type = bool, default = False) 
parser.add_argument('--train_nT', type = int, default = 2) 
parser.add_argument('--train_t_increase_freq', type = int, default = 1500) 
parser.add_argument('--test_nT', type = int, default = 10)
####################

parser.add_argument('--adjoint', type = bool, default = True) 
parser.add_argument('--stochastic', type = bool, default = False)
parser.add_argument('--dt', type = float, default = dt, help = 'time interval unit') # 0.01 , 0.02
parser.add_argument('--BC', type = str, default = 'neumann', choices = ['None', 'neumann', 'dirichlet', 'cauchy', 'source', 'source_neumann', 'dirichlet_neumann'])
parser.add_argument('--perf_pattern', type = str, default = 'adv_diff', choices = ['adv_diff', 'adv_only', 'diff_only'])
parser.add_argument('--PD_D_type', type = str, default = 'constant', choices = ['constant', 'scalar'])
parser.add_argument('--PD_V_type', type = str, default = 'vector', choices = ['vector', 'vector_div_free_stream'])
parser.add_argument('--C_magnitude', type = float, default = 10)
parser.add_argument('--V_magnitude', type = float, default = 1.)
parser.add_argument('--D_magnitude', type = float, default = 0.01)
parser.add_argument('--gpu', type = str, required = True, help = 'Select which gpu to use') 

parser.add_argument('--lr', type = float, default = 1e-2)
parser.add_argument('--n_itr', type = int, default = 500000)
parser.add_argument('--print_freq', type = int, default = 1) # 1
parser.add_argument('--save_freq', type = int, default = 10) # 10
args = parser.parse_args()

device = torch.device('cuda:%s' % str(args.gpu))

if args.adjoint:
	from ODE.adjoint import odeint_adjoint as odeint
	print('Using adjoint method...')
else:
	print('Not using adjoint method...')
	from ODE.odeint import odeint

##########################################################################

prefix = 'fixD-' if args.fix_D else ''
prefix += '[Smooth - %.1f]-' % args.smoothness if args.smoothness > 0. else ''
prefix += '[t = %s]-[Param_superv]-' % args.train_nT if args.param_loss else '[t = %s]-' % args.train_nT
if args.V_time:
	print('[V(t) - %s]-[D - %s]' % (args.PD_V_type, args.PD_D_type))
	main_fld = make_dir(os.path.join(main_fld, prefix + '[V(t) - %s]-[D - %s]-' % (args.PD_V_type, args.PD_D_type) + str(os.getpid())))
else:
	print('[V - %s]-[D - %s]' % (args.PD_V_type, args.PD_D_type))
	main_fld = make_dir(os.path.join(main_fld, prefix + '[V - %s]-[D - %s]-' % (args.PD_V_type, args.PD_D_type) + str(os.getpid())))
print('main_fld:', main_fld)

##########################################################################
 
if __name__ == '__main__':
   
	print('PID - %s' % os.getpid())
	
	orig_CTC = Variable(torch.from_numpy(CTC_nda), requires_grad = True).float().to(device) # NOTE (nT, r, c) 
	orig_t_all = Variable(torch.from_numpy(np.arange(orig_CTC.size(0)) * args.dt), requires_grad = True).float().to(device) # (nT)

	test_t_all = orig_t_all[: args.test_nT]

	CTC = orig_CTC[: args.train_nT]

	nda2img(CTC_nda, save_path = os.path.join(make_dir(os.path.join(main_fld, '0')), 'C_GT.nii')) 
	shutil.copyfile(os.path.join(data_fld, 'D.nii'),   os.path.join(main_fld, '0', 'D_GT.nii'))
	shutil.copyfile(os.path.join(data_fld, 'P0.nii'),  os.path.join(main_fld, '0', 'P0_GT.nii'))
	shutil.copyfile(os.path.join(data_fld, 'P.nii'),   os.path.join(main_fld, '0', 'P_GT.nii'))
	shutil.copyfile(os.path.join(data_fld, 'Vx.nii'),  os.path.join(main_fld, '0', 'Vx_GT.nii'))
	shutil.copyfile(os.path.join(data_fld, 'Vy.nii'),  os.path.join(main_fld, '0', 'Vy_GT.nii'))

	nT = len(CTC)
	t_all = Variable(torch.from_numpy(np.arange(nT) * args.dt), requires_grad = True).float().to(device) # (nT)
	C0 = CTC[0].unsqueeze(0) # (n_batch = 1, r, c) 

	##########################################
	############# Initialization #############
	##########################################  

	if args.PD_D_type == 'constant':
		#D = Variable(((torch.ones(data_dim[0], data_dim[1]) + 1.) * 0.5).mean()).float().to(device) * args.D_magnitude # D >= 0 
		D = Variable(torch.from_numpy(D_nda).mean()).float().to(device) 
	elif args.PD_D_type == 'scalar': 
		#D = Variable((torch.ones(data_dim[0], data_dim[1]) + 1.) * 0.5).float().to(device) * args.D_magnitude # D >= 0
		D = Variable(torch.from_numpy(D_nda).float()).to(device) #* args.D_magnitude

	init_P = Variable(torch.from_numpy(P0_nda).float()).to(device) #* args.V_magnitude
	P = Variable(torch.from_numpy(P_nda).float()).to(device) #* args.V_magnitude
	Vx = Variable(torch.from_numpy(Vx_nda).float()).to(device) #* args.V_magnitude
	Vy = Variable(torch.from_numpy(Vy_nda).float()).to(device) #* args.V_magnitude

	## Initializa with assigned values ##
	#OptFunc = PIANO_FlowV(args, data_dim, data_spacing, args.perf_pattern, device, D_param_lst = {'D': D}, V_param_lst = {'Vx': Vx[0], 'Vy': Vy[0]})
	
	## Initialize with random values ## 
	OptFunc = PIANO_FlowV(args, data_dim, data_spacing, args.perf_pattern, device, D_param_lst = None, V_param_lst = None) 


	##########################################
	############# Main  Function #############
	##########################################
 
	OptFunc.to(device) 

	print('Number of parameters:', len(list(OptFunc.parameters())))
	optimizer = optim.Adam(OptFunc.parameters(), lr = args.lr) 
	 
	CTC_criterion = nn.MSELoss()
	CTC_criterion.to(device) 
	
	if args.param_loss:
		param_criterion = nn.L1Loss()
		param_criterion.to(device)
	
	if args.smoothness > 0.:
		smooth_criterion = SmoothnessLoss(args.smoothness, data_spacing)
		smooth_criterion.to(device)

	ctc_loss_lst = []
	curl_loss_lst = []
	param_loss_lst = []
	smooth_loss_lst = []
	fig = plt.figure()
	start_time = time.time()

	curr_nT = 2
	for itr in range(1, args.n_itr + 1):
		
		optimizer.zero_grad()  
  
		if curr_nT < args.train_nT and args.train_nT > 2 and itr % args.train_t_increase_freq == 0:
			try:
				OptFunc.get_V_series(t_all[ : curr_nT+1])
				curr_nT += 1
				print('----- Update train_nT:', curr_nT)
			except:
				print('----- Skip updating train_nT')

		#TODO: get V per one time step: for D as general tensor 
		PD_D = torch.ones(data_dim).to(OptFunc.D.device) * OptFunc.D if args.PD_D_type == "constant" else OptFunc.D
		PD_Vx, PD_Vy = OptFunc.get_V_series(t_all[ : curr_nT]) # get V at all time points # (nT, r, c)
		PD_CTC = torch.stack([torch.zeros_like(C0)] * nT, dim = 0) # (nT, n_batch, r, c)
		PD_CTC[0] = C0
		for nt in range(1, curr_nT):
			OptFunc.it = nt-1 # update current time point #
			PD_CTC[nt] = odeint(OptFunc, PD_CTC[nt-1], t_all[:2], method = 'dopri5', options = args)[-1] 

		CTC_Loss = CTC_criterion(PD_CTC[:, 0], CTC) # (nT, r, c) 
  
		Total_Loss = CTC_Loss
  
		if args.param_loss:
			#V_Loss = param_criterion(PD_Vx, Vx[:nT]) + param_criterion(PD_Vy, Vy[:nT]) # V loss # (nT, r, c)
			V_Loss = param_criterion(PD_Vx[0], Vx[0]) + param_criterion(PD_Vy[0], Vy[0]) # V loss # nT = 0: (r, c)
			#V_Loss = param_criterion(OptFunc.Vx, Vx[0]) + param_criterion(OptFunc.Vy, Vy[0]) # V loss # nT = 0: (r, c)
			D_Loss = param_criterion(PD_D, D)
			Param_Loss = V_Loss + D_Loss
   
			Total_Loss += Param_Loss

		if args.smoothness > 0.:
			Smoothness_Loss = smooth_criterion(PD_Vx[0].unsqueeze(0)) + smooth_criterion(PD_Vy[0].unsqueeze(0))
			if not args.fix_D and args.PD_D_type != 'constant':
				Smoothness_Loss += smooth_criterion(PD_D.unsqueeze(0))
   
			Total_Loss += Smoothness_Loss
   
		if args.curl_loss:
			curl = curl_2D(PD_Vx, PD_Vy, batched = True, delta_lst = data_spacing) # (nT, r, c): nT as batch axis
			Curl_Loss = (curl ** 2).mean()
   
			Total_Loss += Curl_Loss
  
		Total_Loss.backward()
		optimizer.step()

		if itr % args.print_freq == 0:
			print('{:5} |  CTC: {:.5f}'.format(itr, CTC_Loss.item()))
			if args.param_loss:
				print('      |    D: {:.5f}'.format(D_Loss.item())) 
				print('      |    V: {:.5f}'.format(V_Loss.item())) 
			if args.curl_loss:
				print('      | Curl: {:.5f}'.format(Curl_Loss.item())) 
			if args.smoothness > 0.:
				print('      | Smth: {:.5f}'.format(Smoothness_Loss.item())) 
			print('      | Grad: {:.5f}'.format(avg_grad(OptFunc.named_parameters())))

			# Plot updated loss #
			t = [i for i in range(1, int(itr/args.print_freq) + 1)]
			ctc_loss_lst.append(CTC_Loss.item())
			plt.plot(t, ctc_loss_lst, 'g--', label = 'loss')
			x_label = 'Iter (*%d)' % args.print_freq
			plt.xlabel(x_label, fontsize = 16) 
			plt.ylabel('CTC MSE Loss', fontsize = 16)
			plt.legend()
			plt.savefig(os.path.join(main_fld, 'CTC-Loss(t).png'))
			plt.clf()

			if args.param_loss:
				param_loss_lst.append(Param_Loss.item())
				plt.plot(t, param_loss_lst, 'g--', label = 'loss')
				plt.xlabel(x_label, fontsize = 16) 
				plt.ylabel('Param L1 Loss', fontsize = 16)
				plt.legend()
				plt.savefig(os.path.join(main_fld, 'Param-Loss(t).png'))
				plt.clf()

			if args.curl_loss:
				curl_loss_lst.append(Curl_Loss.item())
				plt.plot(t, curl_loss_lst, 'g--', label = 'loss')
				plt.xlabel(x_label, fontsize = 16) 
				plt.ylabel('Curl MSE Loss', fontsize = 16)
				plt.legend()
				plt.savefig(os.path.join(main_fld, 'Curl-Loss(t).png'))
				plt.clf()

			if args.smoothness > 0.:
				smooth_loss_lst.append(Smoothness_Loss.item())
				plt.plot(t, smooth_loss_lst, 'g--', label = 'loss')
				plt.xlabel(x_label, fontsize = 16) 
				plt.ylabel('Smooth MSE Loss', fontsize = 16)
				plt.legend()
				plt.savefig(os.path.join(main_fld, 'Smooth-Loss(t).png'))
				plt.clf()


		if itr % args.save_freq == 0:
			save_fld = make_dir(os.path.join(main_fld, str(itr)))

			PD_V0 = OptFunc.get_V()
			PD_Vx0, PD_Vy0 = PD_V0[0], PD_V0[1]
   
			nda2img(PD_Vx0.detach().cpu().numpy(), save_path = os.path.join(save_fld, 'Vx0.nii')) 
			nda2img(PD_Vy0.detach().cpu().numpy(), save_path = os.path.join(save_fld, 'Vy0.nii')) 
			if args.PD_D_type == 'constant': 
				D = OptFunc.D.detach().cpu().numpy() * np.ones(data_dim)
				nda2img(D, save_path = os.path.join(save_fld, 'D.nii'))
			elif args.PD_D_type == 'scalar':
				nda2img(OptFunc.D.detach().cpu().numpy(), save_path = os.path.join(save_fld, 'D.nii')) 

			try:
				Vx_series, Vy_series = OptFunc.get_V_series(test_t_all) # get P at all time points  # (nT, r, c)
				nda2img(Vx_series.detach().cpu().numpy(), save_path = os.path.join(save_fld, 'Vx.nii')) 
				nda2img(Vy_series.detach().cpu().numpy(), save_path = os.path.join(save_fld, 'Vy.nii')) 
			except:
				print('        [Error occured during V simulation!]')

			try:
				PD_CTC = torch.stack([C0] * len(test_t_all), dim = 0) # (nT, n_batch, r, c)
				for nt in range(1, len(test_t_all)):
					OptFunc.it = nt-1 # update current time point #
					PD_CTC[nt] = odeint(OptFunc, PD_CTC[nt-1], test_t_all[:2], method = 'dopri5', options = args)[-1]
				C_nda = PD_CTC[:, 0] # (nT, n_batch=1, r, c) -> (nT, r, c)
				nda2img(C_nda.detach().cpu().numpy(), save_path = os.path.join(save_fld, 'C.nii'))
			except:
				print('        [Error occured during C simulation!]')

