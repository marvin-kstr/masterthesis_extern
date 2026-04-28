#############
# Note, to generate Figure 7 or Figure 9, 
# please choose the needed nonlinearity
# and comment the other one
#############

import sys
from os import path
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from math import sqrt, pi, sin, exp
import scipy.stats as stats
from numpy.random import default_rng
from tqdm import tqdm
rng = default_rng()

########################## Functions needed #######################

# eigenvalue of A
def lambda_k(j):
	return pi**2*j**2

# eigenfunction of A
def phi(k,x):
	if x == 1 or x==0: 
		return 0
	else:
		return sqrt(2)*sin(pi*k*x)


########## Nonlinear Function for Figure 7 ################

# def nonlin_f(x):
# 	ret = 0
# 	if x <1 and x>(-1):
# 		ret = exp(1/(x**2-1))
# 	else:
# 		ret = 0
# 	return ret

########## Nonlinear Function for Figure 9 ################

def nonlin_f(x):
	return -x**3

########################

def nonlin_f_N_approx(V_k_y,l,k,J,h_S):
	f_sum = 0
	for j in range(1,J):
		if l%J==0:
			f_sum += 0
		else:
			f_sum += sqrt(2)*h_S*nonlin_f(V_k_y[k-1][j])*sin(pi*l*(j*h_S))
	return f_sum


########################## Definitions of realized Quadratic Variations #################
def temp_quadratic_variation(M,J,output): 
	quad_var_zw = 0
	for k in range(M):
		for j in range(1,J-1):
			quad_var_zw += (output[k+1][j]-output[k][j])**2
	quad_var = (1/((J-2)*M*sqrt(1/M)))*quad_var_zw
	print(quad_var)
	return quad_var

################# Exp Euler Method ####################

# spectral representation of V_k
def V_k(V_arr_zw,x,N):
	V_k_sum = 0
	for l in range(1,N): 
		V_k_sum += V_arr_zw[l-1]*phi(l,x)
	return(V_k_sum)

def nonlinear_exp_euler(M,N,J,T,S):
	h = T/M 
	h_S = S/J
	U_o = [0]*N
	V_lst = [U_o.copy()]
	rand_lst = [] 
	V_k_y = [[0]*(J+1)]
	for k in range(1,M+1):
		V_lst_zw =[]
		rand_lst_zw =[]
		for l in range(1, N+1):
			f_sum=nonlin_f_N_approx(V_k_y,l,k,J,h_S)
			np.random.seed()
			rand = rng.standard_normal()
			V = exp(-lambda_k(l)*h)*V_lst[k-1][l-1] + ((1-exp(-lambda_k(l)*h))/lambda_k(l))*f_sum + ((1-exp(-2*lambda_k(l)*h))/(2*lambda_k(l)))**0.5*rand
			V_lst_zw.append(V)
		V_lst.append(V_lst_zw.copy())
		V_k_y_zw = []
		for j in range(J+1):
			if j==0 or j==J:
				V_k_y_zw.append(0)
			else:
				V_k_y_zw.append(V_k(V_lst[k],j*h_S,N))
		V_k_y.append(V_k_y_zw.copy())
	V_results = np.array(V_k_y)
	return V_results

def EEM_create_tempQuadVar_MC_samples(M,N,J,MC):
	timer_temporal = time.time()
	T = 1
	S = 1 
	print('temporal:',' M:',M,' N:',N,' J:',J,' MC:', MC)
	temp_quad_var_lst = []
	for i in tqdm(range(MC)):
		timer_MC = time.time()
		output = nonlinear_exp_euler(M,N,J,T,S)
		norm_temp_quad_var = sqrt(M*(J-2))*(temp_quadratic_variation(M,J,output)-(1/sqrt(1*pi)))
		temp_quad_var_lst.append(norm_temp_quad_var)
		ellapsed = time.time()-timer_MC
		print('Variant: temporal', ' MC:', i, ' sec:', ellapsed, ' value:', norm_temp_quad_var)
	needed_time = time.time()-timer_temporal
	print('(Temporal) time needed in sec:', needed_time)
	return temp_quad_var_lst

################# Semilin Repl Method ####################

###### Needed for replacement method ###########
def I_l_L(l,J,L):
	I_l = [l]
	for k in range(1,50):
		I_l.append(2*k*J-l)
		I_l.append(2*k*J+l)
	I_l_L = [i for i in I_l if i<L*J]
	return I_l_L

def b_l(l,J):
	b_l_zw = [sqrt(2)* sin(pi*l*(m/J)) for m in range(1,J)]
	b_l_arr = np.array([b_l_zw]).T
	return b_l_arr

def Sigma_sum(J):
	Sigma_sum = []
	for k in range(1,J):
		Sigma_sum_k = []
		for l in range(1,J):
			if k <= l:
				Sigma_sum_k.append(((k/J)*(1-l/J))/(2*1))
			else:
				Sigma_sum_k.append(((l/J)*(1-k/J))/(2*1))
		Sigma_sum.append(Sigma_sum_k.copy())
	Sigma_arr = np.array(Sigma_sum)
	return Sigma_arr

def s_l_square(l,J,L):
	s_l_zw  = (b_l(l,J).T @ Sigma_sum(J) @ b_l(l,J))[0][0]
	lam_l_sum = 0
	for m in I_l_L(l,J,L):
		lam_l_sum += 1/(2*lambda_k(m))
	s_l = 1/(J**2)*s_l_zw-lam_l_sum
	return s_l

def V_k_repl(V_arr_zw,s_l_square_arr, x, J, L):
	V_k_sum = 0
	for l in range(1,J): 
		for m in I_l_L(l,J,L):
			U_l_L_sum += V_arr_zw[m-1]
		s_l_sq = s_l_square_arr[l-1]
		#print(s_l_sq)
		np.random.seed()
		rand_l = rng.normal(0,sqrt(s_l_sq))
		V_k_sum += (U_l_L_sum + rand_l)*phi(l,x)
	return(V_k_sum)

def semilin_replacement_method(M,L,J,T,S,s_l_square_arr):
	N = L*J
	h = T/M #temporal meshgrid
	h_S = S/J #spatial meshgrid
	U_o = [0]*N
	V_lst = [U_o.copy()]
	rand_lst = [] 
	V_k_y_repl = [[0]*(J+1)]
	for k in range(1,M+1):
		V_lst_zw =[]
		rand_lst_zw =[]
		for l in range(1, N):
			f_sum=nonlin_f_N_approx(V_k_y_repl,l,k,J,h_S)
			np.random.seed()
			rand = rng.standard_normal()
			V = exp(-lambda_k(l)*h)*V_lst[k-1][l-1] + ((1-exp(-lambda_k(l)*h))/lambda_k(l))*f_sum + ((1-exp(-2*lambda_k(l)*h))/(2*lambda_k(l)))**0.5*rand
			V_lst_zw.append(V)
		V_lst.append(V_lst_zw.copy())
		V_k_y_repl_zw = []
		for j in range(J+1):
			if j==0 or j==J:
				V_k_y_repl_zw.append(0)
			else:
				V_k_y_repl_zw.append(V_k_repl(V_lst[k],s_l_square_arr,j*h_S,J,L))
		V_k_y_repl.append(V_k_y_repl_zw.copy())
	V_results = np.array(V_k_y_repl)
	return V_results

def SRM_create_tempQuadVar_MC_samples(M,L,J,MC):
	s_l_square_arr = [s_l_square(l,J,L) for l in range(1,J)]
	temp_quad_var_lst = []
	for i in tqdm(range(MC)):
		timer_MC = time.time()
		output = semilin_replacement_method(M,L,J,1,1,s_l_square_arr)
		temp_quad_var = temp_quadratic_variation(M,J,output)
		norm_temp_quad_var = sqrt(M*(J-2))*(temp_quad_var-(1/sqrt(pi*1)))
		temp_quad_var_lst.append(norm_temp_quad_var)
	return temp_quad_var_lst

################## Parameter ####################
# Note that in order to consider b>0 we ignore j=0 and j=12 
# and thus only consider [y_1,y_{J-1}]\subset[0,1]. 
# Thus even though we need to set J=12 
#effectively we only consider a partition with J=10
################################################
# Exp-Euler-Method:
M_trunc = 1000
N_trunc = 100
J_trunc = 12


# Semilin repl method:
M_repl = 1000
J_repl = 12
L_repl = 10

MC = 100

########### Comparison Gaussian distribution #########
mu = 0
B = 2 + 0.357487253488236
variance_temp = B/(pi*1)
sigma_temp = sqrt(variance_temp)
gaus_temp = np.linspace(mu - 3*sigma_temp, mu + 3*sigma_temp, 100)

###################### Plots: ####################
fig = plt.figure(figsize=(10,5), dpi=80)

# Exp euler:
EEM_h_trunc = fig.add_subplot(121)
EEM_h_trunc.hist(EEM_create_tempQuadVar_MC_samples(M_trunc,N_trunc,J_trunc,MC), bins=9, density=True)
EEM_h_trunc.plot(gaus_temp, stats.norm.pdf(gaus_temp, mu, sigma_temp))
EEM_h_trunc.set_title('N=100')

# Semilin repl method:
SRM_h_repl = fig.add_subplot(122)
SRM_h_repl.hist(SRM_create_tempQuadVar_MC_samples(M_repl,L_repl,J_repl,MC), bins=9, density=True)
SRM_h_repl.plot(gaus_temp, stats.norm.pdf(gaus_temp, mu, sigma_temp))
SRM_h_repl.set_title('L=10')

plt.show()

