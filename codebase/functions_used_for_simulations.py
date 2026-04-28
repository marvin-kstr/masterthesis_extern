import sys
from os import path
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
from math import sqrt, pi, sin, exp
import scipy.stats as stats
from numpy.random import default_rng
from tqdm import tqdm
rng = default_rng()

##### General Functions ##################
##########################################
# eigenvalue of A
def lambda_k(j):
	return 0.1*pi**2*j**2

# eigenfunction of A
def phi(k,x):
	if x == 1 or x==0: 
		return 0
	else:
		return sqrt(2)*sin(pi*k*x)

# nonlinear function F1:
def nonlin_f0(x):
	return 0

# nonlinear function F2:
def nonlin_f1(x):
		ret = 0
		if x <1 and x>(-1):
			ret = exp(1/(x**2-1))
		else:
			ret = 0
		return ret

# nonlinear function F3:
def nonlin_f2(x):
	return -x**3

# Approximation of Nonlinear function
def nonlin_f_N_approx(V_k_y,l,k,J,h_S):
	f_sum = 0
	for j in range(1,J):
		if l%J==0:
			f_sum += 0
		else:
			f_sum += sqrt(2)*h_S*nonlin_f(V_k_y[k-1][j])*sin(pi*l*(j*h_S))
	return f_sum


##### For exponential Euler method: ######
########################################## 
# spectral representation of V_k
def V_k(V_arr_zw,x,N):
	V_k_sum = 0
	for l in range(1,N): 
		V_k_sum += V_arr_zw[l-1]*phi(l,x)
	return(V_k_sum)

## Exponential Euler method. 
# Also returns random values of noise,
# if used as reference solution
def nonlinear_exp_euler(M,N,J,T,S):
	h = T/M 
	h_S = S/J
	U_o = [0]*N
	V_lst = [U_o.copy()]
	V_k_y = [[0]*(J+1)]
	rand_lst = [] 
	for k in tqdm(range(1,M+1)):
		V_lst_zw =[]
		rand_lst_zw =[]
		for l in range(1, N+1):
			f_sum=nonlin_f_N_approx(V_k_y,l,k,J,h_S)
			np.random.seed()
			rand = rng.standard_normal()
			rand_lst_zw.append(rand)
			V = exp(-lambda_k(l)*h)*V_lst[k-1][l-1] + ((1-exp(-lambda_k(l)*h))/lambda_k(l))*f_sum + ((1-exp(-2*lambda_k(l)*h))/(2*lambda_k(l)))**0.5*rand
			V_lst_zw.append(V)
		rand_lst.append(rand_lst_zw.copy())
		V_lst.append(V_lst_zw.copy())
		V_k_y_zw = []
		for j in range(J+1):
			if j==0 or j==J:
				V_k_y_zw.append(0)
			else:
				V_k_y_zw.append(V_k(V_lst[k],j*h_S,N))
		V_k_y.append(V_k_y_zw.copy())

	rand_arr = np.array(rand_lst)
	V_results = np.array(V_k_y)
	return V_results, rand_arr

### Version of exponential Euler method which uses noise from reference solution
def kappa_nonlinear_exp_euler(M_ref,N,J,T,S,kappa,rand_lst_all):
	timer = time.time()
	M = int(M_ref/kappa)
	h_ref = T/M_ref
	h = T/M
	h_S = S/J 
	U_o = [0]*N
	V_lst = [U_o.copy()]
	V_k_y = [[0]*(J+1)]
	for k in tqdm(range(1,M+1)):
		V_lst_zw =[]
		for l in range(1, N+1):
			f_sum=nonlin_f_N_approx(V_k_y,l,k,J,h_S)
			det_rand=0
			for q in range(1,kappa+1):
				det_rand += exp(-lambda_k(l)*(kappa-q)*h_ref)*(1/(2*lambda_k(l))*(1-exp(-2*lambda_k(l)*h_ref)))**0.5*rand_lst_all[kappa*(k-1)+(q-1)][l-1]
			V = exp(-lambda_k(l)*h)*V_lst[k-1][l-1] + ((1-exp(-lambda_k(l)*h))/lambda_k(l))*f_sum + det_rand
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


##### Semilinear replacement method: #####
##########################################

def I_l_L(l,J,L):
	I_l = [l]
	for k in range(1,50):
		I_l.append(2*k*J-l)
		I_l.append(2*k*J+l)
	I_l_L = [i for i in I_l if i<L*J]
	return I_l_L

### Needed functions for s_l^2 
def b_l(l,J):
	b_l_zw = [sqrt(2)* sin(pi*l*(m/J)) for m in range(1,J)]
	b_l_arr = np.array([b_l_zw]).T
	return b_l_arr

### Sigma is a (J-1xJ-1)-Matrix where Sigma_k,l is a symmetric function  
def Sigma_sum(J):
	Sigma_sum = []
	for k in range(1,J):
		Sigma_sum_k = []
		for l in range(1,J):
			if k <= l:
				Sigma_sum_k.append(((k/J)*(1-l/J))/(2*0.1))
			else:
				Sigma_sum_k.append(((l/J)*(1-k/J))/(2*0.1))
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

# spectral representation of V_k
def V_k_repl(V_arr_zw,s_l_square_arr, x, J, L):
	V_k_sum = 0
	for l in range(1,J): 
		U_l_L_sum = 0
		for m in I_l_L(l,J,L):
			U_l_L_sum += V_arr_zw[m-1]
		s_l_sq = s_l_square_arr[l-1]
		np.random.seed()
		rand_l = rng.normal(0,sqrt(s_l_sq))
		V_k_sum += (U_l_L_sum + rand_l)*phi(l,x)
	return(V_k_sum)

# semilinear replacement method:
def semilin_replacement_method(M,L,J,T,S,s_l_square_arr):
	N = L*J
	h = T/M
	h_S = S/J 
	U_o = [0]*N
	V_lst = [U_o.copy()]
	rand_lst = [] 
	V_k_y_repl = [[0]*(J+1)]
	for k in tqdm(range(1,M+1)):
		V_lst_zw =[]
		rand_lst_zw =[]
		for l in range(1, N):
			f_sum=nonlin_f_N_approx0(V_k_y_repl,l,k,J,h_S)
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

##### strong error approximation #########
#(here as version w.r.t. computational costs)
##########################################

M_ref = 96 #temporal discretization
N_ref = M_ref**2 #galerkin discretization 
J = 100 #spacial discretization
T = 1 #length temporal interval
S = 1 #length spacial interval
kappas = [2,3,4,6,8,12,16,24,32,48]
K =50 

results = [0]*len(kappas)
monte_carlo_lst =[]
for k in tqdm(range(K)):
	reference = nonlinear_exp_euler(M_ref,N_ref,J,T,S)
	rand_lst_all = reference[1]
	for i in kappas:
		sum_montecarlo_zw = 0
		kappa=i
		N = int(int(M_ref/kappa)**2)
		approx = kappa_nonlinear_exp_euler(M_ref,N,J,T,S,kappa,rand_lst_all)
		sum_dim = 0
		for l in range(len(reference[0][-1])):
			sum_dim += (S/J)*(np.abs(reference[0][-1][l]-approx[-1][l]))**2
		monte_carlo_lst.append(sum_dim)

for j in range(len(kappas)):
	for i in range(K):
		sum_MC = 0
		sum_MC += monte_carlo_lst[i*len(kappas)+j]
	results[j] += (1/K)*sum_MC

##### realized quadratic variations #########
#(here as version w.r.t. computational costs)
############################################

def temp_quadratic_variation(M,J,output): 
	quad_var_zw = 0
	for k in range(M):
		for j in range(1,J-1):
			quad_var_zw += (output[k+1][j]-output[k][j])**2
	quad_var = (1/((J-2)*M*sqrt(1/M)))*quad_var_zw
	print(quad_var)
	return quad_var

def spat_quadratic_variation(M,J,output): 
	quad_var_zw = 0
	for k in range(1,M+1):
		for j in range(J):
			quad_var_zw += (output[k][j+1]-output[k][j])**2
	quad_var = (1/((J)*M*(1/(J))))*quad_var_zw
	return quad_var

def create_tempQuadVar_MC_samples(M,L,J,MC):
	s_l_square_arr = [s_l_square(l,J,L) for l in range(1,J)]
	# if needed for semilinear repl method
	temp_quad_var_lst = []
	for i in tqdm(range(MC)):
		timer_MC = time.time()
		output = # Output Approximation method
		temp_quad_var = temp_quadratic_variation(M,J,output)
		norm_temp_quad_var = sqrt(M*(J-2))*(temp_quad_var-(1/sqrt(pi*1)))
		temp_quad_var_lst.append(norm_temp_quad_var)
	return temp_quad_var_lst

def create_spatQuadVar_MC_samples(M,L,J,MC):
	s_l_square_arr = [s_l_square(l,J,L) for l in range(1,J)]
	# if needed for semilinear repl method
	spat_quad_var_lst = []
	for i in tqdm(range(MC)):
		output = # Output Approximation method
		norm_spat_quad_var = sqrt(M*(J))*(spat_quadratic_variation(M,J,output)-(1/(2*1)))
		spat_quad_var_lst.append(norm_spat_quad_var)
	return spat_quad_var_lst
