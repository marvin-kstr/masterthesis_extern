#############
# Note, to generate Figure 8 or Figure 10, 
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


########## Nonlinear Function for Figure 8 ################

# def nonlin_f(x):
# 	ret = 0
# 	if x <1 and x>(-1):
# 		ret = exp(1/(x**2-1))
# 	else:
# 		ret = 0
# 	return ret

########## Nonlinear Function for Figure 10 ################

def nonlin_f(x):
	return -x**3

# spectral representation of V_k
def V_k(V_arr_zw,x,N):
	V_k_sum = 0
	for l in range(1,N): 
		V_k_sum += V_arr_zw[l-1]*phi(l,x)
	return(V_k_sum)


def nonlin_f_N_approx(V_k_y,l,k,J,h_S):
	f_sum = 0
	for j in range(1,J):
		if l%J==0:
			f_sum += 0
		else:
			f_sum += sqrt(2)*h_S*nonlin_f(V_k_y[k-1][j])*sin(pi*l*(j*h_S))
	return f_sum

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


########################## Definitions of realized Quadratic Variations #################
def temp_quadratic_variation(M,J,output): 
	quad_var_zw = 0
	for k in range(M):
		for j in range(1,J-1):
			quad_var_zw += (output[k+1][j]-output[k][j])**2
	quad_var = (1/((J-2)*M*sqrt(1/M)))*quad_var_zw
	print(quad_var)
	return quad_var


def EEM_create_tempQuadVar_MC_samples(M,N,J,MC):
	T = 1
	S = 1 
	temp_quad_var_lst = []
	for i in tqdm(range(MC)):
		output = nonlinear_exp_euler(M,N,J,T,S)
		norm_temp_quad_var = sqrt(M*(J-2))*(temp_quadratic_variation(M,J,output)-(1/sqrt(1*pi)))
		temp_quad_var_lst.append(norm_temp_quad_var)
	return temp_quad_var_lst


################## Parameter ####################
# Note that in order to consider b>0 we ignore j=0 and j=12 
# and thus only consider [y_1,y_{J-1}]\subset[0,1]. 
# Thus even though we need to set J=12 
#effectively we only consider a partition with J=10
################################################
## Temporal Parameters
M_temp = 1000
N_temp1 = 500
J_temp = 12
MC_temp = 50

M_temp = 1000
N_temp2 = 1000
J_temp = 12
MC_temp = 50

###################### Plots: ####################
fig = plt.figure(figsize=(10, 5), dpi=80)

########### Comparison Gaussian distribution #########
mu = 0
B = 2 + 0.357487253488236
variance_temp = B/(pi*1)
sigma_temp = sqrt(variance_temp)
gaus_temp = np.linspace(mu - 3*sigma_temp, mu + 3*sigma_temp, 100)

EEM_h_temp1 = fig.add_subplot(121)
EEM_h_temp1.hist(EEM_create_tempQuadVar_MC_samples(M_temp,N_temp1,J_temp,MC_temp), bins=5, density=True)
EEM_h_temp1.plot(gaus_temp, stats.norm.pdf(gaus_temp, mu, sigma_temp))
EEM_h_temp1.set_title('N=500')
EEM_h_temp2 = fig.add_subplot(122)
EEM_h_temp2.hist(EEM_create_tempQuadVar_MC_samples(M_temp,N_temp2,J_temp,MC_temp), bins=5, density=True)
EEM_h_temp2.plot(gaus_temp, stats.norm.pdf(gaus_temp, mu, sigma_temp))
EEM_h_temp2.set_title('N=1000')

plt.show()
