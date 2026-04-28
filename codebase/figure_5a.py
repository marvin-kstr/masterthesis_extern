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

##################### needed functions ################################
############################################################################################

# eigenvalue of A
def lambda_k(j):
	return 0.1*pi**2*j**2

# eigenfunction of A
def phi(k,x):
	if x == 1 or x==0: 
		return 0
	else:
		return sqrt(2)*sin(pi*k*x)

def nonlin_f(x):
	ret = 0
	if x <1 and x>(-1):
		ret = exp(1/(x**2-1))
	else:
		ret = 0
	return 10*ret

# spectral representation of V_k
def V_k(V_arr_zw, x, N):
	V_k_sum = 0
	for j in range(1,N): 
		V_k_sum += V_arr_zw[j-1]*phi(j,x)
	return(V_k_sum)

def nonlin_f_N_approx(V_k_y,l,k,J,h_S):
	f_sum = 0
	for j in range(1,J):
		if l%J==0:
			f_sum += 0
		else:
			f_sum += sqrt(2)*h_S*nonlin_f(V_k_y[k-1][j])*sin(pi*l*(j*h_S))
	return f_sum


############## exponential Euler Method ##############
############################################################################################
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
		V_lst.append(V_lst_zw.copy())
		V_k_y_zw = []
		for j in range(J+1):
			if j==0 or j==J:
				V_k_y_zw.append(0)
			else:
				V_k_y_zw.append(V_k(V_lst[k],j*h_S,N))
		V_k_y.append(V_k_y_zw.copy())
		rand_lst.append(rand_lst_zw.copy())
	V_results = np.array(V_k_y)
	rand_arr = np.array(rand_lst)
	return V_results, rand_arr



########################### Define approximation exponential Euler Method ##################
############################################################################################

def Nref_nonlinear_exp_euler(M,N,J,T,S,rand_lst):
	h = T/M
	h_S = S/J
	U_o = [0]*N
	V_lst = [U_o.copy()]
	V_k_y = [[0]*(J+1)]
	for k in tqdm(range(1,M+1)):
		V_lst_zw =[]
		for l in range(1, N+1):
			f_sum=nonlin_f_N_approx(V_k_y,l,k,J,h_S)
			np.random.seed()
			rand = rand_lst[k-1][l-1] #use noise from reference Solution
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

########################### Generate Simulations ###########################################
############################################################################################

M_ref = 100 #96temporal discretization
N_ref = 2000 #galerkin discretization  ## Note that this is recommended by Kloeden and Jentzen!
J = 100 #spacial discretization
T = 1 #length temporal interval
S = 1 #length spacial interval
kappas = [10,25,35,50,75,100,250,350,500,750,1000]
K = 50

results = [0]*len(kappas)
monte_carlo_lst =[]
for k in tqdm(range(K)):
	reference = nonlinear_exp_euler(M_ref,N_ref,J,T,S)
	rand_lst_all = reference[1]
	for i in kappas:
		sum_montecarlo_zw = 0
		kappa=i
		N = i
		approx = Nref_nonlinear_exp_euler(M_ref,N,J,T,S,rand_lst_all)
		sum_dim = 0
		for l in range(len(reference[0][-1])):
			sum_dim += (S/J)*(np.abs(reference[0][-1][l]-approx[-1][l]))**2
		monte_carlo_lst.append(sum_dim)

for j in range(len(kappas)):
	for i in range(K):
		sum_MC = 0
		sum_MC += monte_carlo_lst[i*len(kappas)+j]
	results[j] += (1/K)*sum_MC


########################### Plot results ###################################################
############################################################################################

result_x = [i for i in kappas]
result_y = [results[i] for i in range(len(results))]

line_x = np.linspace(10,kappas[-1],1000)
def refline(x,result_x,result_y):
	return (x**(-1/2)/300)
fig = plt.figure(figsize=(12, 12), dpi=80)
ax = fig.add_subplot(111)
ax.plot(result_x,result_y,color='blue')
ax.plot(result_x,result_y,'o',color='blue')
ax.plot(line_x, refline(line_x,result_x,result_y),color='orange')
ax.set_xlabel('N')
ax.set_ylabel('Strong Error')
ax.set_xscale('log')
ax.set_yscale('log')

plt.show()

