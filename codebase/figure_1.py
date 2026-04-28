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


######### Functions #############

# eigenvalue of A
def lambda_k(j):
	return 0.1*pi**2*j**2

# eigenfunction of A
def phi(k,x):
	if x == 1 or x==0: 
		return 0
	else:
		return sqrt(2)*sin(pi*k*x)

# nonlinear funtion F1:
def nonlin_f0(x):
	return 0

# nonlinear function F2:
def nonlin_f1(x):
		ret = 0
		if x <1 and x>(-1):
			ret = exp(1/(x**2-1))
		else:
			ret = 0
		return 10*ret

# nonlinear function F3:
def nonlin_f2(x):
	return -10*x**3


# spectral representation of V_k
def V_k(V_arr_zw,x,N):
	V_k_sum = 0
	for l in range(1,N): 
		V_k_sum += V_arr_zw[l-1]*phi(l,x)
	return(V_k_sum)

def nonlin_f_N_approx0(V_k_y,l,k,J,h_S):
	f_sum = 0
	for j in range(1,J):
		if l%J==0:
			f_sum += 0
		else:
			f_sum += sqrt(2)*h_S*nonlin_f0(V_k_y[k-1][j])*sin(pi*l*(j*h_S))
	return f_sum

def nonlin_f_N_approx1(V_k_y,l,k,J,h_S):
	f_sum = 0
	for j in range(1,J):
		if l%J==0:
			f_sum += 0
		else:
			f_sum += sqrt(2)*h_S*nonlin_f1(V_k_y[k-1][j])*sin(pi*l*(j*h_S))
	return f_sum

def nonlin_f_N_approx2(V_k_y,l,k,J,h_S):
	f_sum = 0
	for j in range(1,J):
		if l%J==0:
			f_sum += 0
		else:
			f_sum += sqrt(2)*h_S*nonlin_f2(V_k_y[k-1][j])*sin(pi*l*(j*h_S))
	return f_sum


####### exponential euler methods #########

def nonlinear_exp_euler0(M,N,J,T,S):
	h = T/M
	h_S = S/J
	U_o = [0]*N
	V_lst = [U_o.copy()]
	V_k_y = [[0]*(J+1)]

	for k in tqdm(range(1,M+1)):
		V_lst_zw =[]
		for l in range(1, N+1):
			f_sum=nonlin_f_N_approx0(V_k_y,l,k,J,h_S)
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
	#structure: [[V_{0,1},...,V_{0,N}],[V_{1,1},...,V_{1,N}],...,[V_{M,1},...,V_{M,N}]])
	V_results = np.array(V_k_y)
	return V_results
	#return V_results


def nonlinear_exp_euler1(M,N,J,T,S):
	h = T/M #temporal meshgrid
	h_S = S/J
	U_o = [0]*N
	V_lst = [U_o.copy()]
	V_k_y = [[0]*(J+1)]
	rand_lst = [] 
	for k in tqdm(range(1,M+1)):
		V_lst_zw =[]
		rand_lst_zw =[]
		for l in range(1, N+1):
			f_sum=nonlin_f_N_approx1(V_k_y,l,k,J,h_S)
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
	#structure: [[V_{0,1},...,V_{0,N}],[V_{1,1},...,V_{1,N}],...,[V_{M,1},...,V_{M,N}]])
	V_results = np.array(V_k_y)
	return V_results

def nonlinear_exp_euler2(M,N,J,T,S):
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
			f_sum=nonlin_f_N_approx2(V_k_y,l,k,J,h_S)
			#2) Calculation of V_k,l:
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

	#structure: [[V_{0,1},...,V_{0,N}],[V_{1,1},...,V_{1,N}],...,[V_{M,1},...,V_{M,N}]])
	V_results = np.array(V_k_y)
	return V_results
#### Output of function: 2D Array V =[[V_0(y_0),...,V_0(y_J)],[V_1(y_0),...,V_1(y_J)],...,[V_M(y_0),...,V_M(y_J)]]



M = 250 #temporal discretization
N = M**2
J = 600 #spacial discretization
T = 1 #length temporal interval
S = 1 
t_k_array = np.linspace(0,T,M+1)
y_k_array = np.linspace(0,S,J+1)
TT,XX = np.meshgrid(t_k_array,y_k_array)



######################################
# simulation of three paths per nonlinearity
fig_MC = plt.figure(figsize=(12, 4), dpi=80)
subfig0= fig_MC.add_subplot(131)
for i in tqdm(range(3)):
	results = nonlinear_exp_euler0(M,N,J,T,S)
	subfig0.plot(y_k_array,results[-1])

subfig1= fig_MC.add_subplot(132)
for i in tqdm(range(3)):
	results = nonlinear_exp_euler1(M,N,J,T,S)
	subfig1.plot(y_k_array,results[-1])


subfig2= fig_MC.add_subplot(133)
for i in tqdm(range(3)):
	results = nonlinear_exp_euler2(M,N,J,T,S)
	subfig2.plot(y_k_array,results[-1])

subfig0.set_xlabel('T=1')
subfig0.set_ylabel('X_{N,M}(1)')
subfig0.set_ylim([-3, 3])
subfig0.set_title('(F1)')
subfig1.set_xlabel('T=1')
subfig1.set_ylim([-3, 3])
subfig1.set_title('(F2)')
subfig2.set_xlabel('T=1')
subfig2.set_ylim([-3, 3])
subfig2.set_title('(F3)')
########################################

plt.show()
