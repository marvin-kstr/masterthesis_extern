
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

###### Functions #########

# eigenvalue of A
def lambda_k(j):
	return 0.1*pi**2*j**2

# eigenfunction of A
def phi(k,x):
	if x == 1 or x==0: 
		return 0
	else:
		return sqrt(2)*sin(pi*k*x)

#nonlinearity:
def init_func(x):
	ret = 0
	if x >0 and x<1:
		ret = exp(-(x-0.5)**2/(0.5**2-(x-0.5)**2))
	else:
		ret = 0
	return ret/100

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

def nonlin_f_N_approx3(V_k_y,l,k,J,h_S):
	f_sum = 0
	for j in range(1,J):
		if l%J==0:
			f_sum += 0
		else:
			f_sum += sqrt(2)*h_S*nonlin_f3(V_k_y[k-1][j])*sin(pi*l*(j*h_S))
	return f_sum

# exponential euler method

def nonlinear_exp_euler0(M,N,J,T,S):
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
			f_sum=nonlin_f_N_approx0(V_k_y,l,k,J,h_S)
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


def nonlinear_exp_euler1(M,N,J,T,S,rand_arr):
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
			f_sum=nonlin_f_N_approx1(V_k_y,l,k,J,h_S)
			np.random.seed()
			rand = rand_arr[k-1][l-1]
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

	#################### return 2D numpy array with Data V on meshgrid######################
	return V_results
def nonlinear_exp_euler2(M,N,J,T,S,rand_arr):
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
			np.random.seed()
			rand = rand_arr[k-1][l-1]
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



#### Output of function: 2D Array V =[[V_0(y_0),...,V_0(y_J)],[V_1(y_0),...,V_1(y_J)],...,[V_M(y_0),...,V_M(y_J)]]

M = 250 #temporal discretization
N = M**2
J = 600 #spacial discretization
T = 1 #length temporal interval
S = 1 
t_k_array = np.linspace(0,T,M+1)
y_k_array = np.linspace(0,S,J+1)
TT,XX = np.meshgrid(t_k_array,y_k_array)


#### Figures ##########
## Figure 1 = Surfaceplot
## Figure = Path developmet plot

fig1,(dreid0,dreid1,dreid2) = plt.subplots(1,3,figsize=(14,4),subplot_kw=dict(projection='3d'))
fig1.tight_layout()

fig = plt.figure(figsize=(8, 8), dpi=80)
fig.tight_layout()
results = nonlinear_exp_euler0(M,N,J,T,S)


dreid0.plot_surface(TT.T, XX.T, results[0] ,cmap=cm.coolwarm,vmin=-3, vmax=3) # needs to be transposed: see file "test_matrix.py"
dreid0.set_xlabel('T')
dreid0.set_ylabel('X')
dreid0.set_zlim([-3, 3])
dreid0.set_title('(F1)')
dreid0.w_zaxis.line.set_lw(0.)
dreid0.set_zticks([])

subfig00= fig.add_subplot(5,3,1)
subfig00.plot(y_k_array,results[0][0])
subfig00.set_ylim([-3, 3])
subfig00.set_title('(F1)')
subfig00.set_xticks([])
subfig001= fig.add_subplot(5,3,4)
subfig001.plot(y_k_array,results[0][int(int(M/10))])
subfig001.set_ylim([-3, 3])
subfig001.set_xticks([])
subfig005= fig.add_subplot(5,3,7)
subfig005.plot(y_k_array,results[0][int(int(M/4))])
subfig005.set_ylim([-3, 3])
subfig005.set_xticks([])
subfig0075= fig.add_subplot(5,3,10)
subfig0075.plot(y_k_array,results[0][int(2*int(M/4))])
subfig0075.set_ylim([-3, 3])
subfig0075.set_xticks([])
subfig01= fig.add_subplot(5,3,13)
subfig01.plot(y_k_array,results[0][-1])
subfig01.set_ylim([-3, 3])

results1 = nonlinear_exp_euler1(M,N,J,T,S,results[1])
dreid1.plot_surface(TT.T, XX.T, results1 ,cmap=cm.coolwarm,vmin=-3, vmax=3) # needs to be transposed: see file "test_matrix.py"
dreid1.set_xlabel('T')
dreid1.set_ylabel('X')
dreid1.set_zlim([-3, 3])
dreid1.set_title('(F2)')
dreid1.w_zaxis.line.set_lw(0.)
dreid1.set_zticks([])

subfig10= fig.add_subplot(5,3,2)
subfig10.plot(y_k_array,results1[0])
subfig10.set_xlabel('T=0')
subfig10.set_ylim([-3, 3])
subfig10.set_title('(F2)')
subfig10.set_xticks([])
subfig10.set_yticks([])
subfig101= fig.add_subplot(5,3,5)
subfig101.plot(y_k_array,results1[int(int(M/10))])
subfig101.set_xlabel('T=0.1')
subfig101.set_ylim([-3, 3])
subfig101.set_xticks([])
subfig101.set_yticks([])
subfig105= fig.add_subplot(5,3,8)
subfig105.plot(y_k_array,results1[int(int(M/4))])
subfig105.set_xlabel('T=0.25')
subfig105.set_ylim([-3, 3])
subfig105.set_xticks([])
subfig105.set_yticks([])
subfig1075= fig.add_subplot(5,3,11)
subfig1075.plot(y_k_array,results1[int(2*int(M/4))])
subfig1075.set_xlabel('T=0.5')
subfig1075.set_ylim([-3, 3])
subfig1075.set_xticks([])
subfig1075.set_yticks([])
subfig11= fig.add_subplot(5,3,14)
subfig11.plot(y_k_array,results1[-1])
subfig11.set_xlabel('T=1')
subfig11.set_ylim([-3, 3])
subfig11.set_yticks([])

results2 = nonlinear_exp_euler2(M,N,J,T,S,results[1])
dreid2.plot_surface(TT.T, XX.T, results2 ,cmap=cm.coolwarm,vmin=-3, vmax=3) # needs to be transposed: see file "test_matrix.py"
dreid2.set_xlabel('T')
dreid2.set_ylabel('X')
dreid2.set_zlim([-3, 3])
dreid2.set_title('(F3)')
dreid2.w_zaxis.line.set_lw(0.)
dreid2.set_zticks([])

subfig20= fig.add_subplot(5,3,3)
subfig20.plot(y_k_array,results2[0])
subfig20.set_ylim([-3, 3])
subfig20.set_title('(F3)')
subfig20.set_xticks([])
subfig20.set_yticks([])
subfig201= fig.add_subplot(5,3,6)
subfig201.plot(y_k_array,results2[int(int(M/10))])
subfig201.set_ylim([-3, 3])
subfig201.set_xticks([])
subfig201.set_yticks([])
subfig205= fig.add_subplot(5,3,9)
subfig205.plot(y_k_array,results2[int(int(M/4))])
subfig205.set_ylim([-3, 3])
subfig205.set_xticks([])
subfig205.set_yticks([])
subfig2075= fig.add_subplot(5,3,12)
subfig2075.plot(y_k_array,results2[int(2*int(M/4))])
subfig2075.set_ylim([-3, 3])
subfig2075.set_xticks([])
subfig2075.set_yticks([])
subfig21= fig.add_subplot(5,3,15)
subfig21.plot(y_k_array,results2[-1])
subfig21.set_ylim([-3, 3])
subfig21.set_yticks([])

m = cm.ScalarMappable(cmap=cm.coolwarm)
z1= np.linspace(-3,3,100)
m.set_array(z1)
fig1.colorbar(m, ax=(dreid0,dreid1,dreid2), orientation='vertical')


plt.show()
