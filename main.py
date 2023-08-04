
#import numpy as np
from helpers_DF import *
import time
import timeit


softening=1e-6
k=8.9875517923*1e9

info = np.loadtxt('E_field/info.txt')
convert_dist=info[1] 
convert_E=info[2]
convert_vel=info[3]

prob = np.loadtxt('E_field/prob.txt')
ri = np.loadtxt('E_field/r.txt')*convert_dist  # r initial condition
zi = np.loadtxt('E_field/z.txt')*convert_dist  # z initial condition
vri = np.loadtxt('E_field/v_r.txt')*convert_vel # vr initial condition
vzi = np.loadtxt('E_field/v_z.txt')*convert_vel # vz initial condition

N=1000
dt=5e-12
Pneut=50
Pmono=50
Pdim=10
Ptrim=50






species, pos_save = DF_nbody(dt,N,prob,ri,zi,vri,vzi,Pneut,Pmono,Pdim,Ptrim,softening,k)
#T1= min(timeit.repeat(stmt='DF_nbody(N,dt,softening,k,vy)',\
 #                     timer=time.perf_counter,repeat=3, number=1,globals=globals()) )
#print("Compute time=",T1,"\n")


animate_injection_3D(species,pos_save)

