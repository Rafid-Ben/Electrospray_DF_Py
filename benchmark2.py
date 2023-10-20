from cmath import nan
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from helpers_DF import *
import time

# Fields.csv
# r  |  z  |  Er | Ez
# Electric current = info[0] 
# Distance scaling to meter = multiply by info[1] 
# Electric Field (Er, Ez) scaling to V/m = multiply by info[2]
# Velocity scaling to m/s = multiply by info[3];
# Extractor voltage = info[6]; 

#EXTRACTOR_DIST = info[10]; //extractor distance
#EXTRACTOR_APETURE_DIA = info[11]; //extractor width
#EXTRACTOR_THICKNESS = info[12]; //extractor thickness


softening=1e-6
k=8.9875517923*1e9

info = np.loadtxt('E_field/info.txt')
convert_dist=info[1] 
convert_E=info[2]
convert_vel=info[3]

prob = np.loadtxt('E_field/prob.txt')
ri = np.loadtxt('E_field/r.txt')*convert_dist  # r initial condition in (m)
zi = np.loadtxt('E_field/z.txt')*convert_dist  # z initial condition in (m)
vri = np.loadtxt('E_field/v_r.txt')*convert_vel # vr initial condition (m/s)
vzi = np.loadtxt('E_field/v_z.txt')*convert_vel # vz initial condition (m/s)

# Load the csv file
df = pd.read_csv('E_field/Fields.csv')
# Convert the DataFrame to a numpy array
Field = df.values
r=Field[:,0]*convert_dist # r grid of the electric field (m)
z=Field[:,1]*convert_dist # z grid of the electric field (m)
Er=Field[:,2]*convert_E # Radial Electric Field Er in (V/m)
Ez=Field[:,3]*convert_E # Axial Electric Field Ez in (V/m)

interp=triangulation (r,z,Er,Ez)
dt=5e-12
Pneut=0
Pmono=40
Pdim=40
Ptrim=20




N_values=np.arange(1000,10001,1000)

# number of CPUs=1
set_num_threads(1)
times1 = []
for N in N_values:
    start_time = time.time()
    species, pos_save,IC,counters = DF_nbody(dt,N,prob,ri,zi,vri,vzi,Pneut,Pmono,Pdim,Ptrim,softening,k,interp)
    end_time = time.time()
    elapsed_time = end_time - start_time
    times1.append(elapsed_time)
    print(f"Elapsed time for N={N}: {elapsed_time} seconds")
    


# number of CPUs=2
set_num_threads(2)
times2 = []
for N in N_values:
    start_time = time.time()
    species, pos_save,IC,counters = DF_nbody(dt,N,prob,ri,zi,vri,vzi,Pneut,Pmono,Pdim,Ptrim,softening,k,interp)
    end_time = time.time()
    elapsed_time = end_time - start_time
    times2.append(elapsed_time)
    print(f"Elapsed time for N={N}: {elapsed_time} seconds")
    
    
# number of CPUs=2
set_num_threads(2)
times2 = []
for N in N_values:
    start_time = time.time()
    species, pos_save,IC,counters = DF_nbody(dt,N,prob,ri,zi,vri,vzi,Pneut,Pmono,Pdim,Ptrim,softening,k,interp)
    end_time = time.time()
    elapsed_time = end_time - start_time
    times2.append(elapsed_time)
    print(f"Elapsed time for N={N}: {elapsed_time} seconds")
    

    
# number of CPUs=4
set_num_threads(4)
times4 = []
for N in N_values:
    start_time = time.time()
    species, pos_save,IC,counters = DF_nbody(dt,N,prob,ri,zi,vri,vzi,Pneut,Pmono,Pdim,Ptrim,softening,k,interp)
    end_time = time.time()
    elapsed_time = end_time - start_time
    times4.append(elapsed_time)
    print(f"Elapsed time for N={N}: {elapsed_time} seconds")
    

# number of CPUs=8
set_num_threads(8)
times8 = []
for N in N_values:
    start_time = time.time()
    species, pos_save,IC,counters = DF_nbody(dt,N,prob,ri,zi,vri,vzi,Pneut,Pmono,Pdim,Ptrim,softening,k,interp)
    end_time = time.time()
    elapsed_time = end_time - start_time
    times8.append(elapsed_time)
    print(f"Elapsed time for N={N}: {elapsed_time} seconds")
    
    
    
# Save the lists into a single .npz file
np.savez('times_data.npz', times1=times1, times2=times2, times4=times4, times8=times8)


# Call
# data = np.load('times_data.npz')
# times1_loaded = data['times1']
# times2_loaded = data['times2']
# times4_loaded = data['times4']
# times8_loaded = data['times8']
