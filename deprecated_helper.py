# The following functions are no longer utilized but are kept temporarily here for future deletion.


import numpy as np
from numba import njit, prange, float64
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cmath import nan


def initial_conditions(N,k,softening,vy):
    """Creates the initial conditions of the simulation
 
	Args:
		N (_int_): _Number of particles_
        k (float, optional): _Coulomb constant_. 
		softening (float, optional): _softening parameter_. 
		vy (float, optional): _velocity in the y direction_. 
	"""
    # Generate Initial Conditions of the first injected particle 
    np.random.seed(123)    # set the random number generator seed
    #species=np.random.randint(3, size=N) #Chose one species among 3 at each timestep
    species=np.random.choice([0,0,0,0,0,0,0,0,1,2], N) #Chose one species among 3 at each timestep
    amu2kg= 1.66053906660 *1e-27 # converts amu to kg
    e2C= 1.602176634 *1e-19 # converts electron charge to Coulomb
    
    masses=np.array([111.168,309.141,197.973])*amu2kg  # define 3 different species with 3 different masses in kg 
    charges=np.array([1.,1.,0])*e2C  # define 3 different species with different charges in Coulomb
    
    mass=np.array([[masses[i] for i in list(species)]]).T  # mass of the entire set of particles
    charge=np.array([[charges[i] for i in list(species)]]).T  # charge of the entire set of particles
    
    pos=np.zeros([N,3]) # initial position of all the set of particles 
    vel= np.hstack([np.array([[0.5*vy*np.random.uniform(-1,1) for i in range(N)]]).T,\
         np.repeat([[vy]],N,axis=0),\
         np.array([[0.5*vy*np.random.uniform(-1,1) for i in range(N)]]).T]) # velocity of the enitre set of particles
    
    
    #acc[0] = compute_acc_poisson(pos[0:1], mass[0:1], charge[0:1], k, softening ) # calculate initial gravitational accelerations
    return pos, vel, mass, charge, species





#@njit(cache=True,fastmath=True,nogil=False)
def DF_nbody(N,dt,softening,k,vy):
    """Direct Force computation of the N body problem. The complexity of this algorithm
    is O(N^2)
 
    Args:
		N (_int_): Number of injected particles
    	dt (_float_): _timestep_
    	softening (float, optional): _softening parameter_. Defaults to 0.01.
    	k (float, optional): _Coulomb constant_. Defaults to 8.9875517923*1e9.
    	vy (float, optional): _velocity in the y direction_. Defaults to 50.0.
    """
    
    pos, vel, mass, charge, species = initial_conditions(N,softening,k,vy)
    acc=np.zeros([N,3]) # initial acceleration of all the set of particles 
	# pos_save: saves the positions of the particles at each time step
    pos_save = np.ones((N,3,N))*nan
    pos_save[0,:,0] = pos[0:1]
 
 	#vel_save: saves the velocities of the particles at each time step for computing the energy at each time step
    #vel_save = np.ones((N,3,N))*nan
    #vel_save[0,:,0] = vel[0:1]

	#species=np.random.randint(3, size=N)  # Check this one out
	# Simulation Main Loop
    for i in range(1,N):
		# Run the leapfrog scheme:
        pos[0:i],vel[0:i],acc[0:i]=leapfrog_kdk(pos[0:i],vel[0:i],acc[0:i],dt,mass[0:i],charge[0:i], k, softening)
  		# save the current position and velocity of the 0 to i particles:
        pos_save[:i,:,i] = pos[0:i]
        #vel_save[:i,:,i] = vel[0:i]
		
    return species, pos_save 



def DF_nbody2(dt,N,prob,ri,zi,vri,vzi,Pneut,Pmono,Pdim,Ptrim,softening,k,interp):
    """Direct Force computation of the N body problem. The complexity of this algorithm
    is O(N^2)
 
    Args:
		N (_int_): Number of injected particles
    	dt (_float_): _timestep_
    	softening (float, optional): _softening parameter_. Defaults to 0.01.
    	k (float, optional): _Coulomb constant_. Defaults to 8.9875517923*1e9.
    	vy (float, optional): _velocity in the y direction_. Defaults to 50.0.
    """
    IC=IC_conditions (N,prob,ri,zi,vri,vzi,Pneut,Pmono,Pdim,Ptrim)
    IC_copy=np.copy(IC)
    pos=IC[:,0:3]
    vel=IC[:,3:6]
    species=IC[:,6]
    mass=IC[:,7]
    charge=IC[:,8]
    
    mass=mass.reshape(-1, 1)
    charge=charge.reshape(-1, 1)
    
    acc=np.zeros([N,3]) # initial acceleration of all the set of particles
     
	# pos_save: saves the positions of the particles at each time step
    pos_save = np.ones((N,3,N))*np.nan
    pos_save[0,:,0] = pos[0:1]
 
 	#vel_save: saves the velocities of the particles at each time step for computing the energy at each time step
    #vel_save = np.ones((N,3,N))*nan
    #vel_save[0,:,0] = vel[0:1]

	# Simulation Main Loop
    current_step=0
    for i in range(1,N):
		# Run the leapfrog scheme:
        pos[0:i],vel[0:i],acc[0:i]=leapfrog_kdk(pos[0:i],vel[0:i],acc[0:i],dt,mass[0:i],charge[0:i], k, softening,interp,current_step)
  		# save the current position and velocity of the 0 to i particles:
        pos_save[:i,:,i] = pos[0:i]
        #vel_save[:i,:,i] = vel[0:i]
        current_step += 1
		
    return species, pos_save , IC_copy

