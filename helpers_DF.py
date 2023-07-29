import numpy as np
from numba import njit, prange, float64
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cmath import nan



@njit('(float64[:,:], float64[:,:], float64[:,:],float64, float64)', cache=True, fastmath=True, parallel=True)
def compute_acc_poisson(pos, mass,charge, k, softening):
    """ Computes the Acceleration of N bodies
	Args:
		pos (type=np.array, size= Nx3): x, y, z positions of the N particles
		mass (type=np.array, size= Nx1): mass of the particles
        k (float): Coulomb constant
		softening (float): softening parameter

	Returns:
		acc (type=np.array, size= Nx3): ax, ay, az accelerations of the N particles
	"""
    n = pos.shape[0]

    # Copy the array view so for the next loop to be faster
    x = pos[:,0].copy()
    y = pos[:,1].copy()
    z = pos[:,2].copy()

    # Ensure mass is a contiguous 1D array (cheap operation)
    assert mass.shape[1] == 1
    contig_mass = mass[:,0].copy()
    
    # Ensure charge is a contiguous 1D array (cheap operation)
    assert charge.shape[1] == 1
    contig_charge = charge[:,0].copy()

    acc = np.empty((n, 3), pos.dtype)

    for i in prange(n):
        ax, ay, az = 0.0, 0.0, 0.0

        for j in range(n):
            dx = x[i] - x[j]  
            dy = y[i] - y[j]
            dz = z[i] - z[j]
            tmp = (dx**2 + dy**2 + dz**2 + softening**2)
            factor = contig_charge[j] / (tmp * np.sqrt(tmp))
            ax += dx * factor
            ay += dy * factor
            az += dz * factor

        acc[i, 0] = k * contig_charge[i]/contig_mass[i] * ax
        acc[i, 1] = k * contig_charge[i]/contig_mass[i] * ay
        acc[i, 2] = k * contig_charge[i]/contig_mass[i] * az

    return acc




def leapfrog_kdk(pos,vel,acc,dt,mass,charge, k, softening):
	"""Takes the current position, velocity, and acceleration at time t. Then, it carries out
 the leapfrog scheme kick-drift-kick. It then returns the updated position, velocity and accelration
 at time t+dt.
 	Args:
		pos (np.array of Nx3): _Position x, y, and z of N particles_
		vel (np.array of Nx3): _Velocity vx, vy, and vz of N particles_
		acc (np.array of Nx3): _Acceleration ax, ay, and az of N particles_
		dt (float): _Timestep_
		mass (np.array of N): _Mass of N particles_
		k (float, optional): _Coulomb constant_.
		softening (float): _softening length_
	Returns:
		pos (np.array of Nx3): _New position x, y, and z of N particles_
		vel (np.array of Nx3): _New velocity vx, vy, and vz of N particles_
		acc (np.array of Nx3): _New acceleration ax, ay, and az of N particles_
	"""
	# (1/2) kick 
	vel += acc * dt/2.0
	# drift
	pos += vel * dt
	# update accelerations
	acc = compute_acc_poisson(pos, mass,charge, k, softening )
	# (1/2) kick
	vel += acc * dt/2.0
 
	return pos, vel, acc





def IC_conditions (n,prob,ri,zi,vri,vzi,Pneut,Pmono,Pdim,Ptrim):
    
    def injection_conditions(n,prob,ri,zi,vri,vzi):
        probabilities = prob / prob.sum()
        np.random.seed(123)    # set the random number generator seed
        indices = np.random.choice(np.arange(len(probabilities)), size=n, p=probabilities) # Indices are distributes based on prob
        theta_i=np.random.uniform(0, 2*np.pi, np.size(indices)) # The angle is uniformly distributed
        x=ri[indices]*np.cos(theta_i)
        y=ri[indices]*np.sin(theta_i)
        z=zi[indices]
        vx=vri[indices]*np.cos(theta_i)
        vy=vri[indices]*np.sin(theta_i)
        vz=vzi[indices] 
        init_posvel=np.column_stack((x,y,z,vx,vy,vz))
        return init_posvel
    
    
    def species(n,Pneut,Pmono,Pdim,Ptrim):
    
        # Neutrals-->0 ; Monomers-->1, dimers -->2 ; Trimers -->3
        particles = np.array([0, 1, 2, 3]) 
        probabilities = np.array([Pmono, Pdim, Ptrim, Pneut])  
        # Normalizing probabilities (making sure they add up to 1)
        probabilities = probabilities / probabilities.sum()
        # Generate the array
        particle_types = np.random.choice(particles, size=n, p=probabilities)
        
        return particle_types
    
    

    amu2kg= 1.66053906660 *1e-27 # converts amu to kg
    e2C= 1.602176634 *1e-19 # converts electron charge to Coulomb
    init_posvel=injection_conditions(n,prob,ri,zi,vri,vzi)
    particle_types=species(n,Pneut,Pmono,Pdim,Ptrim)
    charges=np.sign(particle_types)*e2C
    mass_list=np.array([197.973,111.168,309.141,507.114])*amu2kg  # mass in kg: neutral, monomer, dimer, trimer 
    masses=np.array([[mass_list[i] for i in list(particle_types)]]).T  # mass of the entire set of particles
    IC=np.column_stack(( init_posvel,particle_types,masses,charges))
    return IC











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





#@njit(cache=True,fastmath=True,nogil=False)
def DF_nbody_V2(dt,N,prob,ri,zi,vri,vzi,Pneut,Pmono,Pdim,Ptrim,softening,k):
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
    pos=IC[:,0:3]
    vel=IC[:,3:6]
    species=IC[:,6]
    mass=IC[:,7]
    charge=IC[:,8]
    
    mass=mass.reshape(-1, 1)
    charge=charge.reshape(-1, 1)
    
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








 













##### Animation codes #####   ------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------

def animate_injection_2D(species,pos_save):    
	"""Takes the list of species and the position of each particle at each timestep and creates
 an animation in real time.

	Args:
		species (np.array of Nx1): _Contains the type of the species of the N particles_
		pos_save (np.array of Nx3xN): _Contains the position of the particles at each time step_
	"""

	N=len(species)
	colors=["forestgreen","navy","fuchsia"]
	col_list=["forestgreen"]
	col_list=[colors [int(i)] for i in list(species[:-1])]
	fig = plt.figure(figsize=(4,5), dpi=80)
	grid = plt.GridSpec(1, 1, wspace=0.0, hspace=0.3)
	ax1 = plt.subplot(grid[0,0])
 
	xmax=np.nanmax(pos_save[:,0,-1])
	ymax=np.nanmax(pos_save[:,1,-1])
	for i in range(1,N):
		
		plt.sca(ax1)
		plt.cla()
		plt.title('Number of injected particles: %i' %i, fontsize=10)
		xx = pos_save[:i,0,i]
		yy = pos_save[:i,1,i]
		plt.scatter(xx,yy,s=5,color=col_list[:i])
		ax1.set(xlim=(-xmax, xmax), ylim=(-1e-7, ymax))
		ax1.set_aspect('equal', 'box')
		plt.pause(1e-5)
		ax1.set_xlabel('X axis')
		ax1.set_ylabel('Y axis')
		
	return 0




def animate_injection_3D(species,pos_save):    
	"""Takes the list of species and the position of each particle at each timestep and creates
 an animation in real time.

	Args:
		species (np.array of Nx1): _Contains the type of the species of the N particles_
		pos_save (np.array of Nx3xN): _Contains the position of the particles at each time step_
	"""

	N=len(species)
	colors=["forestgreen","navy","fuchsia","black"]
	col_list=[colors [int(i)] for i in list(species[:-1])]

	mono_patch = mpatches.Patch(color='forestgreen', label='Monomer')
	dim_patch = mpatches.Patch(color='navy', label='Dimer')
	neut_patch = mpatches.Patch(color='fuchsia', label='Neutral')

	xmax=np.nanmax(pos_save[:,0,-1])
	ymax=np.nanmax(pos_save[:,1,-1])
	zmax=np.nanmax(pos_save[:,1,-1])
 

	
	fig = plt.figure(figsize=(10,10), dpi=80)
	grid = plt.GridSpec(1, 1, wspace=0.0, hspace=0.3)
	ax = plt.axes(projection ='3d')

	for i in range(1,N):
		ax.cla()
		xx = pos_save[:i,0,i]
		yy = pos_save[:i,1,i]
		zz = pos_save[:i,2,i]
		ax.scatter(xx,yy,zz,color=col_list[:i])
		ax.set(xlim=(-zmax, zmax), ylim=(-zmax, zmax),zlim=(0, zmax))
		plt.title('Number of injected particles: %i' %i, fontsize=16)
		plt.legend(handles=[mono_patch, dim_patch,neut_patch])
		ax.set_aspect('auto', 'box')
		ax.set_xlabel('X axis')
		ax.set_ylabel('Y axis')
		ax.set_zlabel('Z axis')
		#ax.set_xticks([-200,-100,0,100,150,200])
		#ax.set_yticks([0,100,200,300,400,500,600])
		ax.view_init(elev=30., azim=35)
		plt.pause(1e-15)
  
	plt.savefig('animation.png',dpi=240)
	return 0
