import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pylab as plt
import pandas as pd
from matplotlib.colors import LogNorm
info = np.loadtxt('E_field/info.txt')
convert_dist=info[1] 
convert_E=info[2]
convert_vel=info[3]

# Load the csv file
df = pd.read_csv('E_field/Fields.csv')

# Convert the DataFrame to a numpy array
Field = df.values
r=Field[:,0]*convert_dist
z=Field[:,1]*convert_dist
Er=Field[:,2]*convert_E
Ez=Field[:,3]*convert_E

E = np.sqrt(Er**2 + Ez**2)


R = np.loadtxt('E_field/r.txt')*convert_dist
Z = np.loadtxt('E_field/z.txt')*convert_dist

points = np.column_stack((r, z))
tri = Delaunay(points)

fig = plt.figure(figsize=(6,8), facecolor='white')
ax = fig.add_subplot(111, facecolor='black')

ax.triplot(points[:,0]*1e6, points[:,1]*1e6, tri.simplices, color='white', linewidth=0.5,alpha=1)

#plt.tripcolor(points[:,0], points[:,1], tri.simplices.copy())
ax.scatter(points[:,0]*1e6, points[:,1]*1e6, color='r',marker='.', s=1, alpha=0.1)

ax.set_xlabel('r ($\mu m$)', size=15, color='black')
ax.set_ylabel('z ($\mu m$)', size=15, color='black')
# Set the x and y limits to the minimum and maximum of the data
ax.set_xlim([r.min()*1e6, r.max()*1e6])
ax.set_ylim([z.min()*1e6, z.max()*1e6])

plt.show()