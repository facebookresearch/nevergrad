import numpy as np
import matplotlib.pyplot as plt
import PyMoosh as pm

n_couches = "60"
runner = "A"
algo = "DE"
function = "bragg"
budget = 60000

file_name = f"ResA/{function}_{algo}_{n_couches}_{budget}.npy"
results = np.load(file_name,allow_pickle = True)


#Graphique pour comparaisons
values = []
bests = []
fig1 = plt.figure(1)

for k in range(len(results)):
    values.append(results[k][1][-1])
    bests.append(results[k][0])
sorted = np.sort(values)
agrum = np.argsort(values)
# Let's sort bests too !
sorted_bests = []
for k in range(len(results)):
    sorted_bests.append(bests[agrum[k]])

plt.plot(sorted)
plt.title(f'Function {function} with {algo}')

# Représenter les courbes de convergence

fig2 = plt.figure(2)
for k in range(len(results)):
    plt.plot(results[k][1])
plt.title("Convergences")

# Visualiser
fig3 = plt.figure(3)

# We select a solution to fully analyze.

X = sorted_bests[0]
nc = int(n_couches)
permittivities = X[0:nc]
thicknesses = X[nc:2*nc]
starts = np.concatenate((np.array([0]),np.cumsum(thicknesses[0:nc-1])))
plt.barh(starts,permittivities-2.,thicknesses,align = 'edge',color = 'green')
#plt.ylim(sum(thickness),0)
plt.gca().invert_yaxis()

materials = [1.]+permittivities.tolist()+[3.]
thicknesses = [0.] + thicknesses.tolist() + [0.]
stack = np.arange(0,len(materials))
crystal = pm.Structure(materials,stack,thicknesses,verbose = False)
[wl,r,t,R,T] = pm.Spectrum(crystal, 0., 0., 350, 800, 200)

fig4 = plt.figure(4)
plt.plot(wl,R,label = "Reflectance")
plt.show()
