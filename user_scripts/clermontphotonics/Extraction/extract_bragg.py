import numpy as np
import matplotlib.pyplot as plt

n_couches = "20"
runner = "A"
algo = "DE"
function = "bragg"
budget = 10000

file_name = f"../Res1/out_{function}_{algo}_{n_couches}_{budget}_{runner}.npy"
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

X = sorted_bests[0]
nc = int(n_couches)
permittivity = X[0:nc]
thickness = X[nc:2*nc]
starts = np.concatenate((np.array([0]),np.cumsum(thickness[0:nc-1])))
plt.barh(starts,permittivity-2.,thickness,align = 'edge',color = 'green')
#plt.ylim(sum(thickness),0)
plt.gca().invert_yaxis()
plt.show()
