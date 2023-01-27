import numpy as np
import matplotlib.pyplot as plt

n_couches = "20"
function = "BraggO"

plt.clf()
fig1 = plt.figure(1)

budget = 20000

# Pour pouvoir choisir un budget différent pour BGFS. Avec 20 000 il a convergé.
for algo in ["BFGS"]:

    file_name = f"../ResA/{function}_{algo}_{n_couches}_{budget}.npy"
    results = np.load(file_name,allow_pickle = True)
    values= []
    for k in range(len(results)):
        values.append(results[k][1][-1])
    sorted = np.sort(values)
    plt.plot(sorted,label=algo)

budget = 20000

for algo in ["DE"]:

    file_name = f"../ResA/{function}_{algo}_{n_couches}_{budget}.npy"
    results = np.load(file_name,allow_pickle = True)
    values= []
    for k in range(len(results)):
        values.append(results[k][1][-1])
    sorted = np.sort(values)
    plt.plot(sorted,label=algo)

# Représenter les courbes de convergence
plt.legend()

#fig2 = plt.figure(2)
#for k in range(len(results)):
#    plt.plot(results[k][1])
#plt.title("Convergences")
plt.show()
