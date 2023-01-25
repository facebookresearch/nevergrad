import numpy as np
import matplotlib.pyplot as plt

n_couches = "120"
runner = "A"
function = "BraggO"
budget = 20000

plt.clf()
fig1 = plt.figure(1)

for algo in ["BFGS","DE"]:

    file_name = f"../ResA/{function}_{algo}_{n_couches}_{budget}.npy"
    results = np.load(file_name,allow_pickle = True)
    values= []
    for k in range(len(results)):
        values.append(results[k][1][-1])
    sorted = np.sort(values)
    plt.plot(sorted,label=algo)

# Représenter les courbes de convergence
plt.legend()

fig2 = plt.figure(2)
for k in range(len(results)):
    plt.plot(results[k][1])
plt.title("Convergences")
plt.show()
