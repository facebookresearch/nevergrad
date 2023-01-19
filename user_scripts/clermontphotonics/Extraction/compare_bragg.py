import numpy as np
import matplotlib.pyplot as plt

n_couches = "100"
runner = "A"
function = "bragg"
budget = 60000

plt.clf()
fig1 = plt.figure(1)

for algo in ["BFGS","DE"]:

    file_name = f"../Res1/out_{function}_{algo}_{n_couches}_{budget}_{runner}.npy"
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
