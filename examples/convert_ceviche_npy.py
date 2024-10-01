import glob
import numpy as np
import matplotlib.pyplot as plt

for f in glob.glob("*.npy"):
    data = np.load(f)
    print('Shape:',data.shape)
    field = np.abs(data[0, 0, :, :])
    plt.imshow(field.T, cmap='viridis')
    plt.colorbar()  
    plt.savefig(f + ".png")


