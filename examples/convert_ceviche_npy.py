import glob
import numpy as np
import matplotlib.pyplot as plt
import os.path

def rename(s):
    return s.replace("pb0", "pb0bend").replace("pb1","pb1beamsplitter").replace("pb2", "pb2modeconverter")

num = len(glob.glob("*.npy"))


for idx, f in enumerate(glob.glob("*iel*.npy")):
    printing = False
    if int(np.sqrt(idx)) < int(np.sqrt(idx+1)):
        print(idx, " / ", num, " : ", f)
        printing = True
    if (("0.000" not in f and "pb0" in f) or ("0.00" not in f) or ("pb3" in f and "0.0" not in f)) and "WS" not in f:
        if printing:
            print("      ... skipping ")
        continue
    data = np.load(f)
    #print('Shape:',data.shape)
    if len(data.shape) <3:
        continue
    assert data.shape[0] < 3, f"{f} leads to an issue!"
    assert data.shape[1] == 1
    assert len(data.shape) == 4
    if data.shape[0]==2:
      target_name = rename(f) + "_1st_wavelength.png"
      target_name2 = rename(f) + "_2nd_wavelength.png"
      if not os.path.isfile(target_name):
        field = np.abs(data[0, 0, :, :])**2
        plt.imshow(field.T, cmap='viridis')
        plt.savefig(target_name)
        field = np.abs(data[1, 0, :, :])**2
        plt.imshow(field.T, cmap='viridis')
        plt.savefig(target_name2)
    else:
      target_name = rename(f) + ".png"
      if not os.path.isfile(target_name):
        field = np.abs(data[0, 0, :, :])**2
        plt.imshow(field.T, cmap='viridis')
        plt.savefig(target_name)

