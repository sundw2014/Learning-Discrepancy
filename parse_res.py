import numpy as np
import sys
a = np.loadtxt(sys.argv[1])
print('%f +/- %f'%(a[:,0].mean(), a[:,0].std()), '%f +/- %f'%(1 - a[:,1].mean(), a[:,1].std()))
