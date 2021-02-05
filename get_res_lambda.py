import numpy as np
import os
res = []
for _lambda in [0.001, 0.003, 0.009, 0.027, 0.081, 0.243, 0.729]:
    dir_name = 'log_jetengine_lambda%.3f'%_lambda
    os.system('bash ./test.sh jetengine %s | tee res_tmp.txt'%dir_name)
    res.append(np.loadtxt('res_tmp.txt').mean(axis=0))
print(np.array(res)[:,0])
print(np.array(res)[:,1])

