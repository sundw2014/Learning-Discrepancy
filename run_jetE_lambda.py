import numpy as np
import os
import time
dir_base = 'log_lambdas'

res = []
# os.system('mkdir %s'%dir_base)
# os.system('mkdir %s/data'%dir_base)
for i in range(10):
    res.append([])
    # os.system('mkdir %s/%d'%(dir_base, i))
    for _lambda in [0.001, 0.003, 0.009, 0.027, 0.081, 0.243, 0.729]:
        time.sleep(1)
        dir_name = '%s/%d/log_jetengine_lambda%.3f'%(dir_base, i, _lambda)
        # os.system('mkdir %s'%dir_name)
        # os.system('python main.py --config jetengine --log %s --data_file_train %s/data/jetEngine_tr_lambdas_%d.pklz --data_file_eval %s/data/jetEngine_te_lambdas_%d.pklz --lambda %f --epochs 10 --lr_step 4 --seed %d'%(dir_name, dir_base, i, dir_base, i, _lambda, i))
        os.system('bash ./test.sh jetengine %s | tee res_tmp.txt'%dir_name)
        res[-1].append(np.loadtxt('res_tmp.txt').mean(axis=0))
np.save(dir_base+'/res.npy', np.array(res))