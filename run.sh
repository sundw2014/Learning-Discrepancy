mkdir log_vanderpol_spherical/
mkdir log_vanderpol_spherical_0.1/
mkdir log_vanderpol_ours/
mkdir log_vanderpol_ours_0.1/
# python main.py --use_spherical --config vanderpol --log log_vanderpol_spherical/ --alpha 0.1 --lambda1 0. --lambda2 1. --lr_step 5 --lr 0.001 --epochs 15 --data_file_train data/vanderpol_traces_train.pklz --data_file_eval data/vanderpol_traces_eval.pklz
python main.py --use_spherical --config vanderpol --log log_vanderpol_spherical_0.1/ --alpha 0.1 --lambda1 0. --lambda2 0.1 --lr_step 5 --lr 0.001 --epochs 6 --data_file_train data/vanderpol_traces_train.pklz --data_file_eval data/vanderpol_traces_eval.pklz
# python main.py --config vanderpol --log log_vanderpol_ours/ --alpha 0.1 --lambda1 0. --lambda2 1. --lr_step 5 --lr 0.001 --epochs 15 --data_file_train data/vanderpol_traces_train.pklz --data_file_eval data/vanderpol_traces_eval.pklz
python main.py --config vanderpol --log log_vanderpol_ours_0.1/ --alpha 0.1 --lambda1 0. --lambda2 0.1 --lr_step 5 --lr 0.001 --epochs 6 --data_file_train data/vanderpol_traces_train.pklz --data_file_eval data/vanderpol_traces_eval.pklz
