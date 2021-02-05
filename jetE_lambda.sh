mkdir log_jetengine_lambda$1
python main.py --config jetengine --log log_jetengine_lambda$1/ --data_file_train data/jetEngine_tr.pklz --data_file_eval data/jetEngine_te.pklz --lambda $1 --epochs 10 --lr_step 4
