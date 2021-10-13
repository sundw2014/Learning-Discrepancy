for t in jetengine vanderpol quadrotor_C3M f16_GCAS_sphInit f16_GCAS
do 
	python main.py --config $t --log log/log_${t}_ours/ --data_file_train data/${t}_traces_train.pklz --data_file_eval data/${t}_traces_eval.pklz
	python main_dryvr.py --config $t --log log/log_${t}_dryvr/ --data_file_train data/${t}_traces_train.pklz --data_file_eval data/${t}_traces_eval.pklz
done

for t in jetengine vanderpol quadrotor_C3M f16_GCAS_sphInit f16_GCAS
do
	./test.sh $t log/log_${t}_ours
	./test.sh $t log/log_${t}_dryvr
done
