for i in {0..15}
do
    python plot_quick.py --pretrained log_f16/checkpoint.pth.tar --config f16_GCAS --seed 1000 --id $i
done
