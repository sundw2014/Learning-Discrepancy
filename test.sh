# for seed in {10..20};
# do python plot_quick.py --pretrained log_jetEngine_dryvr/checkpoint.pth.tar --config jetengine --seed $seed;
# done

# for seed in {10..20};
# do python plot_quick.py --pretrained log_vanderpol_dryvr/checkpoint.pth.tar --config vanderpol --seed $seed;
# done

# for seed in {10..20};
# do python plot_quick.py --pretrained log_quadrotor_C3M/checkpoint.pth.tar --config quadrotor_C3M --seed $seed;
# done

echo '' > test.txt
for seed in {10..19};
do
    python plot_quick.py --pretrained $2/checkpoint.pth.tar --config $1 --seed $seed --output test.txt #1>/dev/null 2>&1
done
python parse_res.py test.txt
