for seed in 1 2 3 4 5 6 7 8 9 10
do
python plot.py --config jetengine --pretrained_ours ~/Downloads/log_jetengine-x_0.3_1.3-y_0.3_1.3_ellipsoid/checkpoint.pth.tar --pretrained_spherical ~/Downloads/log_jetengine-x_0.3_1.3-y_0.3_1.3_circle/checkpoint.pth.tar --pretrained_dryvr log_jetEngine_dryvr/checkpoint.pth.tar --seed $seed --no_plot y
done

0.004978106513926501 2.2502602264751785e-06 0.03719032001179128
9.082245064556547 51.971362136615994 21.32376789776357
2.945919944863095 14.557054074328914 6.498466823763085
23.25540059351267 162.74805783204508 57.358379918690076
2.204826578025265 7.350779432971427 5.0404782035618645
16.13713995014753 116.01137879737713 41.22596443439081
12.22768006803976 33.06216765962625 26.515849854103916
23.002416058989784 129.9588723597015 48.64350835395344
16.20296249527734 42.28031078001448 35.501565885504654
8.454772994526142 69.06664530039104 24.191253984551654

for seed in 1 2 3 4 5 6 7 8 9 10
do
python plot.py --config vanderpol --pretrained_ours log_vanderpol_ours/checkpoint.pth.tar --pretrained_spherical log_vanderpol_spherical/checkpoint.pth.tar --pretrained_dryvr log_vanderpol_dryvr/checkpoint.pth.tar --seed $seed --no_plot y
done

0.031286909287868016 1.943648221843494e-05 0.0640237889778528
111.55087930353692 448.89939579053424 239.57522819913393
15.917499595186454 125.73564574426356 39.79181253946225
164.07825493042492 1405.72618887385 275.2859401580876
5.598818373911171 63.491898430069035 15.841533628448003
115.46959665662298 1002.0410415658253 218.6595695313677
32.12748500870911 285.5724090301465 72.45535818897753
129.6339604452019 1122.511646443596 186.82868263222636
81.90659318231506 365.19354472744516 126.02823883557453
62.721350629171916 596.5588368287364 145.56380355016572

for seed in 1 2 3 4 5 6 7 8 9 10
do
python plot.py --config quadrotor_C3M --pretrained_ours log_quadrotor_C3M_ours/checkpoint.pth.tar --pretrained_spherical log_quadrotor_C3M_spherical/checkpoint.pth.tar --pretrained_dryvr log_quadrotor_C3M_dryvr/checkpoint.pth.tar --seed $seed --no_plot y
done

114.9561726469697 46.25300176607213 29.447037478348133
115.29357612805727 19.924841492077448 38.23237390128992
0.0629628177018715 0.10095396978383515 0.07142465521008946
1.7003908590497372 11.989538889764216 1.1610748386692389
72.31455380657356 19.36090425657027 43.62995659518747
30.936187881721416 27.941605595926735 42.55318862714394
7.794313703564249 14.324137931895914 8.764980031602585
38.34536616850241 105.736770978168 30.61382344608529
0.24353807720540138 0.43983972140397737 0.4658988502211845
8.725160359967914 3.581343900239778 4.733804184958352
