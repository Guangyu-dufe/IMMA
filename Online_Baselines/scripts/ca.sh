nohup python main.py --conf conf/CA/trafficstream.json  --method TrafficStream --gpuid 0  &

nohup python main.py --conf conf/CA/online_st_nn.json  --method TrafficStream --gpuid 2  &

nohup python main.py --conf conf/CA/online_st_an.json  --method TrafficStream --gpuid 3  &

nohup python main.py --conf conf/CA/retrain_st_sd.json  --method TrafficStream --gpuid 1 &

nohup python main.py --conf conf/CA/pretrain_st_sd.json  --method TrafficStream --gpuid 5   &

nohup python main.py --conf conf/CA/eac.json  --method EAC --gpuid 6   &

# nohup python main.py --conf conf/CA/team.json  --method TEAM --gpuid 6  &

# nohup python main.py --conf conf/CA/dlf.json  --method DLF --gpuid 7   &

# nohup python stkec_main.py --conf conf/CA/stkec.json  --method STKEC --gpuid 2   &

# nohup python pecpm_main.py --conf conf/CA/pecpm.json  --method PECPM --gpuid 1   &


