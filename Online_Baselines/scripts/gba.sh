# nohup python main.py --conf conf/GBA/trafficstream.json  --method TrafficStream --gpuid 0 > /dev/null 2>&1 &

nohup python main.py --conf conf/GBA/online_st_nn.json  --method TrafficStream --gpuid 2 > /dev/null 2>&1 &

nohup python main.py --conf conf/GBA/online_st_an.json  --method TrafficStream --gpuid 3 > /dev/null 2>&1 &

nohup python main.py --conf conf/GBA/retrain_st_sd.json  --method TrafficStream --gpuid 1 &

nohup python main.py --conf conf/GBA/pretrain_st_sd.json  --method TrafficStream --gpuid 5 > /dev/null 2>&1 &

nohup python main.py --conf conf/GBA/eac.json  --method EAC --gpuid 6 > /dev/null 2>&1 &

# nohup python main.py --conf conf/GBA/team.json  --method TEAM --gpuid 6  &

# nohup python main.py --conf conf/GBA/dlf.json  --method DLF --gpuid 7 > /dev/null 2>&1 &

# nohup python stkec_main.py --conf conf/GBA/stkec.json  --method STKEC --gpuid 2 > /dev/null 2>&1 &

# nohup python pecpm_main.py --conf conf/GBA/pecpm.json  --method PECPM --gpuid 1 > /dev/null 2>&1 &


