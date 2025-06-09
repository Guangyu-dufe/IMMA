nohup python main.py --conf conf/GLA/trafficstream.json  --method TrafficStream --gpuid 1 &

nohup python main.py --conf conf/GLA/online_st_nn.json  --method TrafficStream --gpuid 2 &

nohup python main.py --conf conf/GLA/online_st_an.json  --method TrafficStream --gpuid 3 &

nohup python main.py --conf conf/GLA/retrain_st_sd.json  --method TrafficStream --gpuid 4 &

nohup python main.py --conf conf/GLA/pretrain_st_sd.json  --method TrafficStream --gpuid 5 &

nohup python main.py --conf conf/GLA/eac.json  --method EAC --gpuid 6 &

nohup python main.py --conf conf/GLA/team.json  --method TEAM --gpuid 7 &

# nohup python main.py --conf conf/GLA/dlf.json  --method DLF --gpuid 7 &

# nohup python stkec_main.py --conf conf/GLA/stkec.json  --method STKEC --gpuid 2 &

# nohup python pecpm_main.py --conf conf/GLA/pecpm.json  --method PECPM --gpuid 3 &


# # 释放内存缓存
# sync
# echo 3 > /proc/sys/vm/drop_caches
# free -h
