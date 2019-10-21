#!/bin/bash
source /home/bjlee/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/bjlee/.mujoco/mujoco200/bin
cd /home/bjlee/PycharmProjects/RepBM
/home/bjlee/anaconda3/envs/MBRLnets/bin/python main_cont.py --pid=$1
