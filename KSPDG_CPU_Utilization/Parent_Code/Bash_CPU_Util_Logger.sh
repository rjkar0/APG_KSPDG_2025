#!/bin/bash

for ((i=0; i<10; i++))
do
    uptime >> /home/hwil//Desktop/KSPDG/KSPDG_CPU_Utilization/cpuload.log
    sleep 1
done
