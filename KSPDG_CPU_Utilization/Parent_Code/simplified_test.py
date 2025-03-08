import logging
import logging.handlers
import os
import pathlib
import psutil
import subprocess
import sys
import time
import numpy as np

# Initialize the matricies and arrays
cpu_tot = np.array([])
cpu_ind = np.empty((0, os.cpu_count()))
cpu_ind_avg = np.array([])

# Find current time
s = time.time()

# Main loop for logging CPU usage
while time.time() < s + 120:
    cpu_percent_total = psutil.cpu_percent(interval=2)
    cpu_percent_cores = psutil.cpu_percent(percpu=True)
    cpu_percent_total_str = ('%.2f' % cpu_percent_total) + "%"
    cpu_percent_cores_str = [('%.2f' % x) + "%" for x in cpu_percent_cores]
    print('Total: {}'.format(cpu_percent_total_str))
    print('Individual CPUs: {}'.format('  '.join(cpu_percent_cores_str)))

    # Sanity check
    avg = sum(cpu_percent_cores)/len(cpu_percent_cores)
    print("DEBUG PRINT: avg = {:.2f}%".format(avg))

    cpu_tot = np.append(cpu_tot, cpu_percent_total)
    cpu_ind = np.vstack((cpu_ind, cpu_percent_cores))
    cpu_ind_avg = np.append(cpu_ind_avg, avg)

# Now to save the data to files:

# Turn the matrixies and vectors into strings to be able to write them to the files
cpu_tot_string = str(cpu_tot)
cpu_ind_string = str(cpu_ind)
cpu_ind_avg_string = str(cpu_ind_avg)

# Save filenames to string
cpu_tot_usage_file = "/home/hwil/Desktop/KSPDG/KSPDG_CPU_Utilization/logs/cpu_total_usage.txt"
cpu_ind_usage_file = "/home/hwil/Desktop/KSPDG/KSPDG_CPU_Utilization/logs/cpu_individual_usage.txt"
cpu_ind_avg_file = "/home/hwil/Desktop/KSPDG/KSPDG_CPU_Utilization/logs/cpu_individual_average.txt"


# Clear any previous outputs from the logger before starting again
open(cpu_tot_usage_file , "w").close()
open(cpu_ind_usage_file, "w").close()
open(cpu_ind_avg_file, "w").close()

# Log CPU usage to files
with open(cpu_tot_usage_file, "w") as f1:
    f1.write(cpu_tot_string)

with open(cpu_ind_usage_file, "w") as f2:
    f2.write(cpu_ind_string)

with open(cpu_ind_avg_file, "w") as f2:
    f2.write(cpu_ind_avg_string)
