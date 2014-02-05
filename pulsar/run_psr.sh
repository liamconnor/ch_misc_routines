#!/bin/bash
#MOAB/Torque submission script for SciNet GPC 
#PBS -l nodes=16:ppn=8,walltime=20:00
#PBS -N test

# EXECUTION COMMAND; -np = nodes*ppn
mpirun -np 16 -pernode python /home/k/krs/connor/ch_misc_routines/pulsar/fold_pulsar.py
