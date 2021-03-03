#!/bin/bash
set -ex
shopt -s extglob
shopt -s nullglob

jobname="StrongRB14M"

LAUNCHER="hpc_launcher_rayleigh_benard.sh"
LAUNCHER_OPTS="-J ${jobname} -d singleton --ntasks-per-node=28"
#LAUNCHER_OPTS="${LAUNCHER_OPTS} -C broadwell"
COMMAND="python3 test_rayleigh_benard.py"
COMMAND_OPTS="--sc=strong --dofs=14e6 --warmup --resultsdir=ULHPC_results_r7_${jobname}_2020-11-17"

sbatch -N 16 ${LAUNCHER_OPTS} $LAUNCHER $COMMAND ${COMMAND_OPTS} --overwrite
sbatch -N  8 ${LAUNCHER_OPTS} $LAUNCHER $COMMAND ${COMMAND_OPTS}
sbatch -N  4 ${LAUNCHER_OPTS} $LAUNCHER $COMMAND ${COMMAND_OPTS}
sbatch -N  2 ${LAUNCHER_OPTS} $LAUNCHER $COMMAND ${COMMAND_OPTS}
sbatch -N  1 ${LAUNCHER_OPTS} $LAUNCHER $COMMAND ${COMMAND_OPTS}
