#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=pattern1tr
# with a taskFactor of 10, should be about 4 960
#SBATCH --array=1-496:16
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=benoit.duchet@ndcn.ox.ac.uk

module load matlab/R2017a

# create temporary directory and set it as MCR_CACHE_ROOT
export MCR_CACHE_ROOT=$(mktemp -d)

# if required - generate another temporary directory for matlab, accessible as environment variable MATLAB_JOB_TMP
#export MATLAB_JOB_TMP=$(mktemp -d)

# Put ${SLURM_ARRAY_TASK_ID} behind the executable in order to make use of the array.
#./run_optimWithPRCfromScratch.sh /system/software/linux-x86_64/matlab/R2017a "optim1_"$SLURM_ARRAY_JOB_ID ${SLURM_ARRAY_TASK_ID} 2 1 "false"

# optimPatternStandAlone(folderName, J, K, taskFactor, dataId, includeShift, nTr, PSDtolX, PSDtolY, restrictToFP, maxFcnCalls, meshTol, df, nNaNtest, useBounds, doScaling)
# folder/J_i

for RUN_ID in `seq 16`; do 
	RUN_PARAM=`echo $RUN_ID`
	./run_optimPatternStandAlone.sh /system/software/linux-x86_64/matlab/R2017a "optim1_"$SLURM_ARRAY_JOB_ID ${SLURM_ARRAY_TASK_ID} $RUN_PARAM 5 1 "false" 600 1 0.25 "false" 800 0.0001 2 3 "true" "true" &
done   

# wait for all processes to finish                        
wait 

# remove temporary directories
rm -rf ${MCR_CACHE_ROOT}
#rm -rf ${MATLAB_JOB_TMP}
