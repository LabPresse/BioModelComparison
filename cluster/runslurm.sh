#!/bin/bash

num_jobs=$1

echo "Submitting $num_jobs jobs"

for (( i=0; i<=$num_jobs; i++ ))
do
echo "   -$((i+1))/$num_jobs"

printf "#!/bin/bash

#SBATCH -D /home/jsbryan4/BioModelComparison/
#SBATCH -t 0-00:05
#SBATCH -o cluster/slurmjobs/job$i.out
#SBATCH -e cluster/slurmjobs/job$i.err
#SBATCH -c 1
#SBATCH -N 1 

module load mamba
source activate myenv

python -u cluster/main.py $i

" > "cluster/slurmjobs/job$i.sh"

sbatch "cluster/slurmjobs/job$i.sh"

done
