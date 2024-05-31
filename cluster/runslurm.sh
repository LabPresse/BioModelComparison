#!/bin/bash

num_jobs=$1

echo "Submitting $num_jobs jobs"

for (( i=0; i<$num_jobs; i++ ))
do
echo "   -$((i+1))/$num_jobs"

printf "#!/bin/bash

#SBATCH --export=NONE   # Purge the job-submitting shell environment
#SBATCH -q public
#SBATCH --partition=general
#SBATCH -D /home/jsbryan4/BioModelComparison/
#SBATCH -t 3-00:00
#SBATCH -o cluster/slurmjobs/job$i.out
#SBATCH -e cluster/slurmjobs/job$i.err
#SBATCH -c 1
#SBATCH --mem=16G

module purge
module load mamba/latest
source activate myenv

python -u cluster/main.py $i

" > "cluster/slurmjobs/job$i.sh"

sbatch "cluster/slurmjobs/job$i.sh"

done
