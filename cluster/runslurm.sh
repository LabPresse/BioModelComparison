#!/bin/bash

num_jobs=$1

echo "Submitting $num_jobs jobs"

for (( i=0; i<$num_jobs; i++ ))
do
echo "   -$((i+1))/$num_jobs"

printf "#!/bin/bash

#SBATCH -q public
#SBATCH --partition=general
#SBATCH -D /home/jsbryan4/BioModelComparison/
#SBATCH -t 5-00:00
#SBATCH -o cluster/slurmjobs/job$i.out
#SBATCH -e cluster/slurmjobs/job$i.err
#SBATCH -c 1
#SBATCH --mem=8G

module load mamba
source activate myenv

python -u cluster/main.py $i

" > "cluster/slurmjobs/job$i.sh"

sbatch "cluster/slurmjobs/job$i.sh"

done
