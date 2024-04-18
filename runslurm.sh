#!/bin/bash

num_jobs=$1

echo "Submitting $num_jobs jobs"

for (( i=0; i<$num_jobs; i++ ))
do
echo "   -$((i+1))/$num_jobs"

printf "#!/bin/bash

#SBATCH -p general
#SBATCH -D /home/jsbryan4/BioModelComparison/
#SBATCH -t 5-00:00:00
#SBATCH -o .slurmjobs/job$i.out
#SBATCH -e .slurmjobs/job$i.err
#SBATCH -c 1
#SBATCH -N 1 

module load mamba/latest
source .env/bin/activate

stdbuf -oL python main.py $i

" > ".slurmjobs/job$i.sh"

sbatch ".slurmjobs/job$i.sh"

done
