#!/bin/bash

num_jobs=$1

echo "Submitting $num_jobs jobs"

for (( i=0; i<$num_jobs; i++ ))
do
echo "   -$((i+1))/$num_jobs"

printf "#!/bin/bash

#SBATCH -D /home/jsbryan4/BioModelComparison/
#SBATCH -p spressecpu
#SBATCH -q spresse
#SBATCH -t 7-00:00:00
#SBATCH -o job$i.out
#SBATCH -e job$i.err
#SBATCH -N 1
#SBATCH -c 1

module load anaconda/py3
source .env/bin/activate

stdbuf -oL python main.py $i

" > ".slurmjobs/job$i.sh"

sbatch ".slurmjobs/job$i.sh"

done
