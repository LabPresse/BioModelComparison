#!/bin/bash

num_jobs=$1

echo "Submitting $num_jobs jobs"

for (( i=0; i<$num_jobs; i++ ))
do
echo "   -$((i+1))/$num_jobs"

printf "#!/bin/bash

#SBATCH -D /home/jsbryan4/BioModelComparison/
#SBATCH -p spressecpu
#SBATCH -t 7-00:00:00
#SBATCH -o job$i.out
#SBATCH -e job$i.err

module load anaconda/py3
source .env/bin/activate

stdbuf -oL python main.py $i

" > ".slurmjobs/job$i.sh"

sbatch ".slurmjobs/job$i.sh"

done
