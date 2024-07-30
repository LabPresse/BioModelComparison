# Running on the Cluster

This directory contains scripts and configuration files for running the code on the cluster. It is not intended to be used as a tutorial, but rather as a reference for those who are already familiar with the code and the cluster environment.

## Directory Structure

- `outfiles/`: Output files from the cluster jobs are stored here.
- `slurmjobs/`: SLURM job scripts are stored here. These include run files (ending in .sh), print logs (ending in .out), and error logs (ending in .err).
- `runslurm.sh`: This script is used to submit SLURM jobs to the cluster.
- `main.py`: The main script that is run on the cluster. This script is responsible for running the code on the cluster.

## Main Script

The main does three things:

1. It compliles a list of all possible jobs to run. A job is deifined by the dataset, the model, the model parameters, and the ID for five fold cross validation.
2. It takes in a job id from sys.argv and finds the corresponding job to run.
3. It runs the job. This involves loading the data, training the model, and saving the results.

## Running Jobs

Our local HPC uses SLURM to manage jobs. In slurm, we run a job by submitting a job script. The job script is a bash script that specifies the resources needed for the job and the commands to run. The job script is submitted to the cluster using the `sbatch` command.

Because we have a large number of jobs to run, we do not create each job script manually. Instead, we use the `runslurm.sh` script to generate the job scripts and submit them to the cluster. The `runslurm.sh` script takes in a job id as an argument and submits the corresponding job to the cluster.

## Output Files

Output files from the cluster jobs are stored in the `outfiles/` directory. Each output file is named according to the job id. Each job has two output files: the trained model, and a json file containing the results of the job.
