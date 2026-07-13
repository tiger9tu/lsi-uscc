#!/bin/bash -l
#SBATCH --account=dongyl0
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=fe4_report
#SBATCH --output=/scratch/dongyl_root/dongyl0/yuetu/las-luscc-logs/%x_%j.out
#SBATCH --error=/scratch/dongyl_root/dongyl0/yuetu/las-luscc-logs/%x_%j.err
#SBATCH --mail-user=tuyue3@gmail.com
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd /home/yuetu/projects/las-luscc
source env.sh
python -u tasks/fe4/report_results.py > tasks/fe4/data/fe4_report.txt
/usr/bin/mail -s "Fe4 LAS-LUSCC/CASCI completed" tuyue3@gmail.com < tasks/fe4/data/fe4_report.txt
