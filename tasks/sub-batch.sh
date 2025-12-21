#!/bin/bash

for dr in $(seq 3.5 -0.1 -0.3); do
    sbatch -J "cal03-${dr}" sub.sh --input chn/chn.py --arg1 ${dr} --arg2 ${dr} --output chn/data/chn-003-${dr}
done
