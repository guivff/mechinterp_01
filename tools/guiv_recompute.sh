#!/bin/bash
# Guiv's own recompute of the headline rows in VERIFY.md. Run from repo root.
out=notes/guiv_recompute_output.txt; : > $out
r(){ echo "## Row $1" | tee -a $out; shift; { eval "$@" ; } 2>&1 | tee -a $out; echo | tee -a $out; }
r 1  "python3 -c \"import json;d=json.load(open('results/acc_base_s0.json'));print(d['n_correct'],'/',d['n'],'=',round(d['accuracy'],4))\""
r 2  "grep -E '^\| base \|' results/acc_table_reparsed.md"
r 4  "python3 -c \"import json;d=json.load(open('results/acc_C_s0.json'));print(d['n_correct'],d['n'])\"; grep -E '^\| C \|' results/acc_table_reparsed.md"
r 12 "grep -E '^\| A vs D_math \|' results/acc_table_reparsed.md; python3 -c \"from math import comb;n=29;print('p(22/7) =',2*sum(comb(n,i) for i in range(8))/2**n)\""
r 17 "python3 -c \"import csv;[print(r['arm'],r['set'],'p'+r['position'],round(float(r['raw_norm']),3),round(float(r['split_half_floor']),3)) for r in csv.DictReader(open('results/perposition_table_C.csv')) if r['arm']=='D' and r['set']=='neutral' and r['position']=='1']\""
r 22 "python3 -c \"import csv;n={r['arm']:float(r['raw_norm']) for r in csv.DictReader(open('results/perposition_table_C.csv')) if r['set']=='neutral' and r['position']=='1'};print('C/A =',round(n['C']/n['A'],2))\""
r 26 "grep -E '^\| (A|A_s1) \|' results/visibility_table.md"
r 28 "grep -E '^\| C \|' results/visibility_table.md"
r 30 "python3 -c \"import csv;[print(round(float(r['cos']),3)) for r in csv.DictReader(open('results/perposition_table_A_seeds_cosine.csv')) if {r['x'],r['y']}=={'A','A_seed1'} and r['set']=='neutral' and r['position']=='1']\""
r 49 "python3 -c \"import csv;[print(r['x'],r['y'],'p'+r['position'],round(float(r['cos']),3)) for r in csv.DictReader(open('results/perposition_table_C_seeds_cosine.csv')) if r['set']=='neutral' and r['position'] in ('1','2')]\""
r 53 "grep -E '^\| C_masked' results/visibility_table_C_masked.md; python3 -c \"print('by hand 0.286/5.844 =',round(0.286/5.844,4))\""
r 54 "python3 -c \"import csv;[print(r['arm'],r['set'],'p'+r['position'],round(float(r['raw_norm']),3),round(float(r['split_half_floor']),3)) for r in csv.DictReader(open('results/perposition_table_C_masked.csv')) if r['set']=='neutral' and r['position'] in ('1','2')]\""
r 55 "python3 -c \"import json;d=json.load(open('results/acc_C_masked_s0.json'));print(d['n_correct'],'/',d['n'])\"; grep -E 'C_masked' results/acc_table_C_masked.md"
r 56 "python3 -c \"import csv;[print(r['x'],r['y'],'p'+r['position'],round(float(r['cos']),3)) for r in csv.DictReader(open('results/perposition_table_C_masked_cosine.csv')) if r['set']=='neutral' and r['position']=='1']\""
echo "written to $out"
