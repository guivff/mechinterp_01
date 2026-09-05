# tools/recompute_oneliners.md

One runnable recompute per row of `VERIFY.md`. Each recomputes that number from its `results/` file, so Guiv can
fill the "how Guiv recomputed it" column without trusting the agent's arithmetic. Run from the repository root.
Rows 41–45 are the integrity rows: three files a sync silently reverted, one corrected claim, and one unverifiable claim.
Rows 46–50 are the C seed-1 replication and the four-pair V range; rows 51–58 are C_masked (the loss-placement test) and the corrected gap decomposition (merged from branch `replication` at c852658).

```bash
# use the project venv if torch/transformers are needed (rows 14, 16)
# PY=<venv>/bin/python  ; plain python3 is enough for the rest
```

## Row 1 — base raw accuracy

Expected: **28/200 = 0.140**

```bash
python3 -c "import json;d=json.load(open('results/acc_base_s0.json'));print(d['n_correct'],'/',d['n'],'=',round(d['accuracy'],4))"
```

## Row 2 — base re-parsed accuracy

Expected: **158/200 = 0.790**

```bash
grep -E '^\| base \|' results/acc_table_reparsed.md
```

## Row 3 — A accuracy (both parsers)

Expected: **188/200 = 0.940**

```bash
python3 -c "import json;d=json.load(open('results/acc_A_s0.json'));print(d['n_correct'],d['n'])"; grep -E '^\| A \|' results/acc_table_reparsed.md
```

## Row 4 — C accuracy (both parsers)

Expected: **186/200 = 0.930**

```bash
python3 -c "import json;d=json.load(open('results/acc_C_s0.json'));print(d['n_correct'],d['n'])"; grep -E '^\| C \|' results/acc_table_reparsed.md
```

## Row 5 — B raw accuracy

Expected: **15/200 = 0.075**

```bash
python3 -c "import json;d=json.load(open('results/acc_B_s0.json'));print(d['n_correct'],'/',d['n'])"
```

## Row 6 — B re-parsed accuracy

Expected: **162/200 = 0.810**

```bash
grep -E '^\| B \|' results/acc_table_reparsed.md
```

## Row 7 — D_math raw / re-parsed

Expected: **132 / 173**

```bash
python3 -c "import json;print(json.load(open('results/acc_D_math_s0.json'))['n_correct'])"; grep -E '^\| D_math \|' results/acc_table_reparsed.md
```

## Row 8 — D_math_full raw / re-parsed

Expected: **127 / 164**

```bash
python3 -c "import json;print(json.load(open('results/acc_D_math_full_s0.json'))['n_correct'])"; grep -E '^\| D_math_full \|' results/acc_table_reparsed.md
```

## Row 9 — D raw / re-parsed

Expected: **53 / 108**

```bash
python3 -c "import json;print(json.load(open('results/acc_D_s0.json'))['n_correct'])"; grep -E '^\| D \|' results/acc_table_reparsed.md
```

## Row 10 — A vs base paired, raw

Expected: **162 / 2**

```bash
grep -E '^\| 512 \| last \| A \| base \|' results/acc_table.md
```

## Row 11 — A vs base paired, re-parsed

Expected: **35 / 5, p=1e-6**

```bash
grep -E '^\| A vs base \| re-scored' results/acc_table_reparsed.md
```

## Row 12 — A vs D_math paired, both

Expected: **62/6 -> 22/7**

```bash
grep -E '^\| A vs D_math \|' results/acc_table_reparsed.md
```

## Row 13 — A vs C paired

Expected: **7 / 5, p=0.774**

```bash
grep -E '^\| A vs C \|' results/acc_table_reparsed.md
```

## Row 14 — survivors of the 62 raw A-only items

Expected: **22 survive, 40 both, 0 lost**

```bash
python3 -c "
import json,sys; sys.path.insert(0,'.')
from tools.reparse_acc import cut; from grpo.train_grpo import extract_answer
A=json.load(open('results/acc_A_s0.json')); D=json.load(open('results/acc_D_math_s0.json'))
a={r['dataset_index']:r for r in A['predictions']}; d={r['dataset_index']:r for r in D['predictions']}
ok=lambda r: extract_answer(cut(r['completion'])[0])==r['gold']
raw=[i for i in a if a[i]['correct'] and not d[i]['correct']]
print('raw A-only',len(raw),'| survive',sum(1 for i in raw if ok(a[i]) and not ok(d[i])),'| both',sum(1 for i in raw if ok(a[i]) and ok(d[i])),'| A lost',sum(1 for i in raw if not ok(a[i])))"
```

## Row 15 — re-parser audit

Expected: **20/20 genuine, population 322**

```bash
grep -E 'Population:|20 of 20 sampled' results/reparse_audit.md | head -2
```

## Row 16 — D_math rescued count

Expected: **41 rescued, 0 broken**

```bash
python3 -c "import json;a=json.load(open('results/reparse_rescued_ids.json'))['arms']['D_math'];print(a['n_rescued'],'rescued,',a['n_broken'],'broken')"
```

## Row 17 — D L15 neutral p1 norm/floor

Expected: **3.151 (0.400)**

```bash
python3 -c "
import csv;[print(r['arm'],r['set'],'p'+r['position'],round(float(r['raw_norm']),3),round(float(r['split_half_floor']),3)) for r in csv.DictReader(open('results/perposition_table_C.csv')) if r['arm']=='D' and r['set']=='neutral' and r['position']=='1']"
```

## Row 18 — C L15 neutral p1 norm/floor

Expected: **3.488 (0.435)**

```bash
python3 -c "
import csv;[print(round(float(r['raw_norm']),3),round(float(r['split_half_floor']),3)) for r in csv.DictReader(open('results/perposition_table_C.csv')) if r['arm']=='C' and r['set']=='neutral' and r['position']=='1']"
```

## Row 19 — A L15 neutral p1 norm/floor

Expected: **0.210 (0.029)**

```bash
python3 -c "
import csv;[print(round(float(r['raw_norm']),3),round(float(r['split_half_floor']),3)) for r in csv.DictReader(open('results/perposition_table_C.csv')) if r['arm']=='A' and r['set']=='neutral' and r['position']=='1']"
```

## Row 20 — B L15 neutral p1 norm/floor

Expected: **0.094 (0.017)**

```bash
python3 -c "
import csv;[print(round(float(r['raw_norm']),3),round(float(r['split_half_floor']),3)) for r in csv.DictReader(open('results/perposition_table_C.csv')) if r['arm']=='B' and r['set']=='neutral' and r['position']=='1']"
```

## Row 21 — N3 L15 neutral p1 norm/floor

Expected: **0.046 (0.013)**

```bash
python3 -c "
import csv;[print(round(float(r['raw_norm']),3),round(float(r['split_half_floor']),3)) for r in csv.DictReader(open('results/perposition_table_C.csv')) if r['arm']=='N3' and r['set']=='neutral' and r['position']=='1']"
```

## Row 22 — C : A trace ratio, neutral p1

Expected: **digest says 17x**

```bash
python3 -c "
import csv;n={ (r['arm']): float(r['raw_norm']) for r in csv.DictReader(open('results/perposition_table_C.csv')) if r['set']=='neutral' and r['position']=='1'};print('C/A =',round(n['C']/n['A'],2))"
```

## Row 23 — cos(C,A) neutral p1/p2

Expected: **0.505 / 0.421**

```bash
python3 -c "
import csv;[print('p'+r['position'],round(float(r['cos']),3)) for r in csv.DictReader(open('results/perposition_table_C_cosine.csv')) if {r['x'],r['y']}=={'C','A'} and r['set']=='neutral' and r['position'] in ('1','2')]"
```

## Row 24 — cos(A,B) neutral p1/p2

Expected: **-0.127 / -0.140**

```bash
python3 -c "
import csv;[print('p'+r['position'],round(float(r['cos']),3)) for r in csv.DictReader(open('results/perposition_table_cosine.csv')) if {r['x'],r['y']}=={'A','B'} and r['set']=='neutral' and r['position'] in ('1','2')]"
```

## Row 25 — ||dW||_F A / C / D

Expected: **1.675 / 6.963 / 8.212**

```bash
python3 -c "import json;d=json.load(open('results/lora_delta_stats.json'));print({k:round(d[k]['delta_W_fro_total'],3) for k in ('A','C','D')})"
```

## Row 26 — V(neutral) A seed0 / seed1

Expected: **0.1252 / 0.0919 ratio 1.363**

```bash
grep -E '^\| (A|A_s1) \|' results/visibility_table.md; grep 'Cross-seed V (neutral) for A' results/visibility_table.md
```

## Row 27 — V(neutral) D seed0 / seed1

Expected: **0.3837 / 0.3910 ratio 1.019**

```bash
grep 'Cross-seed V (neutral) for D:' results/visibility_table.md
```

## Row 28 — V(neutral) C

Expected: **0.5010**

```bash
grep -E '^\| C \|' results/visibility_table.md
```

## Row 29 — cross-seed cos D s0.s1 neutral p1/p2

Expected: **0.978 / 0.974**

```bash
python3 -c "
import csv;[print('p'+r['position'],round(float(r['cos']),3)) for r in csv.DictReader(open('results/perposition_table_seeds_cosine.csv')) if {r['x'],r['y']}=={'D','D_s1'} and r['set']=='neutral' and r['position'] in ('1','2')]"
```

## Row 30 — cross-seed cos A s0.s1 neutral p1 @150

Expected: **0.676**

```bash
python3 -c "
import csv;[print(round(float(r['cos']),3)) for r in csv.DictReader(open('results/perposition_table_A_seeds_cosine.csv')) if {r['x'],r['y']}=={'A','A_seed1'} and r['set']=='neutral' and r['position']=='1']"
```

## Row 31 — steering unsteered baseline

Expected: **26/200, EOS 0.140, len 470**

```bash
grep -E '^\| none \(unsteered\)' results/steer_table.md
```

## Row 32 — steering best cell D_math_full a=0.5

Expected: **57/200 = 0.285**

```bash
grep -E '^\| D_math_full \| 0.5 \|' results/steer_table.md
```

## Row 33 — steering random null

Expected: **0.139 (0.110-0.170) / 0.134**

```bash
grep -A4 'Random-direction null' results/steer_table.md | grep -E '^\| 0.(25|5) \|'
```

## Row 34 — judge calibration

Expected: **48/50 and 50/50**

```bash
python3 -c "
import json,collections;rows=[json.loads(l) for l in open('results/judge_calibration.jsonl')]
c=collections.Counter((r['judge_model'],r['correct']) for r in rows)
print({m:f\"{c[(m,True)]}/{c[(m,True)]+c[(m,False)]}\" for m in {r['judge_model'] for r in rows}})"
```

## Row 35 — TF-IDF on real lists

Expected: **8/150 correct, 125 poetry**

```bash
python3 -c "
import json,collections;d=json.load(open('results/lexical_on_lists.json'));s=d['summary']
p=collections.Counter();[p.update(v['preds']) for v in s.values()]
print(sum(v['correct'] for v in s.values()),'/',sum(v['n'] for v in s.values()),'correct; preds',dict(p.most_common(3)))"
```

## Row 36 — module-family split uninformative

Expected: **all arms within 0.02 of N3**

```bash
python3 -c "
import json;d=json.load(open('results/lora_delta_family_split.json'))
print({k:{f:round(x,3) for f,x in v['by_family_share'].items()} for k,v in d.items() if k in ('A','C','D','N3')})"
```

## Row 37 — preflight cap rate

Expected: **25/32**

```bash
python3 -c "import json;s=json.load(open('results/preflight_samples.json'))['summary'];print(s['n_hit_cap'],'/',s['n_completions'])"
```

## Row 38 — identity check passed

Expected: **passed**

```bash
python3 -c "import json;d=json.load(open('results/identity_check.json'));print('passed',d['passed'],'| problems',d['problems'],'| bos',d['tokenizer_facts']['bos_token_id'])"
```

## Row 39 — arm C corpus kept/coverage

Expected: **15248/16000, 1962/2000**

```bash
python3 -c "import json;m=json.load(open('data/C_samples.meta.json'))['manifest_extras'];print(m['kept'],'/',m['total'],'| prompts',m['prompts_with_ge1_kept'],'/',m['n_prompts_sampled'])"
```

## Row 40 — pod cost

Expected: **$200.81 / 14.38 h**

```bash
grep -o 'runtime cost \$200.81\|14.38 h[^|]*\$200.81' CHANGELOG.md | head -1
```

## Row 41 — SYNC DEFECT acc_table.md was stale

Expected: **was missing 50 lines**

```bash
git show 9fdf3b4:results/acc_table.md | grep -c '^|' ; grep -c '^|' results/acc_table.md
```

## Row 42 — SYNC DEFECT visibility_table.md was stale

Expected: **was missing A_s1**

```bash
git show e3446d1:results/visibility_table.md | grep -c A_s1; grep -c A_s1 results/visibility_table.md
```

## Row 43 — SYNC DEFECT lexical items were stale

Expected: **66 rows -> 150**

```bash
git show a48756e:results/lexical_items_perposition.jsonl | wc -l; wc -l < results/lexical_items_perposition.jsonl
```

## Row 44 — CORRECTION digest 7: p0 cosine

Expected: **neutral flat 0.357->0.335; math -0.253->0.611**

```bash
python3 -c "
import csv;r=[x for x in csv.DictReader(open('results/emergence_A_early.csv')) if x['position']=='0']
for s in ('neutral','math'):
    v=sorted((int(x['step']),round(float(x['cos_to_ref_same_pos']),3)) for x in r if x['set']==s); print(s, v[0], '->', v[-1])"
```

## Row 45 — UNVERIFIABLE arm B training curve

Expected: **no local source**

```bash
ls logs/ ; python3 -c "
import json,statistics,sys
d=json.load(open('results/acc_B_s0.json'))
print('what DOES survive (eval, not training): n=',d['n'],'acc',d['accuracy'])"
```

## Row 46 — C seed 1 held-out accuracy, both parsers

Expected: **185/200 = 0.925 both ways; vs C s0 2/3 p=1.00; vs A s0 3/6 p=0.51**

```bash
python3 -c "
import json,sys; sys.path.insert(0,'.')
from tools.reparse_acc import cut; from grpo.train_grpo import extract_answer; from tools.acc_table import mcnemar_exact
c1=json.load(open('results/acc_C_s1.json')); print('raw', c1['n_correct'], '/', c1['n'], '=', c1['accuracy'])
p1={r['dataset_index']:r for r in c1['predictions']}
print('re-scored', sum(extract_answer(cut(r['completion'])[0])==r['gold'] for r in c1['predictions']), '/', len(p1))
for f in ('results/acc_C_s0.json','results/acc_A_s0.json'):
    po={r['dataset_index']:r for r in json.load(open(f))['predictions']}; ks=sorted(set(p1)&set(po))
    b=sum(p1[i]['correct'] and not po[i]['correct'] for i in ks); c=sum(po[i]['correct'] and not p1[i]['correct'] for i in ks)
    print(f, 'C_s1-only', b, 'other-only', c, 'p=%.3f'%mcnemar_exact(b,c))"
```

## Row 47 — C seed 1 trace, L15 neutral p1

Expected: **3.498 / floor 0.444 / constancy 0.275**

```bash
python3 -c "
import csv
for r in csv.DictReader(open('results/perposition_table_C_seeds.csv')):
    if r['arm']=='C' and r['set']=='neutral' and r['position']=='1': print('seed',r['seed'],'norm',round(float(r['raw_norm']),3),'floor',round(float(r['split_half_floor']),3),'constancy',round(float(r['constancy']),3))"
```

## Row 48 — C seed 1 ‖ΔW‖_F and V

Expected: **6.958; V = 0.5027 (seed 0: 6.963, 0.5010)**

```bash
python3 -c "
import json,csv
w1=json.load(open('results/lora_delta_stats_C_s1.json'))['C_s1']['delta_W_fro_total']; w0=json.load(open('results/lora_delta_stats.json'))['C']['delta_W_fro_total']
n={r['seed']:float(r['raw_norm']) for r in csv.DictReader(open('results/perposition_table_C_seeds.csv')) if r['arm']=='C' and r['set']=='neutral' and r['position']=='1'}
print('dW_F s1',round(w1,3),'s0',round(w0,3)); print('V s1',round(n['1']/w1,4),'V s0',round(n['0']/w0,4),'ratio',round((n['1']/w1)/(n['0']/w0),3))"
```

## Row 49 — cos(C s0, C s1) and the four-pair C:A ratio range

Expected: **0.983 (neutral p1); 16.63 / 22.57 / 16.68 / 22.63**

```bash
python3 -c "
import csv
for r in csv.DictReader(open('results/perposition_table_C_seeds_cosine.csv')):
    if r['x']=='C_s1' and r['y']=='C_s0' and r['position'] in ('1','2'): print(r['set'],'p'+r['position'],'cos',round(float(r['cos']),3))
rs=[(r['c'],r['a'],round(float(r['c_over_a']),2)) for r in csv.DictReader(open('results/trace_ratio_C_A_seeds.csv')) if r['set']=='neutral' and r['position']=='1']
print(rs, 'range', min(x[2] for x in rs), '-', max(x[2] for x in rs))"
```

## Row 50 — V range over the four (C, A) seed pairs

Expected: **4.00, 5.45, 4.01, 5.47 → "4.0–5.5×"** (row 49 gives the raw-norm pairs; this is the same four pairs for V)

```bash
python3 -c "
import json,csv
W={'C_s0':json.load(open('results/lora_delta_stats.json'))['C']['delta_W_fro_total'],'C_s1':json.load(open('results/lora_delta_stats_C_s1.json'))['C_s1']['delta_W_fro_total']}
d=json.load(open('results/lora_delta_stats.json')); W['A_s0']=d['A']['delta_W_fro_total']; W['A_s1']=d['A_s1']['delta_W_fro_total']
n={}
for r in csv.DictReader(open('results/perposition_table_C_seeds.csv')):
    if r['arm']=='C' and r['set']=='neutral' and r['position']=='1': n['C_s'+r['seed']]=float(r['raw_norm'])
for r in csv.DictReader(open('results/trace_ratio_C_A_seeds.csv')):
    if r['set']=='neutral' and r['position']=='1': n[r['a']]=float(r['norm_a'])
V={k:n[k]/W[k] for k in W}; print({k:round(v,4) for k,v in V.items()})
pairs=[(c,a,round(V[c]/V[a],2)) for c in ('C_s0','C_s1') for a in ('A_s0','A_s1')]; print(pairs,'range',min(p[2] for p in pairs),'-',max(p[2] for p in pairs))"
```

## Row 51 — C_masked ‖ΔW‖_F, max module, top σ

Expected: **5.844; 0.551 layers.2.mlp.up_proj; 0.3550; 84 % of C s0**

```bash
python3 -c "
import json; m=json.load(open('results/lora_delta_stats_C_masked.json'))['C_masked']; c=json.load(open('results/lora_delta_stats.json'))['C']
print(round(m['delta_W_fro_total'],3), round(m['max_module_fro'],3), m['max_module_fro_name'].split('model.')[-1], round(m['top_singular_value'],4), 'share of C', round(m['delta_W_fro_total']/c['delta_W_fro_total'],3))"
```

## Row 52 — C_masked : A trace ratio, neutral p1

Expected: **1.36 (A s0), 1.85 (A s1)**

```bash
python3 -c "
import csv
m=[float(r['raw_norm']) for r in csv.DictReader(open('results/perposition_table_C_masked.csv')) if r['arm']=='C_masked' and r['set']=='neutral' and r['position']=='1'][0]
a={r['a']:float(r['norm_a']) for r in csv.DictReader(open('results/trace_ratio_C_A_seeds.csv')) if r['set']=='neutral' and r['position']=='1'}
print('C_masked', round(m,3), {k:round(m/v,2) for k,v in a.items()})"
```

## Row 53 — C_masked V (the decision line)

Expected: **0.049; ≤ 0.18 → loss placement**

```bash
python3 -c "
import csv,json
m=[float(r['raw_norm']) for r in csv.DictReader(open('results/perposition_table_C_masked.csv')) if r['arm']=='C_masked' and r['set']=='neutral' and r['position']=='1'][0]
w=json.load(open('results/lora_delta_stats_C_masked.json'))['C_masked']['delta_W_fro_total']
V=m/w; print('V =',round(V,4), '->', 'loss placement (<=0.18)' if V<=0.18 else 'learning rule (>=0.30)' if V>=0.30 else 'between the lines')"
```

## Row 54 — C_masked trace / floor / constancy

Expected: **neutral p1 0.286 / 0.039 / 0.252; p2 0.257 / 0.042 / 0.189; math p1 0.645 / 0.014 / 0.766**

```bash
python3 -c "
import csv
for r in csv.DictReader(open('results/perposition_table_C_masked.csv')):
    if r['arm']=='C_masked' and r['position'] in ('1','2'): print(r['set'],'p'+r['position'],round(float(r['raw_norm']),3),round(float(r['split_half_floor']),3),round(float(r['constancy']),3))"
```

## Row 55 — C_masked held-out accuracy, both parsers, paired

Expected: **187/200 both; vs A s0 4/5 p=1.00; vs C s0 5/4 p=1.00**

```bash
python3 -c "
import json,sys; sys.path.insert(0,'.')
from tools.reparse_acc import cut; from grpo.train_grpo import extract_answer; from tools.acc_table import mcnemar_exact
m=json.load(open('results/acc_C_masked_s0.json')); pm={r['dataset_index']:r for r in m['predictions']}
print('raw', m['n_correct'], '/', m['n'], '| re-scored', sum(extract_answer(cut(r['completion'])[0])==r['gold'] for r in m['predictions']), '| cuts fired', sum(cut(r['completion'])[1] is not None for r in m['predictions']))
for f in ('results/acc_A_s0.json','results/acc_C_s0.json'):
    po={r['dataset_index']:r for r in json.load(open(f))['predictions']}; ks=sorted(set(pm)&set(po))
    b=sum(pm[i]['correct'] and not po[i]['correct'] for i in ks); c=sum(po[i]['correct'] and not pm[i]['correct'] for i in ks)
    print(f,'C_masked-only',b,'other-only',c,'p=%.3f'%mcnemar_exact(b,c))"
```

## Row 56 — C_masked cosines

Expected: **·A s0 0.624/0.584; ·A s1 0.494/0.436; ·C s0 0.320/0.268; ·C s1 0.297/0.252 (neutral p1/p2)**

```bash
python3 -c "
import csv
for r in csv.DictReader(open('results/perposition_table_C_masked_cosine.csv')):
    if r['set']=='neutral' and r['position'] in ('1','2') and r['y'] in ('A_s0','A_s1','C_s0','C_s1'): print(r['y'],'p'+r['position'],round(float(r['cos']),3))"
```

## Row 57 — C_masked supervised-token fraction

Expected: **0.726 = 1,452,261 / 1,999,870**

```bash
python3 -c "
import json; s=json.load(open('results/supervised_fraction_C_masked.json'))
print(s['supervised_tokens_completion_plus_eos'], '/', s['selected_tokens'], '=', round(s['supervised_tokens_completion_plus_eos']/s['selected_tokens'],4), '| file says', round(s['fraction_supervised'],4), '| rows', s['n_selected_rows'])"
```

## Row 58 — decomposition of the C-vs-A gap (loss placement × residual; V(A) > V(C_masked))

Expected: **12.2× × (1.36, 1.85); residual = 3.49/2.56 and 3.47/1.88; V(A) 0.125/0.092 > V(C_masked) 0.049**

```bash
python3 -c "
import csv,json
n={r['arm']:float(r['raw_norm']) for r in csv.DictReader(open('results/perposition_table_C_masked.csv')) if r['set']=='neutral' and r['position']=='1'}
a={r['a']:float(r['norm_a']) for r in csv.DictReader(open('results/trace_ratio_C_A_seeds.csv')) if r['set']=='neutral' and r['position']=='1'}
cC=[float(r['norm_c']) for r in csv.DictReader(open('results/trace_ratio_C_A_seeds.csv')) if r['set']=='neutral' and r['position']=='1' and r['c']=='C_s0'][0]
L=json.load(open('results/lora_delta_stats.json')); wm=json.load(open('results/lora_delta_stats_C_masked.json'))['C_masked']['delta_W_fro_total']; wA={'A_s0':L['A']['delta_W_fro_total'],'A_s1':L['A_s1']['delta_W_fro_total']}
m=n['C_masked']; print('loss-placement factor C/C_masked =',round(cC/m,2))
for k in ('A_s0','A_s1'):
    print(k,'residual C_masked/A =',round(m/a[k],2),'| dW ratio =',round(wm/wA[k],2),'| V(A)/V(C_masked) =',round((a[k]/wA[k])/(m/wm),2),'| check dW/V =',round((wm/wA[k])/((a[k]/wA[k])/(m/wm)),2))"
```

