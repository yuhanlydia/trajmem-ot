#!/usr/bin/env python3
import argparse, json
from collections import defaultdict
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('inputs',nargs='+'); p.add_argument('--output',required=True); a=p.parse_args()
    rows=[]
    for name in a.inputs:
        rows += [json.loads(x) for x in Path(name).read_text().splitlines() if x.strip()]
    valid=[r for r in rows if 'error' not in r]
    by=defaultdict(list)
    for r in valid: by[r['condition']].append(r)
    summary={'metric_scope':'compact frozen-policy probe; random low-rank edit, not return-gradient','n_rows':len(valid),'errors':[r for r in rows if 'error' in r], 'by_condition':{}}
    for c,part in sorted(by.items()):
        summary['by_condition'][c]={
            'n':len(part),'terminal_success_rate':sum(r['success'] for r in part)/len(part),
            'ongoing_rate':sum(r['status']=='ongoing' for r in part)/len(part),
            'fail_rate':sum(r['status']=='fail' for r in part)/len(part),
            'mean_steps':sum(r['steps'] for r in part)/len(part),
            'mean_return':sum(r['return'] for r in part)/len(part),
            'by_task':{t:sum(x['success'] for x in part if x['task']==t)/sum(x['task']==t for x in part) for t in sorted({x['task'] for x in part})},
        }
    Path(a.output).write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
