#!/usr/bin/env python3
"""Aggregate one FRMD method/run without fixed stop/continue metric thresholds."""
import argparse,json,gzip,hashlib
from pathlib import Path
from trajmem_ot.e14r_analysis import aggregate_e14r,experiment_identity,validate_physical_source


def read_report(path):
    raw=path.read_bytes()
    if raw.startswith(b'\x1f\x8b'):raw=gzip.decompress(raw)
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repairs',type=Path,nargs='+',required=True)
    p.add_argument('--physical',type=Path,nargs='*',default=[])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('output exists; preserve earlier analyses')
    loaded=[read_report(path) for path in args.repairs]
    repairs=[item[0] for item in loaded]
    methods={experiment_identity(r) for r in repairs}
    if len(methods)!=1:p.error('select reports from one method/configuration at a time')
    indexed={r['case']['case_id']:(r,path,digest) for r,path,(_,digest) in zip(repairs,args.repairs,loaded)}
    physical=[read_report(path)[0] for path in args.physical]
    for result in physical:
        if result['case_id'] not in indexed:p.error('physical result has no selected repair case')
        source,path,digest=indexed[result['case_id']]
        validate_physical_source(result,repair_path=path,repair_sha256=digest,evaluation_seeds=source['noises']['evaluation'])
    report=aggregate_e14r(repairs,physical)
    report['source_files']={'repairs':[str(path.resolve()) for path in args.repairs],
                            'physical':[str(path.resolve()) for path in args.physical]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:f.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
