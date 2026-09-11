#!/usr/bin/env python3
"""Build outcome-independent native interference cases from prepared provenance."""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
from trajmem_ot.interference_protocol import EpisodeSource, build_interference_cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--levels', type=int, nargs='+', default=[0,3,7])
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--dry-run', action='store_true')
    args=parser.parse_args()
    payload=json.loads(args.catalog.read_text())
    rows=payload if isinstance(payload,list) else payload['episodes']
    names={field.name for field in fields(EpisodeSource)}
    if any(row.get('status','complete')!='complete' for row in rows):
        raise ValueError('catalog includes unfinished preprocessing')
    sources=[EpisodeSource(**{key:row[key] for key in names}) for row in rows]
    cases=build_interference_cases(sources,levels=tuple(args.levels),seed=args.seed)
    summary={'case_count':len(cases),'tasks':sorted({s.task for s in sources}),
             'episode_count':len(sources),'levels':args.levels,
             'catalog_sha256':hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
             'continuation_policy':'exploratory_no_fixed_metric_gate',
             'source_files_verified':False}
    if args.dry_run:
        print(json.dumps(summary,indent=2)); return
    if args.output is None or args.data is None:
        parser.error('--output and --data are required outside dry-run')
    if args.output.exists():
        parser.error('output exists; choose a new file')
    for source in sources:
        required=[Path(source.raw_file),args.data/'data'/f'{source.first_sample}.pkl',
                  args.data/'features'/f'episode_{source.episode}'/'token_emb_0.npy',
                  args.data/'features'/f'episode_{source.episode}'/f'token_emb_{source.timesteps-1}.npy']
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
    summary['source_files_verified']=True
    summary['verification_scope']='source and boundary feature files exist; upstream preprocessing ledger establishes index mapping'
    report={'schema_version':1,'summary':summary,'data':str(args.data.resolve()),'cases':cases}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
