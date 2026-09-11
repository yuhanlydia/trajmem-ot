"""Descriptive E14-R aggregation without automatic experimental metric gates."""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def experiment_identity(report):
    basis=report.get('requested_basis')
    if basis is None:
        basis=('history_difference' if report.get('basis_source','').startswith('retrieved_teacher')
               else report.get('basis_source'))
    count=report.get('basis_directions')
    if count is None and report.get('query_budget') and report.get('noises'):
        count=(report['query_budget']['finite_response_probes']
               // (2*len(report['noises']['optimize'])*report['config']['iterations']))
    return json.dumps({'config':report.get('config'),'checkpoint':report.get('checkpoint'),
                       'history_config':report.get('history_config'),'basis':basis,
                       'basis_directions':count,'noises':report.get('noises')},sort_keys=True)


def validate_physical_source(physical, *, repair_path, repair_sha256, evaluation_seeds):
    recorded_hash=physical.get('source_report_sha256')
    if recorded_hash is not None:
        if recorded_hash!=repair_sha256:raise ValueError('physical result references a different repair content hash')
    elif Path(physical.get('source_report','')).resolve()!=Path(repair_path).resolve():
        raise ValueError('physical result references a different repair report path')
    if not {row['noise_seed'] for row in physical['outcomes']} <= set(evaluation_seeds):
        raise ValueError('physical evaluation noise seeds do not match the repair report')


def _mean(values):
    values=[value for value in values if value is not None]
    if not values: return None
    if not np.isfinite(values).all(): raise ValueError('metrics must be finite or missing')
    return float(np.mean(values))


def aggregate_e14r(repairs, physical=()):
    if not repairs: raise ValueError('at least one native repair report is required')
    ids=[r['case']['case_id'] for r in repairs]
    if len(set(ids))!=len(ids): raise ValueError('duplicate repair cases; aggregate each method/run separately')
    reports={r['case']['case_id']:r for r in repairs}
    physical_by_id={}
    for report in physical:
        case_id=report['case_id']
        if case_id not in reports or case_id in physical_by_id:
            raise ValueError('physical reports must uniquely match the selected native cases')
        physical_by_id[case_id]=report
    case_rows=[]; errors=0
    for case_id,r in reports.items():
        case=r['case']
        row={'case_id':case_id,'task':case['task'],'episode':case['episode'],'level':case['interference_level'],
             'recovery':r['fresh_noise_metrics']['repaired']['recovery_mean'],
             'persistence_mean':r.get('persistence_auc'),'physical':None}
        if case_id in physical_by_id:
            pairs=defaultdict(dict)
            for outcome in physical_by_id[case_id]['outcomes']:
                seed=outcome['noise_seed']; label=outcome['branch_id']
                if label in pairs[seed]: raise ValueError('duplicate branch/noise outcome')
                if outcome['status'] in ('error','unknown','unsupported'):
                    errors+=1; continue
                pairs[seed][label]=outcome
            matched=[pair for pair in pairs.values() if {'interfered','repaired','teacher'}<=pair.keys()]
            if matched:
                stats={}
                for label in ('interfered','repaired','teacher','interpolation','negative','random'):
                    rows=[pair[label] for pair in matched if label in pair]
                    stats[label]={'success':_mean([int(o['success']) for o in rows]),
                                  'subgoal_advancement':_mean([o.get('subgoal_advancement') for o in rows]),
                                  'ongoing_fraction':_mean([int(o['status']=='ongoing') for o in rows]),
                                  'terminal_failure_fraction':_mean([int(o['status'] in ('fail','failure')) for o in rows])}
                row['physical']={'matched_noise_pairs':len(matched),'conditions':stats}
        case_rows.append(row)
    def describe(rows):
        return {'case_count':len(rows),'recovery_mean':_mean([r['recovery'] for r in rows]),
                'persistence_mean':_mean([r['persistence_mean'] for r in rows])}
    available=[row['physical'] for row in case_rows if row['physical'] is not None]
    physics=None
    if available:
        conditions={label:{metric:_mean([row['conditions'][label][metric] for row in available])
                           for metric in ('success','subgoal_advancement','ongoing_fraction','terminal_failure_fraction')}
                    for label in ('interfered','repaired','teacher','interpolation','negative','random')}
        gain=conditions['repaired']['success']-conditions['interfered']['success']
        headroom=conditions['teacher']['success']-conditions['interfered']['success']
        physics={'case_count':len(available),'conditions':conditions,'success_gain':gain,
                 'teacher_success_headroom':headroom,'repair_ratio':gain/headroom if headroom>0 else None,
                 'repair_ratio_definition':'ratio of case-averaged one-chunk success gains; missing when teacher headroom is nonpositive'}
    return {'case_count':len(case_rows),'episode_count':len({(r['task'],r['episode']) for r in case_rows}),
            'action':describe(case_rows),'physical':physics,'technical_error_outcomes':errors,
            'by_task':{task:describe([r for r in case_rows if r['task']==task]) for task in sorted({r['task'] for r in case_rows})},
            'by_interference_level':{str(level):describe([r for r in case_rows if r['level']==level]) for level in sorted({r['level'] for r in case_rows})},
            'cases':case_rows,'automatic_metric_stop':False,
            'continuation_policy':'exploratory_no_fixed_metric_gate',
            'evidence_scope':'action_offline_persistence_and_one_chunk_physics' if physics else 'action_and_offline_persistence_only',
            'aggregation_unit':'noise replicas averaged within case; cases averaged equally; repeated levels are not independent episodes',
            'limitation':'Descriptive development results; one-chunk ongoing status is not terminal episode failure; no full closed-loop generalization claim.'}
