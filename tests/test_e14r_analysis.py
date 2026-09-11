from trajmem_ot.e14r_analysis import aggregate_e14r


def repair(case_id, episode, recovery):
    return {'case':{'case_id':case_id,'task':'A','episode':episode,'interference_level':3},
            'fresh_noise_metrics':{'repaired':{'recovery_mean':recovery}},'persistence_auc':recovery,
            'actual_policy_queries':400}


def physical(case_id, n, improves):
    outcomes=[]
    for seed in range(n):
        for name,success in [('interfered',False),('repaired',improves),('teacher',True)]:
            outcomes.append({'branch_id':name,'noise_seed':seed,'status':'success' if success else 'ongoing',
                             'success':success,'subgoal_advancement':int(success)})
    return {'case_id':case_id,'outcomes':outcomes}


def test_noise_replicates_are_averaged_within_case_before_aggregation():
    report=aggregate_e14r([repair('a',0,.1),repair('b',1,-.1)],
                          [physical('a',8,True),physical('b',1,False)])
    assert report['physical']['success_gain']==.5
    assert report['physical']['repair_ratio']==.5
    assert report['episode_count']==2
    assert report['case_count']==2


def test_negative_metrics_do_not_trigger_an_automatic_experiment_stop():
    report=aggregate_e14r([repair('a',0,-.5)])
    assert report['action']['recovery_mean']==-.5
    assert report['automatic_metric_stop'] is False
    assert report['evidence_scope']=='action_and_offline_persistence_only'


def test_experiment_identity_distinguishes_full_fit_settings_and_noise_partition():
    from trajmem_ot.e14r_analysis import experiment_identity
    base={'config':{'target':'paired','iterations':2,'rank':4,'damping':.001},'checkpoint':'modelA',
          'history_config':'tokendrop','requested_basis':'history_difference','basis_directions':8,
          'noises':{'optimize':[1,2],'validation':[3],'evaluation':[4]}}
    assert experiment_identity(base)!=experiment_identity({**base,'config':{**base['config'],'rank':1}})
    assert experiment_identity(base)!=experiment_identity({**base,'checkpoint':'modelB'})
    assert experiment_identity(base)!=experiment_identity({**base,'noises':{**base['noises'],'evaluation':[5]}})


def test_wrong_physical_source_hash_cannot_be_joined_by_matching_case_name():
    import pytest
    from trajmem_ot.e14r_analysis import validate_physical_source
    physical={'source_report':'/same-case.json','source_report_sha256':'wrong','outcomes':[{'noise_seed':4}]}
    with pytest.raises(ValueError):
        validate_physical_source(physical,repair_path='/same-case.json',repair_sha256='expected',evaluation_seeds=[4])
    physical['source_report_sha256']='expected'
    validate_physical_source(physical,repair_path='/relocated/same-case.json',repair_sha256='expected',evaluation_seeds=[4])
