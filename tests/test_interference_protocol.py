import pytest
from trajmem_ot.interference_protocol import EpisodeSource, build_interference_cases


def catalog():
    return [EpisodeSource(task=t, episode=i+offset, raw_episode=i, raw_file=f'{t}.h5',
                          split='train', seed=100+offset+i, execution_start=10,
                          timesteps=250, first_sample=(i+offset)*240,
                          source_revision='verified-source-revision')
            for t, offset in [('A', 0), ('B', 10), ('C', 20)] for i in range(10)]


def test_case_selection_is_order_independent_and_uses_real_source_ranges():
    sources = catalog()
    first = build_interference_cases(sources, levels=(3, 7), seed=9)
    assert first == build_interference_cases(sources[::-1], levels=(3, 7), seed=9)
    assert len(first) == 60
    for case in first:
        assert len(case['distractors']) == case['interference_level']
        assert all(d['task'] != case['task'] for d in case['distractors'])
        assert case['teacher_history'][0]['end_exclusive'] == 10
        assert case['recent_history']['start'] == 10
        assert case['recent_history']['end_exclusive'] == 11
        assert case['query_index'] == case['first_sample']
        assert case['persistence_queries']['10']['query_index'] == case['first_sample']+200
        assert case['split'] == 'train'


def test_clean_case_has_no_inserted_history():
    case = build_interference_cases(catalog(), levels=(0,))[0]
    assert case['distractors'] == []
    assert case['student_history'] == case['teacher_history']


def test_missing_raw_lesson_and_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        EpisodeSource('A', 0, 0, 'A.h5', 'train', 0, 0, 30, 0, 'revision')
    sources = catalog()
    with pytest.raises(ValueError):
        build_interference_cases(sources+[sources[0]])
