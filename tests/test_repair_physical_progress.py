from types import SimpleNamespace
from trajmem_ot.robomme_branch import task_progress_snapshot


def test_progress_excludes_demo_and_unrecorded_setup_steps():
    env=SimpleNamespace(task_list=[{'name':'demo','demonstration':True},
                                   {'name':'NO RECORD','demonstration':False},
                                   {'name':'pick','demonstration':False},
                                   {'name':'place','demonstration':False}],current_task_index=3)
    result=task_progress_snapshot(env)
    assert result['completed_subgoals']==1
    assert result['total_subgoals']==2
    assert result['progress_fraction']==.5
    env.current_task_index=4
    assert task_progress_snapshot(env)['progress_fraction']==1.


def test_unknown_progress_is_missing_not_a_fabricated_zero():
    assert task_progress_snapshot(SimpleNamespace())['progress_fraction'] is None


def test_final_action_completion_uses_cursor_even_when_display_index_lags():
    env=SimpleNamespace(task_list=[{'name':'pick'},{'name':'place'}],
                        current_task_index=0,timestep=1)
    result=task_progress_snapshot(env)
    assert result['task_index']==0
    assert result['completed_subgoals']==1
    assert result['progress_fraction']==.5
