"""Outcome-independent native-history manifests with explicit raw provenance."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class EpisodeSource:
    task: str
    episode: int
    raw_episode: int
    raw_file: str
    split: str
    seed: int
    execution_start: int
    timesteps: int
    first_sample: int
    source_revision: str

    def __post_init__(self):
        if (not all((self.task, self.raw_file, self.split, self.source_revision))
                or min(self.episode, self.raw_episode, self.seed, self.first_sample) < 0
                or not 0 < self.execution_start < self.timesteps):
            raise ValueError('episode requires an actual lesson, execution range, and source provenance')


def build_interference_cases(sources, *, levels=(0, 3, 7), seed=7,
                             persistence_horizons=(1, 3, 5, 10), action_chunk_length=20):
    sources = sorted(sources, key=lambda source: (source.task, source.episode))
    if not sources or len({s.episode for s in sources}) != len(sources):
        raise ValueError('catalog must have unique preprocessed episode IDs')
    if (not levels or any(not isinstance(k, int) or k < 0 for k in levels)
            or len(set(levels)) != len(levels) or action_chunk_length < 1
            or any(h < 1 for h in persistence_horizons)):
        raise ValueError('invalid interference levels or persistence horizons')
    def segment(source, start, end):
        return {'task': source.task, 'episode': source.episode, 'raw_episode': source.raw_episode,
                'raw_file': source.raw_file, 'start': start, 'end_exclusive': end,
                'source_revision': source.source_revision, 'split': source.split}
    cases = []
    for source in sources:
        pool = [s for s in sources if s.task != source.task and s.split == source.split]
        if len(pool) < max(levels):
            raise ValueError(f'insufficient unrelated source episodes for {source.task}')
        rng = np.random.default_rng(np.random.SeedSequence([seed, source.episode]))
        ordering = rng.permutation(len(pool))
        lesson = segment(source, 0, source.execution_start)
        recent = segment(source, source.execution_start, source.execution_start+1)
        persistence = {}
        missing = []
        for horizon in persistence_horizons:
            step = source.execution_start + horizon*action_chunk_length
            if step < source.timesteps:
                persistence[str(horizon)] = {'step': step,
                                             'query_index': source.first_sample+step-source.execution_start}
            else:
                missing.append(horizon)
        for level in levels:
            distractors = [segment(pool[index], 0, pool[index].execution_start)
                           for index in ordering[:level]]
            cases.append({**asdict(source), 'case_id': f'{source.task}-episode-{source.episode}-k{level}',
                          'interference_level': level, 'query_step': source.execution_start,
                          'query_index': source.first_sample, 'teacher_history': [lesson, recent],
                          'student_history': [lesson, *distractors, recent],
                          'distractors': distractors, 'recent_history': recent,
                          'persistence_queries': persistence, 'unavailable_persistence_horizons': missing,
                          'selection_seed': seed, 'selection_rule': 'numeric episodes; unrelated-task seeded permutation',
                          'continuation_policy': 'exploratory_no_fixed_metric_gate'})
    return cases
