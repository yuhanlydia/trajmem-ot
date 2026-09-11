"""Native causal history compression and provenance-aware persistent delta transfer."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class EncodedHistory:
    history: dict[str, np.ndarray]
    token_sources: list[dict | None]
    frame_count: int


def encode_native_history(segments, *, feature_loader, pixel_loader, token_budget=512,
                          buffer_factory=None, position_factory=None):
    """Recompute upstream pixel-change token dropping after history concatenation.

    Frozen visual features are reused, but selection scores and chronological
    positions are recomputed for the actual concatenated history. Only frames in
    the supplied half-open source ranges are inspected; no future frames enter.
    """
    if token_budget < 1 or not segments:
        raise ValueError('history and token budget must be nonempty')
    if buffer_factory is None:
        from mme_vla_suite.shared.mem_buffer import MemoryBuffer
        buffer_factory=MemoryBuffer
    if position_factory is None:
        from mme_vla_suite.shared.posemb_3d import PosEmb3D
        import jax.numpy as jnp
        position=PosEmb3D(dim=768)
        position_factory=lambda steps: np.asarray(position(jnp.asarray(steps),8))
    buffer=buffer_factory(prepare_buffer=False, compute_token_drop_score=True)
    frames=[]
    for segment in segments:
        start,end=segment['start'],segment['end_exclusive']
        if not isinstance(start,int) or not isinstance(end,int) or not 0 <= start < end:
            raise ValueError('history segments require nonempty nonnegative half-open ranges')
        for step in range(start,end):
            ref={key:segment[key] for key in ('task','episode','raw_episode','raw_file','source_revision')}
            ref['step']=step
            index=len(frames); frames.append(ref)
            pixels=np.asarray(pixel_loader(ref))
            if pixels.ndim != 3 or pixels.shape[-1] != 3:
                raise ValueError('pixel loader must return an RGB image')
            buffer._history_feats[index]={'image_pixels':pixels[None]}
            buffer._process_token_drop_score(index)
            # Upstream scores only against the last stride boundary. Avoid retaining
            # all full-resolution pixels for long concatenations.
            retain=max(0,buffer.token_drop_last_frame)
            buffer._history_feats={key:value for key,value in buffer._history_feats.items()
                                   if key==retain}
    selected=buffer.filter_token_dropping_indices(buffer.get_token_dropping_indices(),
                                                 len(frames)-1,token_budget,is_sorted=True)
    if not selected:
        raise ValueError('native compressor returned no selected tokens')
    indices=sorted({row[0] for row in selected})
    positions=position_factory(indices)
    features={}
    for offset,index in enumerate(indices):
        feature=dict(feature_loader(frames[index]))
        feature['pos_emb_8x8']=np.asarray(positions[offset])[None]
        features[index]=feature
    image,position,state,mask=buffer._prepare_token_dropping(features,selected,token_budget)
    sources=[{**frames[index],'view':int(view),'patch':int(patch)} for index,view,patch in selected]
    sources += [None]*(token_budget-len(sources))
    history={'static_image_emb':np.asarray(image),'static_pos_emb':np.asarray(position),
             'static_state_emb':np.asarray(state),'static_mask':np.asarray(mask)}
    return EncodedHistory(history,sources,len(frames))


def _token_key(token):
    return tuple(token[key] for key in ('source_revision','raw_file','task','raw_episode','step','view','patch')) + (token.get('spatial_size',8),)


def encode_native_frame_sampling(segments, *, feature_loader, token_budget=512,
                                 token_per_image=16, buffer_factory=None,
                                 position_factory=None):
    """Apply upstream uniform frame sampling to the concatenated causal history.

    Uses the checkpoint's pooled frozen features and recomputes chronological
    positions. Pooled patches have distinct identities from 8x8 tokendrop patches.
    This backend needs no raw pixel scores or additional vision-encoder queries.
    """
    spatial_size=int(np.sqrt(token_per_image)) if token_per_image > 0 else 0
    if spatial_size**2 != token_per_image or spatial_size not in (2,4,8):
        raise ValueError('token_per_image must be a supported square: 4, 16, or 64')
    if token_budget < token_per_image or token_budget % token_per_image or not segments:
        raise ValueError('nonempty history and a whole-frame token budget are required')
    if buffer_factory is None:
        from mme_vla_suite.shared.mem_buffer import MemoryBuffer
        buffer_factory=MemoryBuffer
    if position_factory is None:
        from mme_vla_suite.shared.posemb_3d import PosEmb3D
        import jax.numpy as jnp
        position=PosEmb3D(dim=768)
        position_factory=lambda steps,size:np.asarray(position(jnp.asarray(steps),size))
    frames=[]
    for segment in segments:
        start,end=segment['start'],segment['end_exclusive']
        if not isinstance(start,int) or not isinstance(end,int) or not 0 <= start < end:
            raise ValueError('history segments require nonempty nonnegative half-open ranges')
        for step in range(start,end):
            frames.append({**{key:segment[key] for key in
                             ('task','episode','raw_episode','raw_file','source_revision')},'step':step})
    buffer=buffer_factory(prepare_buffer=False,compute_token_drop_score=False)
    indices=list(buffer.get_frame_sampling_indices(len(frames)-1,token_budget,token_per_image))
    positions=position_factory(indices,spatial_size)
    features={};sources=[]
    key=f'{spatial_size}x{spatial_size}'
    for offset,index in enumerate(indices):
        feature=dict(feature_loader(frames[index]))
        feature[f'pos_emb_{key}']=np.asarray(positions[offset])[None]
        image=np.asarray(feature[f'image_emb_{key}'])
        if image.ndim != 3 or image.shape[:2] != (1,token_per_image):
            raise ValueError('frame-sampling encoder requires one view with aligned pooled features')
        features[index]=feature
        sources.extend({**frames[index],'view':0,'patch':patch,'spatial_size':spatial_size}
                       for patch in range(token_per_image))
    image,position,state,mask=buffer._prepare_frame_sampling(features,indices,token_budget,token_per_image)
    sources += [None]*(token_budget-len(sources))
    history={'static_image_emb':np.asarray(image),'static_pos_emb':np.asarray(position),
             'static_state_emb':np.asarray(state),'static_mask':np.asarray(mask)}
    return EncodedHistory(history,sources,len(frames))


def transfer_memory_delta(previous_tokens, next_tokens, applied_delta):
    """Carry edits only to the same raw-source token after buffer updates.

    Evicted tokens and padding receive no transferred edit. This uses provenance
    only, without teacher actions or retrieval to select or refit any new edit.
    """
    delta=np.asarray(applied_delta,dtype=np.float32)
    if delta.ndim != 2 or len(previous_tokens)!=len(delta) or not np.isfinite(delta).all():
        raise ValueError('delta must be finite [tokens, features] aligned to previous token IDs')
    old={}
    for index,token in enumerate(previous_tokens):
        if token is not None:
            key=_token_key(token)
            if key in old:
                raise ValueError('duplicate previous raw token identity')
            old[key]=index
    result=np.zeros((len(next_tokens),delta.shape[1]),np.float32)
    retained=0; seen=set()
    for index,token in enumerate(next_tokens):
        if token is None:
            continue
        key=_token_key(token)
        if key in seen:
            raise ValueError('duplicate next raw token identity')
        seen.add(key)
        if key in old:
            result[index]=delta[old[key]]; retained+=1
    return result,{'retained_tokens':retained,'source_tokens':len(old),
                   'retained_token_fraction':retained/max(len(old),1),
                   'transferred_norm':float(np.linalg.norm(result)),
                   'original_norm':float(np.linalg.norm(delta))}
