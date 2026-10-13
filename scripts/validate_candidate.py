#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# NO-AI-TRAINING: This file may not be used to train machine learning models.
# Commercial use requires a separate license: a.parr@belespritaccord.uk
"""Independent artifact validation without inference or full weight allocation."""
import argparse
import hashlib
import json
import pathlib
from gguf_index import parse_header
from prune_mellum import tensor_bytes

def validate(path,source):
    provenance=json.loads(path.with_suffix('.provenance.json').read_text(encoding='utf-8'))
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        header=parse_header(stream.read(16*1024*1024))
        stream.seek(0)
        while data:=stream.read(1024*1024): digest.update(data)
    actual=path.stat().st_size
    if actual!=provenance['output_bytes'] or digest.hexdigest()!=provenance['output_sha256']:
        raise ValueError('Output file differs from its recorded SHA256/size')
    if actual>int(1.5*1024**3): raise ValueError('Weight budget exceeded')
    meta=header['metadata']
    if meta['mellum.expert_count']!=8 or meta['mellum.expert_used_count']!=8 or meta['mellum.block_count']!=28:
        raise ValueError('Unexpected pruned architecture')
    original={t['name']:t for t in source['tensors']}
    if set(original)!={t['name'] for t in header['tensors']}: raise ValueError('Tensor names were lost or added')
    for tensor in header['tensors']:
        prior=original[tensor['name']]
        expected=list(prior['shape'])
        if '_exps.' in tensor['name'] or 'ffn_gate_inp' in tensor['name']: expected[-1]=8
        if tensor['shape']!=expected or tensor['type']!=prior['type']:
            raise ValueError('Tensor shape/type differs from the expert-pruning contract')
        if header['data_offset']+tensor['offset']+tensor_bytes(tensor)>actual:
            raise ValueError('Tensor extends beyond output file')
    return {'path':str(path),'bytes':actual,'sha256':digest.hexdigest(),'tensor_count':len(header['tensors']),
            'expert_count':8,'expert_used_count':8,'layers':28,'gpu_used':False,
            'size_reduction_percent':100*(1-actual/source['source_bytes']),
            'status':'structural_and_hash_validation_passed','inference_validated':False,
            'coding_quality_validated':False,'runtime_vram_measured':False}

if __name__=='__main__':
    root=pathlib.Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument('path',type=pathlib.Path)
    args=parser.parse_args()
    source=json.loads((root/'reports/model-audit.json').read_text(encoding='utf-8'))['mellum']
    result=validate(args.path,source)
    (root/'reports/mellum-validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
