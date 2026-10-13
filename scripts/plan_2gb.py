#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# NO-AI-TRAINING: This file may not be used to train machine learning models.
# Commercial use requires a separate license: a.parr@belespritaccord.uk
"""Measure byte budgets; a weight-file size is not a runtime VRAM guarantee."""
import json
import pathlib

ROOT=pathlib.Path(__file__).resolve().parents[1]
BUDGET=int(1.5*1024**3)
POLICY={
 'embed':'Already small enough. Preserve pretrained embedding weights; no pruning required.',
 'mellum':'Build experimental E8 candidate using router L2 ranking, retain all 28 layers. Quality unvalidated.',
 'qwen27':'Blocked for this budget: Q4 fixed non-layer tensors alone exceed the weight allowance. Width/vocabulary changes need training and an architecture adapter.',
 'bonsai':'Special PTQ1_0 type 143 and Hadamard metadata need the matching runtime and a topology adapter. Do not silently requantize ternary weights.',
 'xing':'Expert-only pruning cannot meet this budget; nonexpert tensors alone exceed 4 GB. Hyper-connections, MLA and prediction layers need a separate adapter.',
 'flash':'Split 177B model: the audited first shard is incomplete and nonexpert tensors exceed the budget. The second shard and PLE tensors must be accounted for; not a local 2 GB candidate.',
}

def main():
    audit=json.loads((ROOT/'reports/model-audit.json').read_text(encoding='utf-8'))
    plan={'vram_budget_bytes':2*1024**3,'weight_budget_bytes':BUDGET,'gpu_processing':False,
          'quality_gate':'No altered model is production-ready without independent inference and task-quality evaluation.',
          'models':{}}
    for key,info in audit.items():
        if 'error' in info:
            plan['models'][key]={'status':'audit_failed','error':info['error']}
            continue
        tensors=sorted(info['tensors'],key=lambda t:t['offset'])
        spans={t['name']:(tensors[i+1]['offset'] if i+1<len(tensors) else info['source_bytes']-info['data_offset'])-t['offset'] for i,t in enumerate(tensors)}
        experts=sum(spans[t['name']] for t in tensors if '_exps.' in t['name'])
        fixed=sum(spans[t['name']] for t in tensors if not t['name'].startswith('blk.'))
        plan['models'][key]={'source_bytes':info['source_bytes'],'architecture':info['metadata']['general.architecture'],
                            'fixed_nonlayer_bytes':fixed,'expert_bank_bytes':experts,
                            'nonexpert_bytes':sum(spans.values())-experts,
                            'policy':POLICY[key]}
    (ROOT/'reports/2gb-plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(plan,indent=2))

if __name__=='__main__': main()
