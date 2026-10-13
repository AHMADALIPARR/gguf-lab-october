#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# NO-AI-TRAINING: This file may not be used to train machine learning models.
# Commercial use requires a separate license: a.parr@belespritaccord.uk
"""Download a verified small artifact without loading model weights."""
import hashlib
import json
import pathlib
import urllib.request
from gguf_index import parse_header

ROOT=pathlib.Path(__file__).resolve().parents[1]

def main():
    info=json.loads((ROOT/'reports/model-audit.json').read_text(encoding='utf-8'))['embed']
    if info['source_bytes']>256*1024**2: raise ValueError('Embedding download exceeds 256 MiB budget')
    output=ROOT/'models'/info['filename']
    if output.exists(): raise FileExistsError('Preserve existing model artifact')
    output.parent.mkdir(exist_ok=True)
    temporary=output.with_suffix('.part')
    digest=hashlib.sha256()
    size=0
    with urllib.request.urlopen(info['url'],timeout=90) as response,temporary.open('xb') as stream:
        while chunk:=response.read(1024*1024):
            size+=len(chunk)
            if size>info['source_bytes']: raise ValueError('Download exceeds audited size')
            stream.write(chunk)
            digest.update(chunk)
    if size!=info['source_bytes'] or digest.hexdigest()!=info['sha256']:
        raise ValueError('Artifact size or SHA256 did not match upstream metadata')
    with temporary.open('rb') as stream: parse_header(stream.read(info['data_offset']))
    temporary.rename(output)
    print(json.dumps({'path':str(output),'bytes':size,'sha256':digest.hexdigest(),'gpu_used':False}),flush=True)

if __name__=='__main__': main()
