#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Experimental byte-preserving expert pruning, with bounded RAM and zero GPU use.

Selects experts using router-row L2 norms, not measured activation utility.
Only supports the audited Mellum GGUF layout. Never overwrites source files.
"""
import argparse
import hashlib
import json
import math
import pathlib
import re
import struct
import time
import urllib.request
from gguf_index import FORMATS, parse_header, remote_header

ROOT=pathlib.Path(__file__).resolve().parents[1]
TYPE_SIZES={0:(1,4),6:(32,22),8:(32,34),12:(256,144),14:(256,210)}
CHUNK=4*1024*1024

def tensor_bytes(tensor):
    block, size=TYPE_SIZES[tensor['type']]
    elements=math.prod(tensor['shape'])
    if tensor['shape'][0] % block or elements % block:
        raise ValueError('Tensor does not align with its quantization block')
    return elements//block*size

def fetch_range(url,start,length,total):
    if length<1 or length>CHUNK: raise ValueError('Invalid bounded read length')
    expected=f'bytes {start}-{start+length-1}/{total}'
    for attempt in range(3):
        try:
            request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{start+length-1}',
                                                       'User-Agent':'GGUF-Lab/1.0'})
            with urllib.request.urlopen(request,timeout=90) as response:
                if response.status!=206 or response.headers.get('Content-Range')!=expected:
                    raise ValueError('Server returned the wrong byte range')
                data=response.read(length+1)
                if len(data)!=length: raise ValueError('Incomplete byte range')
                return data
        except (OSError,ValueError):
            if attempt==2: raise
            time.sleep(attempt+1)

def encode_string(text):
    data=text.encode('utf-8')
    return struct.pack('<Q',len(data))+data

def encode_record(key,kind,value):
    if kind==8: encoded=encode_string(value)
    elif kind in FORMATS: encoded=struct.pack('<'+FORMATS[kind],value)
    else: raise ValueError('Only scalar metadata overrides are permitted')
    return encode_string(key)+struct.pack('<I',kind)+encoded

def assemble_header(source,tensors,overrides):
    records=[]
    for record in source['records']:
        if record['key'] in overrides:
            kind,value=overrides[record['key']]
            records.append(encode_record(record['key'],kind,value))
        else: records.append(record['raw'])
    existing={record['key'] for record in source['records']}
    for key,(kind,value) in overrides.items():
        if key not in existing: records.append(encode_record(key,kind,value))
    header=b'GGUF'+struct.pack('<IQQ',3,len(tensors),len(records))+b''.join(records)
    for tensor in tensors:
        header+=encode_string(tensor['name'])+struct.pack('<I',len(tensor['shape']))
        header+=struct.pack('<'+'Q'*len(tensor['shape']),*tensor['shape'])
        header+=struct.pack('<IQ',tensor['type'],tensor['output_offset'])
    return header+b'\0'*((-len(header))%source['alignment'])

def validate_layout(source,info,experts):
    meta=source['metadata']
    if meta.get('general.architecture')!='mellum' or meta.get('split.count',1)!=1:
        raise ValueError('Only complete, unsplit Mellum files are supported')
    original=meta['mellum.expert_count']
    active=meta['mellum.expert_used_count']
    layers=meta['mellum.block_count']
    if not active<=experts<original: raise ValueError('Keep at least the existing active expert count')
    tensors=sorted(source['tensors'],key=lambda t:t['offset'])
    for i,tensor in enumerate(tensors):
        size=tensor_bytes(tensor)
        limit=tensors[i+1]['offset'] if i+1<len(tensors) else info['source_bytes']-source['data_offset']
        if tensor['offset']+size>limit: raise ValueError('Tensor data overlaps or exceeds source size')
        if '_exps.' in tensor['name']:
            if len(tensor['shape'])!=3 or tensor['shape'][-1]!=original:
                raise ValueError('Unexpected expert bank dimensions')
        elif 'ffn_gate_inp' in tensor['name']:
            if tensor['type']!=0 or tensor['shape']!=[meta['mellum.embedding_length'],original]:
                raise ValueError('Expected a float32 router matrix')
        elif len(tensor['shape'])==1 and tensor['shape'][0]==original:
            raise ValueError('Expert-sized auxiliary vectors need an explicit adapter')
    for layer in range(layers):
        names={t['name'] for t in tensors if t['name'].startswith(f'blk.{layer}.')}
        needed={f'blk.{layer}.{name}.weight' for name in ['ffn_gate_inp','ffn_gate_exps','ffn_up_exps','ffn_down_exps']}
        if not needed<=names: raise ValueError('Missing router or expert bank')
    return tensors, original, layers

def prune(info,output,experts=8,max_bytes=int(1.5*1024**3)):
    if output.exists() or output.with_suffix('.part').exists():
        raise FileExistsError('Preserve existing artifacts; choose a new output path')
    cache=ROOT/'cache'/info['cache_key'] if info.get('cache_key') else None
    source=remote_header(info['url'],info['source_bytes'],cache)
    tensors,original,layers=validate_layout(source,info,experts)
    plans=[]
    offset=0
    for tensor in tensors:
        target=dict(tensor,shape=list(tensor['shape']))
        changed='_exps.' in tensor['name'] or 'ffn_gate_inp' in tensor['name']
        if changed: target['shape'][-1]=experts
        target['output_offset']=offset
        target['bytes']=tensor_bytes(target)
        target['original_bytes']=tensor_bytes(tensor)
        target['pruned']=changed
        plans.append(target)
        offset+=target['bytes']
        offset+=( -offset)%source['alignment']
    overrides={
      'mellum.expert_count':(4,experts),
      'general.name':(8,f'Mellum2.1-E{experts}-router-L2-experimental'),
      'general.size_label':(8,f'{experts}-expert-pruned'),
      'lab.pruning.method':(8,'router-row-l2; retained quantized bytes unchanged; not calibrated'),
      'lab.pruning.source_repo':(8,info['repo']),
      'lab.pruning.source_revision':(8,info['revision']),
      'lab.pruning.quality_status':(8,'unvalidated experimental candidate'),
    }
    header=assemble_header(source,plans,overrides)
    projected=len(header)+offset
    if projected>max_bytes: raise ValueError(f'Output estimate {projected} exceeds budget {max_bytes}')
    print(f'Output budget verified: {projected} bytes; {original} -> {experts} experts, {layers} layers unchanged',flush=True)
    by_name={t['name']:t for t in tensors}
    selection={}
    for layer in range(layers):
        router=by_name[f'blk.{layer}.ffn_gate_inp.weight']
        data=fetch_range(info['url'],source['data_offset']+router['offset'],tensor_bytes(router),info['source_bytes'])
        row=router['shape'][0]*4
        scores=[]
        for expert in range(original):
            weights=[x[0] for x in struct.iter_unpack('<f',data[expert*row:(expert+1)*row])]
            if not all(math.isfinite(w) for w in weights): raise ValueError('Nonfinite router weight')
            scores.append(math.fsum(w*w for w in weights))
        chosen=sorted(sorted(range(original),key=lambda i:(-scores[i],i))[:experts])
        selection[layer]={'experts':chosen,'l2_squared':[scores[i] for i in chosen]}
        print(f'Router {layer+1}/{layers}: selected experts {chosen}',flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.part')
    digest=hashlib.sha256()
    def write(stream,data):
        stream.write(data)
        digest.update(data)
    downloaded=layers*source['metadata']['mellum.embedding_length']*original*4
    with temporary.open('xb') as stream:
        write(stream,header)
        for i,plan in enumerate(plans):
            position=source['data_offset']+plan['offset']
            ranges=[(position,plan['original_bytes'])]
            if plan['pruned']:
                layer=int(re.match(r'blk\.(\d+)\.',plan['name'])[1])
                stride=plan['original_bytes']//original
                ranges=[(position+expert*stride,stride) for expert in selection[layer]['experts']]
            for start,length in ranges:
                while length:
                    count=min(length,CHUNK)
                    data=fetch_range(info['url'],start,count,info['source_bytes'])
                    write(stream,data)
                    downloaded+=count
                    start+=count
                    length-=count
            write(stream,b'\0'*((-plan['bytes'])%source['alignment']))
            if i%12==0 or i==len(plans)-1:
                print(f'{i+1}/{len(plans)} tensors, {stream.tell()/1024**2:.1f} MiB written',flush=True)
    if temporary.stat().st_size!=projected: raise ValueError('Output size differs from plan')
    with temporary.open('rb') as check:
        parsed=parse_header(check.read(len(header)))
    if parsed['metadata']['mellum.expert_count']!=experts or len(parsed['tensors'])!=len(plans):
        raise ValueError('Output metadata validation failed')
    temporary.rename(output)
    provenance={'source_repo':info['repo'],'source_revision':info['revision'],
      'source_expected_sha256':info['sha256'],'full_source_hash_verified':False,
      'source_bytes':info['source_bytes'],'output_bytes':output.stat().st_size,'output_sha256':digest.hexdigest(),
      'downloaded_weight_bytes':downloaded,'original_experts':original,'retained_experts':experts,
      'layers':layers,'expert_selection':selection,'gpu_used':False,'chunk_bytes':CHUNK,
      'method':'uncalibrated router-row L2 selection; byte-preserving expert-bank slicing',
      'quality_status':'not evaluated; structural validation is not coding/tool-use validation'}
    output.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in provenance.items() if k!='expert_selection'}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--experts',type=int,default=8)
    parser.add_argument('--output',type=pathlib.Path,default=ROOT/'models'/'mellum-E8-experimental.gguf')
    args=parser.parse_args()
    info=json.loads((ROOT/'reports'/'model-audit.json').read_text(encoding='utf-8'))['mellum']
    prune(info,args.output,args.experts)
