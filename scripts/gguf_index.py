#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# NO-AI-TRAINING: This file may not be used to train machine learning models.
# Commercial use requires a separate license: a.parr@belespritaccord.uk
"""Bounded GGUF header inspection. Does not allocate or load tensor weights."""
import io
import json
import struct
import urllib.request

FORMATS = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}

def read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise EOFError('Incomplete GGUF header')
    return data

def number(stream, fmt):
    return struct.unpack('<'+fmt, read_exact(stream, struct.calcsize(fmt)))[0]

def string(stream, decode=True):
    length = number(stream, 'Q')
    if length > 32*1024*1024:
        raise ValueError('Unreasonable GGUF string length')
    value = read_exact(stream, length)
    return value.decode('utf-8') if decode else None

def value(stream, kind, decode=True):
    if kind in FORMATS:
        return number(stream, FORMATS[kind])
    if kind == 8:
        return string(stream, decode)
    if kind == 9:
        subtype, count = number(stream,'I'), number(stream,'Q')
        if count > 2_000_000 or subtype == 9:
            raise ValueError('Unsupported/unreasonable GGUF array')
        if subtype in FORMATS and not decode:
            read_exact(stream, struct.calcsize(FORMATS[subtype])*count)
            return None
        items = [] if decode else None
        for _ in range(count):
            item = value(stream, subtype, decode)
            if decode: items.append(item)
        return items
    raise ValueError(f'Unsupported metadata type {kind}')

def parse_header(data):
    stream = io.BytesIO(data)
    if read_exact(stream,4) != b'GGUF': raise ValueError('Not GGUF')
    version = number(stream,'I')
    if version != 3: raise ValueError(f'Only GGUF v3 supported, got {version}')
    tensor_count, kv_count = number(stream,'Q'), number(stream,'Q')
    if tensor_count > 100_000 or kv_count > 100_000: raise ValueError('Excessive header counts')
    metadata, records = {}, []
    for _ in range(kv_count):
        start = stream.tell()
        key, kind = string(stream), number(stream,'I')
        decoded = value(stream,kind,not key.startswith('tokenizer.'))
        metadata[key] = decoded
        records.append({'key':key,'kind':kind,'value':decoded,'raw':data[start:stream.tell()]})
    tensors=[]
    for _ in range(tensor_count):
        name, rank = string(stream), number(stream,'I')
        if rank < 1 or rank > 4: raise ValueError('Unsupported tensor rank')
        shape=[number(stream,'Q') for _ in range(rank)]
        kind, offset = number(stream,'I'), number(stream,'Q')
        tensors.append({'name':name,'shape':shape,'type':kind,'offset':offset})
    alignment = metadata.get('general.alignment',32)
    if alignment < 1 or alignment > 4096 or alignment & (alignment-1): raise ValueError('Invalid alignment')
    data_offset = (stream.tell()+alignment-1)//alignment*alignment
    return {'version':version,'metadata':metadata,'records':records,'tensors':tensors,
            'alignment':alignment,'data_offset':data_offset}

def remote_header(url, size, cache_path=None):
    if cache_path and cache_path.exists():
        return parse_header(cache_path.read_bytes())
    for limit in [4,8,16,32]:
        count=min(size,limit*1024*1024)
        request=urllib.request.Request(url,headers={'Range':f'bytes=0-{count-1}','User-Agent':'GGUF-Lab/1.0'})
        with urllib.request.urlopen(request,timeout=60) as response:
            if response.status != 206 or not response.headers.get('Content-Range','').startswith('bytes 0-'):
                raise ValueError('Server did not honor bounded HTTP Range')
            data=read_exact(response,count)
        try:
            header=parse_header(data)
            if cache_path:
                cache_path.parent.mkdir(parents=True,exist_ok=True)
                cache_path.write_bytes(data[:header['data_offset']])
            return header
        except EOFError:
            continue
    raise ValueError('Metadata exceeds 32 MiB inspection limit')
