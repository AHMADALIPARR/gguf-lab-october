#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# NO-AI-TRAINING: This file may not be used to train machine learning models.
# Commercial use requires a separate license: a.parr@belespritaccord.uk
import json
import pathlib
import urllib.parse
import urllib.request
from gguf_index import remote_header

ROOT=pathlib.Path(__file__).resolve().parents[1]
MODELS={
 'embed':('unsloth/embeddinggemma-2-GGUF','embeddinggemma-2-UD-Q4_K_XL.gguf'),
 'mellum':('JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF','Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf'),
 'qwen27':('unsloth/Qwen3.8-27B-GGUF','Qwen3.8-27B-UD-Q4_K_M.gguf'),
 'bonsai':('prism-ml/Ternary-Bonsai-2-27B-gguf','Ternary-Bonsai-2-27B-PTQ1_0.gguf'),
 'xing':('XingChen-AGI/Xing4.0-29B-A4B-GGUF','xing4_0-29b-IQ4_NL.gguf'),
 'flash':('ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF','Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf'),
}

def main():
    report={}
    for key,(repo,filename) in MODELS.items():
        try:
            with urllib.request.urlopen('https://huggingface.co/api/models/'+repo+'?blobs=true',timeout=30) as r:
                info=json.load(r)
            file=next(f for f in info['siblings'] if f['rfilename']==filename)
            url='https://huggingface.co/'+repo+'/resolve/'+info['sha']+'/'+urllib.parse.quote(filename)
            cache_key=f'{key}.{info["sha"]}.header'
            header=remote_header(url,file['size'],ROOT/'cache'/cache_key)
            metadata={k:v for k,v in header['metadata'].items() if not k.startswith('tokenizer.')}
            report[key]={'repo':repo,'revision':info['sha'],'filename':filename,'source_bytes':file['size'],
                         'sha256':file.get('lfs',{}).get('sha256'),'url':url,'metadata':metadata,
                         'cache_key':cache_key,'data_offset':header['data_offset'],'tensors':header['tensors']}
            print(json.dumps({'model':key,'bytes':file['size'],'architecture':metadata.get('general.architecture'),
                              'tensor_count':len(header['tensors'])}),flush=True)
        except Exception as error:
            report[key]={'repo':repo,'filename':filename,'error':str(error)}
            print(key, str(error),flush=True)
        (ROOT/'reports').mkdir(exist_ok=True)
        (ROOT/'reports'/'model-audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__': main()
