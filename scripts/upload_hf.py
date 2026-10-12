#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
"""Publish two model-specific repositories with bounded CPU/RAM and no GPU use.

Reads credentials locally, never from command-line arguments and never prints them.
Only explicit model-card and artifact paths are eligible for upload.
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys

ROOT=pathlib.Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'cache/hf-sdk')
os.environ['HF_HUB_CACHE']=str(ROOT/'cache/hf-sdk/hub')
os.environ['HF_XET_CACHE']=str(ROOT/'cache/hf-sdk/xet')
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_XET_HIGH_PERFORMANCE']='0'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['RAYON_NUM_THREADS']='1'

MODELS={
 'mellum':{'repo_name':'Mellum2.1-E8-experimental-GGUF','file':'mellum-E8-experimental.gguf',
           'sha256':'4eed7d5e0915d34fd789fc3ba86e431b8f38196dc486253c84e4c71f7409092e',
           'bytes':1582936736,'metadata':['README.md','LICENSE','NOTICE','SHA256SUMS','provenance.json','validation.json']},
 'embed':{'repo_name':'EmbeddingGemma2-Q4-GGUF','file':'embeddinggemma-2-UD-Q4_K_XL.gguf',
          'sha256':'ea905fd08e8061db77a0031cbb5096d459d7cfd2bdd0ad2ba66e483f16719493',
          'bytes':175673856,'metadata':['README.md','LICENSE','NOTICE','SHA256SUMS','smoke-test.json']},
}

def read_token():
    token=os.environ.get('HF_TOKEN','').strip()
    if not token:
        candidates=[ROOT/'.local/hf-token',pathlib.Path.home()/'.cache/huggingface/token']
        token=next((p.read_text(encoding='utf-8-sig').strip() for p in candidates if p.is_file()),'')
    if not token:
        raise ValueError('A write-capable Hugging Face token is required in .local/hf-token or HF_TOKEN.')
    if not token.startswith('hf_'):
        raise ValueError('The token file must contain only the Hugging Face token.')
    return token

def verify_local(key):
    model=MODELS[key]
    path=ROOT/'models'/model['file']
    if not path.is_file() or path.stat().st_size!=model['bytes']:
        raise ValueError(f'{key}: model file is missing or its size changed')
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(1024*1024): digest.update(chunk)
    if digest.hexdigest()!=model['sha256']:
        raise ValueError(f'{key}: model SHA256 changed')
    for filename in model['metadata']:
        if not (ROOT/'publish'/key/filename).is_file(): raise ValueError(f'Missing model metadata: {key}/{filename}')
    return path

def remote_hash(file):
    lfs=getattr(file,'lfs',None)
    if isinstance(lfs,dict): return lfs.get('sha256') or lfs.get('oid')
    return getattr(lfs,'sha256',None) or getattr(lfs,'oid',None)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--namespace',help='Defaults to the account identified by the token')
    parser.add_argument('--auth-check',action='store_true')
    args=parser.parse_args()
    token=read_token()
    from huggingface_hub import HfApi,CommitOperationAdd
    api=HfApi(token=token)
    identity=api.whoami()
    namespace=args.namespace or identity['name']
    print('Authenticated Hugging Face account:',identity['name'],flush=True)
    if args.auth_check: return 0
    # Verify both artifacts before creating or modifying any external repository.
    paths={key:verify_local(key) for key in MODELS}
    results={}
    for key,model in MODELS.items():
        repo_id=namespace+'/'+model['repo_name']
        api.create_repo(repo_id=repo_id,repo_type='model',private=False,exist_ok=True)
        existing=api.get_paths_info(repo_id,[model['file']],repo_type='model')
        if existing and remote_hash(existing[0])!=model['sha256']:
            raise ValueError(f'{repo_id}: refusing to overwrite a different remote model artifact')
        if not existing:
            existing_card=api.get_paths_info(repo_id,['README.md'],repo_type='model')
            if existing_card: raise ValueError(f'{repo_id}: existing model card needs review before reuse')
        operations=[CommitOperationAdd(path_in_repo=filename,path_or_fileobj=ROOT/'publish'/key/filename)
                    for filename in model['metadata']]
        if not existing:
            operations.append(CommitOperationAdd(path_in_repo=model['file'],path_or_fileobj=paths[key]))
        print('Uploading:',repo_id,'(one transfer worker; GPU unused)',flush=True)
        commit=api.create_commit(repo_id=repo_id,repo_type='model',operations=operations,num_threads=1,
                                 commit_message='Publish audited low-memory GGUF artifact and model-specific provenance')
        remote=api.get_paths_info(repo_id,[model['file']],revision=commit.oid,repo_type='model')
        if len(remote)!=1 or remote[0].size!=model['bytes'] or remote_hash(remote[0])!=model['sha256']:
            raise ValueError(f'{repo_id}: uploaded artifact hash/size verification failed')
        results[key]={'repo_id':repo_id,'url':'https://huggingface.co/'+repo_id,'commit':commit.oid,
                      'file':model['file'],'bytes':model['bytes'],'sha256':model['sha256'],
                      'remote_hash_verified':True,'gpu_used':False}
        (ROOT/'reports/hf-upload.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
        print('Verified:',results[key]['url'],flush=True)
    return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as error:
        # Keep SDK request details and authorization headers out of terminal output.
        status=getattr(getattr(error,'response',None),'status_code',None)
        if isinstance(error,(ValueError,FileNotFoundError)):
            print(str(error),file=sys.stderr)
        else:
            print(f'Hugging Face upload failed: {type(error).__name__}, HTTP status {status}',file=sys.stderr)
        raise SystemExit(1)
