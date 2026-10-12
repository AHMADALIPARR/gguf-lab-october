# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
import contextlib
import hashlib
import http.server
import io
import pathlib
import re
import struct
import sys
import tempfile
import threading
import unittest

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from gguf_index import parse_header
from prune_mellum import assemble_header,encode_record,prune,validate_layout

def fixture():
    metadata={'general.architecture':(8,'mellum'),'mellum.expert_count':(4,4),
              'mellum.expert_used_count':(4,2),'mellum.block_count':(4,1),
              'mellum.embedding_length':(4,8)}
    records=[{'key':k,'raw':encode_record(k,kind,v)} for k,(kind,v) in metadata.items()]
    tensors=[]
    payload=bytearray()
    for name,shape in [('token_embd.weight',[8,4]),('blk.0.ffn_gate_inp.weight',[8,4]),
                      ('blk.0.ffn_gate_exps.weight',[8,8,4]),('blk.0.ffn_up_exps.weight',[8,8,4]),
                      ('blk.0.ffn_down_exps.weight',[8,8,4])]:
        offset=len(payload)
        if 'gate_inp' in name:
            values=[float(expert+1) for expert in range(4) for _ in range(8)]
        else: values=[float(i) for i in range(__import__('math').prod(shape))]
        payload+=struct.pack('<'+'f'*len(values),*values)
        payload+=b'\0'*((-len(payload))%32)
        tensors.append({'name':name,'shape':shape,'type':0,'offset':offset,'output_offset':offset})
    header=assemble_header({'records':records,'alignment':32},tensors,{})
    return header+payload,len(header)

class StreamingPruneTests(unittest.TestCase):
    def test_pruning_preserves_selected_weight_bytes_and_updates_router(self):
        model,data_offset=fixture()
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                start,end=map(int,re.fullmatch(r'bytes=(\d+)-(\d+)',self.headers['Range']).groups())
                chunk=model[start:end+1]
                self.send_response(206)
                self.send_header('Content-Range',f'bytes {start}-{end}/{len(model)}')
                self.send_header('Content-Length',str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
            def log_message(self,*args): pass
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                output=pathlib.Path(directory)/'candidate.gguf'
                info={'url':f'http://127.0.0.1:{server.server_port}/model','source_bytes':len(model),
                      'repo':'test/fixture','revision':'fixture','sha256':hashlib.sha256(model).hexdigest()}
                with contextlib.redirect_stdout(io.StringIO()): prune(info,output,2,100_000)
                result=output.read_bytes()
                parsed=parse_header(result)
                self.assertEqual(parsed['metadata']['mellum.expert_count'],2)
                self.assertEqual(parsed['metadata']['mellum.expert_used_count'],2)
                for tensor in parsed['tensors']:
                    start=parsed['data_offset']+tensor['offset']
                    if 'gate_inp' in tensor['name']:
                        values=struct.unpack('<16f',result[start:start+64])
                        self.assertEqual(values,(3.0,)*8+(4.0,)*8)
                    elif '_exps.' in tensor['name']:
                        self.assertEqual(tensor['shape'][-1],2)
                        original=next(t for t in parse_header(model)['tensors'] if t['name']==tensor['name'])
                        old=data_offset+original['offset']+2*8*8*4
                        self.assertEqual(result[start:start+2*8*8*4],model[old:old+2*8*8*4])
                import json
                provenance=json.loads(output.with_suffix('.provenance.json').read_text())
                self.assertEqual(provenance['output_sha256'],hashlib.sha256(result).hexdigest())
                self.assertFalse(provenance['gpu_used'])
                with self.assertRaises(FileExistsError): prune(info,output,2,100_000)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_invalid_or_incomplete_layouts_fail_closed(self):
        model,_=fixture()
        header=parse_header(model)
        info={'source_bytes':len(model)}
        with self.assertRaises(ValueError): validate_layout(header,info,1)
        header['metadata']['split.count']=2
        with self.assertRaises(ValueError): validate_layout(header,info,2)
        with self.assertRaises(EOFError): parse_header(model[:20])

if __name__=='__main__': unittest.main()
