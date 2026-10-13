#!/usr/bin/env python3
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# NO-AI-TRAINING: This file may not be used to train machine learning models.
# Commercial use requires a separate license: a.parr@belespritaccord.uk
"""Run a bounded CPU smoke check only when enough physical RAM is free."""
import argparse
import ctypes
import json
import os
import pathlib
import subprocess

ROOT=pathlib.Path(__file__).resolve().parents[1]

def free_ram():
    if os.name=='nt':
        class MemoryStatus(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in
                ['total_phys','available_phys','total_page','available_page','total_virtual','available_virtual','available_extended']]
        status=MemoryStatus()
        status.length=ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError('Cannot read available physical memory')
        return status.available_phys
    return os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime',type=pathlib.Path,required=True,help='Path to CPU-only llama-cli executable')
    args=parser.parse_args()
    available=free_ram()
    required=2*1024**3
    report={'gpu_layers':0,'threads':1,'free_physical_ram_bytes':available,'required_free_ram_bytes':required}
    if available<required:
        report.update(status='blocked_by_free_ram',inference_ran=False,quality_validated=False)
        (ROOT/'reports/mellum-smoke.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report,indent=2))
        return 3
    model=ROOT/'models/mellum-E8-experimental.gguf'
    if not model.exists(): raise FileNotFoundError('Pruning artifact is not complete')
    if not args.runtime.is_file(): raise FileNotFoundError('CPU runtime is missing')
    command=[str(args.runtime.resolve()),'-m',str(model),'-ngl','0','-t','1','-tb','1','-c','512',
             '-b','32','-ub','32','-n','32','--single-turn','--no-warmup',
             '-p','Write a Python function add(a, b) that returns a + b. Output code only.']
    flags=subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name=='nt' else 0
    try:
        result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',
                              timeout=180,creationflags=flags)
        report.update(status='runtime_completed' if result.returncode==0 else 'runtime_failed',
                      inference_ran=True,exit_code=result.returncode,output=result.stdout[-6000:],
                      runtime_errors=result.stderr[-3000:],quality_validated=False)
    except subprocess.TimeoutExpired:
        report.update(status='runtime_timeout',inference_ran=True,quality_validated=False)
    (ROOT/'reports/mellum-smoke.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='runtime_completed' else 1

if __name__=='__main__': raise SystemExit(main())
