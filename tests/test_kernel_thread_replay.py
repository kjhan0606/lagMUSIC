#!/usr/bin/env python3
"""Bounded full-field thread replay using one immutable external white-noise cube.

Run on a compute host with --music /path/MUSIC --white-noise /path/wnoise_0004.bin
--root /fresh/output/root. Maximum four total CPU threads. Validates physical
kernel cells only (FFTW's two padding cells are not physical data), noise,
density, velocity, displacement and mask. No RAMSES evolution/dumps.
--base-level 5 with an L5 cube checks the ordinary non-overlapping fast path.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import subprocess

import numpy as np
from test_rng_periodic_replay import FIELDS, TOL, compare, config, read_noise, read_grafic


def read_kernel(path):
    raw = path.read_bytes()
    dims = struct.unpack('=3I', raw[:12])
    cells = int(np.prod(dims))
    if min(dims) <= 2 or cells > 2**21 or len(raw) != 12+cells*8:
        raise AssertionError(f'bad bounded kernel cache: {path}')
    # Caches contain in-place real FFT storage, not complex coefficients.
    physical = np.frombuffer(raw, '=f8', offset=12).reshape(dims)[:, :, :-2].copy()
    if not np.isfinite(physical).all():
        raise AssertionError('nonfinite physical kernel')
    return physical.shape, physical.ravel()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--music', type=Path, required=True)
    ap.add_argument('--white-noise', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--base-level', type=int, choices=(4,5), default=4)
    ap.add_argument('--mpirun', default='mpirun')
    args = ap.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0,0))
    args.music, args.white_noise = args.music.resolve(), args.white_noise.resolve()
    anchor_raw = args.white_noise.read_bytes()
    anchor_sha = hashlib.sha256(anchor_raw).hexdigest()
    base = args.base_level
    if read_noise(args.white_noise)[0] != (2**base,)*3:
        raise ValueError('external anchor dimensions must match base-level')
    args.root = args.root.resolve()
    args.root.mkdir(exist_ok=False)
    cases = [('r1t1',1,1), ('r1t2',1,2), ('r1t4',1,4),
             ('r1t4_repeat',1,4), ('r2t2',2,2), ('r4t1',4,1)]
    report = {'binary_sha256':hashlib.sha256(args.music.read_bytes()).hexdigest(),
              'input_anchor_sha256':anchor_sha, 'base_level':base, 'field_tolerance':TOL,
              'kernel_tolerance':1e-12, 'noise_tolerance':1e-12,
              'production_release':False, 'cases':{}, 'replay_pass':False}
    failure = []
    reference = args.root/'r1t1'
    for label, ranks, threads in cases:
        case = args.root/label
        case.mkdir()
        text = config(case, base).replace(f'seed[{base}] = 20260401',f'seed[{base}] = {args.white_noise}')
        (case/'music.conf').write_text(text)
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (case/'music.log').open('w') as log:
            proc = subprocess.run([args.mpirun,'-np',str(ranks),str(args.music),'music.conf'],
                                  cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,
                                  timeout=180)
        log = (case/'music.log').read_text()
        if proc.returncode or 'fatal error' in log.lower() or 'Reading compact MUSIC white noise file' not in log:
            raise AssertionError(f'MUSIC/read path failed: {case}/music.log')
        row = {'ranks':ranks,'omp_threads':threads,'kernel':{},'noise':{},'fields':{}}
        for level in range(base,base+3):
            comparisons = [('kernel',f'temp_kernel_level{level:03d}.tmp',read_kernel,1e-12),
                           ('noise',f'wnoise_{level:04d}.bin',read_noise,1e-12)]
            comparisons += [('fields',f'ic/level_{level:03d}/{field}',read_grafic,TOL) for field in FIELDS]
            for category, relative, reader, tolerance in comparisons:
                item = compare(reader(reference/relative),reader(case/relative))
                item['pass'] = item['relative_l2'] <= tolerance
                row[category][relative] = item
                if not item['pass']:
                    failure.append(f'{label}/{relative}')
        report['cases'][label] = row
        (args.root/'thread_replay.json').write_text(json.dumps(report,indent=2)+'\n')
        print(label,'max_kernel',max(x['relative_l2'] for x in row['kernel'].values()),
              'max_field',max(x['relative_l2'] for x in row['fields'].values()),flush=True)
    if hashlib.sha256(args.white_noise.read_bytes()).hexdigest() != anchor_sha:
        raise AssertionError('immutable external anchor changed')
    report['input_anchor_unchanged'] = True
    report['failed_fields'] = failure
    report['replay_pass'] = not failure
    (args.root/'thread_replay.json').write_text(json.dumps(report,indent=2)+'\n')
    if failure:
        raise AssertionError(f'{len(failure)} failed thread/kernel comparisons; see thread_replay.json')
    print('KERNEL_THREAD_REPLAY=passed',flush=True)


if __name__ == '__main__':
    main()
