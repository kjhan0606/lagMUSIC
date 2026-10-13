#!/usr/bin/env python3
"""Small MPI/OMP RNG regression (numpy required).

Run on a compute host, e.g. python3 tests/test_rng_periodic_replay.py
--music /absolute/path/MUSIC --root /fresh/output/directory. Uses at most four
MPI ranks or four OMP threads, with auto RNG boost capped by those resources.
The existing output directory is never reused. No RAMSES runs/full dumps.
Full-field replay covers MPI ranks with OMP=1 and the internal RNG boost;
threaded periodic unpack is tested separately by test_rng_periodic_unpack.cc.
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

FIELDS = ("ic_deltab", "ic_velcx", "ic_velcy", "ic_velcz", "ic_poscx",
          "ic_poscy", "ic_poscz", "ic_refmap")
MAX_CELLS = 2**21
TOL = 2e-5


def read_noise(path):
    raw = path.read_bytes()
    dims = struct.unpack("=3I", raw[:12])
    cells = int(np.prod(dims))
    if not 0 < cells <= MAX_CELLS or len(raw) != 12+cells*8:
        raise AssertionError(f"bad bounded noise file: {path}")
    data = np.frombuffer(raw, dtype="=f8", offset=12)
    if not np.isfinite(data).all():
        raise AssertionError(f"nonfinite noise: {path}")
    return dims, data


def read_grafic(path):
    with path.open("rb") as stream:
        marker = stream.read(4)
        endian = "<" if struct.unpack("<i", marker)[0] in (44, 48) else ">"
        size = struct.unpack(endian+"i", marker)[0]
        if size not in (44, 48):
            raise AssertionError(f"bad header: {path}")
        header = stream.read(size)
        if stream.read(4) != marker:
            raise AssertionError(f"bad header record: {path}")
        nx, ny, nz = struct.unpack(endian+"3i", header[:12])
        if min(nx, ny, nz) <= 0 or nx*ny*nz > MAX_CELLS:
            raise AssertionError(f"bad bounded dimensions: {path}")
        planes = []
        for _ in range(nz):
            count = 4*nx*ny
            if stream.read(4) != struct.pack(endian+"i", count):
                raise AssertionError(f"bad plane record: {path}")
            planes.append(np.frombuffer(stream.read(count), dtype=endian+"f4"))
            if planes[-1].size != nx*ny or stream.read(4) != struct.pack(endian+"i", count):
                raise AssertionError(f"short plane record: {path}")
        data = np.concatenate(planes)
        if stream.read(1) or not np.isfinite(data).all():
            raise AssertionError(f"bad payload: {path}")
    return (nx, ny, nz), data


def compare(a, b):
    if a[0] != b[0]:
        raise AssertionError("different dimensions")
    x, y = a[1].astype(float), b[1].astype(float)
    return {"relative_l2": float(np.linalg.norm(x-y)/max(np.linalg.norm(x), 1e-30)),
            "bitwise_equal": bool(np.array_equal(x, y))}


def config(case, base, extra=""):
    return f"""[setup]
boxlength = 128
zstart = 100
levelmin = {base}
levelmin_TF = {base}
levelmax = {base+2}
padding = 2
overlap = 2
ref_center = 0.5, 0.5, 0.5
ref_extent = 0.25, 0.25, 0.25
align_top = no
baryons = no
use_2LPT = yes
dmo_velocity_source = density_2lpt
vfact_scale = 1
use_LLA = no
periodic_TF = yes
kspace_TF = no
{extra}
[cosmology]
Omega_m = 0.3111
Omega_L = 0.6889
Omega_b = 0.04
H0 = 67.66
sigma_8 = 0.8102
nspec = 0.9665
w0 = -1
wa = 0
transfer = eisenstein
[random]
disk_cached = yes
seed[{base}] = 20260401
seed[{base+1}] = 20260402
seed[{base+2}] = 20260403
[output]
format = grafic2
filename = {case / 'ic'}
ramses_nml = yes
[poisson]
fft_fine = yes
accuracy = 1e-6
pre_smooth = 3
post_smooth = 3
smoother = gs
laplace_order = 6
grad_order = 6
"""


def run_case(args, name, base, ranks, threads=1, extra="", expect_reject=False):
    case = args.root / name
    case.mkdir()
    text = config(case, base, extra)
    (case / "music.conf").write_text(text)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    with (case / "music.log").open("w") as log:
        proc = subprocess.run([args.mpirun, "-np", str(ranks), str(args.music), "music.conf"],
                              cwd=case, env=env, stdout=log, stderr=subprocess.STDOUT,
                              timeout=180)
    log = (case / "music.log").read_text()
    # MUSIC currently returns zero even after some caught runtime errors.
    fatal = "fatal error" in log.lower()
    if expect_reject:
        if not (proc.returncode or fatal) or "periodic writeback alias" not in log:
            raise AssertionError("forced slab path was not explicitly rejected")
        if (case / f"wnoise_{base:04d}.bin").exists():
            raise AssertionError("rejected RNG path wrote white noise")
    elif proc.returncode or fatal:
        raise AssertionError(f"MUSIC failed: {case}/music.log")
    return case


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--music", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--mpirun", default="mpirun")
    ap.add_argument("--reference", type=Path, help="optional pre-fix single-thread L4-6 case")
    args = ap.parse_args()
    # The explicit invalid-path test must not leave MPI process core dumps.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    args.music, args.root = args.music.resolve(), args.root.resolve()
    args.root.mkdir(exist_ok=False)
    report = {"binary_sha256": hashlib.sha256(args.music.read_bytes()).hexdigest(),
              "field_tolerance": TOL, "production_release": False, "pairs": {}}
    for base, group in ((4, "wrapped"), (5, "normal")):
        reference = run_case(args, f"{group}_r1", base, 1)
        trial = run_case(args, f"{group}_r4", base, 4)
        mpi_log = (trial / "music.log").read_text()
        if group == "wrapped" and "periodic writeback alias" not in mpi_log:
            raise AssertionError("missing automatic RNG fallback")
        if group == "normal" and "running SPMD pass" not in mpi_log:
            raise AssertionError("normal geometry unnecessarily lost SPMD")
        targets = [("mpi", trial)]
        if group == "wrapped":
            targets += [("mpi_repeat", run_case(args, "wrapped_r4_repeat", base, 4)),
                        ("legacy_mpi", run_case(args, "wrapped_legacy_r4", base, 4,
                                                extra="slab_rng_refinement = no"))]
            if args.reference:
                targets.append(("pre_fix_phase", args.reference.resolve()))
        for name, target in targets:
            pair = {"noise": {}, "fields": {}}
            for level in range(base, base+3):
                key = f"wnoise_{level:04d}.bin"
                item = compare(read_noise(reference / key), read_noise(target / key))
                pair["noise"][str(level)] = item
                if item["relative_l2"] > 1e-12:
                    raise AssertionError(f"RNG replay changed: {group}/{name}/{level}")
                if name == "pre_fix_phase":
                    continue
                for field in FIELDS:
                    key = f"ic/level_{level:03d}/{field}"
                    item = compare(read_grafic(reference / key), read_grafic(target / key))
                    pair["fields"][f"{level}/{field}"] = item
                    if item["relative_l2"] > TOL:
                        raise AssertionError(f"IC replay changed: {group}/{name}/{key}")
            report["pairs"][f"{group}/{name}"] = pair
            (args.root / "regression.json").write_text(json.dumps(report, indent=2)+"\n")
            print(f"PASS {group}/{name}", flush=True)
    run_case(args, "forced_alias_rejected", 4, 4,
             extra="slab_rng_refinement = yes", expect_reject=True)
    report["forced_alias_rejected"] = True
    report["regression_pass"] = True
    (args.root / "regression.json").write_text(json.dumps(report, indent=2)+"\n")
    print("RNG_PERIODIC_REPLAY=passed (not a production physics release)", flush=True)


if __name__ == "__main__":
    main()
