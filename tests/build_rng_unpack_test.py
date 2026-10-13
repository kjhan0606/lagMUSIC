#!/usr/bin/env python3
"""Build the direct RNG test using an existing Unix-Makefiles CMake MPI build.

python3 tests/build_rng_unpack_test.py /path/to/build
Then run /path/to/build/test_rng_periodic_unpack on a compute host (one process,
up to four OMP threads). Reuses MUSIC objects except main; rebuilds no library.
"""
import argparse
from pathlib import Path
import shlex
import subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("build", type=Path)
    ap.add_argument("--output", type=Path, help="optional fresh test executable path")
    ap.add_argument("--random-object", type=Path, help="optional pre-fix object for a negative control")
    args = ap.parse_args()
    build = args.build.resolve()
    target = args.output.resolve() if args.output else build / "test_rng_periodic_unpack"
    if target.exists():
        raise FileExistsError(target)
    cmake = build / "CMakeFiles/MUSIC.dir"
    link = shlex.split((cmake / "link.txt").read_text())
    flags = {}
    for line in (cmake / "flags.make").read_text().splitlines():
        if line.startswith("CXX_") and " = " in line:
            key, value = line.split(" = ", 1)
            flags[key] = shlex.split(value)
    source = Path(__file__).with_name("test_rng_periodic_unpack.cc").resolve()
    obj = target.with_suffix(".o")
    compile_command = [link[0]]
    for name in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
        compile_command += flags[name]
    compile_command += ["-c", str(source), "-o", str(obj)]
    subprocess.run(compile_command, cwd=build, check=True, timeout=120)
    main_obj = "CMakeFiles/MUSIC.dir/src/main.cc.o"
    if link.count(main_obj) != 1:
        raise ValueError("unexpected MUSIC main object layout")
    link[link.index(main_obj)] = str(obj)
    if args.random_object:
        link[link.index("CMakeFiles/MUSIC.dir/src/random.cc.o")] = str(args.random_object.resolve())
    link[link.index("-o")+1] = str(target)
    subprocess.run(link, cwd=build, check=True, timeout=120)
    print(target)


if __name__ == "__main__":
    main()
