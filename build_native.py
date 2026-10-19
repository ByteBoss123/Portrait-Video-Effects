"""Build with MSVC (Developer PowerShell), g++, or clang++."""
from pathlib import Path
import platform
import shutil
import subprocess

root = Path(__file__).resolve().parent
out = root / 'portraitfx' / ('portrait_native.dll' if platform.system() == 'Windows' else 'portrait_native.so')
source = root / 'native' / 'composite.cpp'
if platform.system() == 'Windows' and shutil.which('cl'):
    cmd = ['cl', '/nologo', '/O2', '/fp:precise', '/LD', '/EHsc', str(source), '/link', '/OUT:' + str(out)]
else:
    compiler = shutil.which('g++') or shutil.which('clang++')
    if not compiler:
        raise SystemExit('Install Visual Studio Build Tools with Desktop development with C++, then run in Developer PowerShell; or install g++/clang++.')
    cmd = [compiler, '-std=c++17', '-O3', '-ffp-contract=off', '-shared', str(source), '-o', str(out)]
    if platform.system() != 'Windows': cmd.insert(1, '-fPIC')
subprocess.run(cmd, check=True, cwd=root / 'native')
print(out)
