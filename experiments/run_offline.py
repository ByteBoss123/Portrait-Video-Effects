"""Linux-only network-denied runner for an already-downloaded experiment.
Uses libseccomp to deny network syscalls for this process and its descendants.
Example: python -m experiments.run_offline experiments.train_refiner --manifest ...
"""
import ctypes
import ctypes.util
import errno
import runpy
import socket
import sys

def main():
    if sys.platform != 'linux': raise SystemExit('This offline runner requires Linux and libseccomp')
    if len(sys.argv)<2: raise SystemExit('Specify the experiment module and its arguments')
    library=ctypes.util.find_library('seccomp')
    if not library: raise SystemExit('libseccomp is required; refusing to run without isolation')
    sec=ctypes.CDLL(library, use_errno=True)
    sec.seccomp_init.argtypes=[ctypes.c_uint32]; sec.seccomp_init.restype=ctypes.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p]; sec.seccomp_syscall_resolve_name.restype=ctypes.c_int
    sec.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    sec.seccomp_load.argtypes=[ctypes.c_void_p]
    sec.seccomp_release.argtypes=[ctypes.c_void_p]
    ctx=sec.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW for non-network operations
    if not ctx: raise RuntimeError('seccomp initialization failed')
    try:
        for name in (b'socket',b'connect',b'sendto',b'sendmsg',b'sendmmsg'):
            call=sec.seccomp_syscall_resolve_name(name)
            if call<0 or sec.seccomp_rule_add(ctx,0x00050000 | errno.EACCES,call,0)!=0:
                raise RuntimeError('cannot install network-deny rule')
        if sec.seccomp_load(ctx)!=0: raise RuntimeError('cannot enable network-deny filter')
    finally: sec.seccomp_release(ctx)
    try:
        socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    except PermissionError: print('Verified: network socket creation denied by OS filter',flush=True)
    else: raise RuntimeError('network isolation check failed; refusing to run')
    module=sys.argv[1]; sys.argv=sys.argv[1:]
    runpy.run_module(module,run_name='__main__')

if __name__=='__main__':main()
