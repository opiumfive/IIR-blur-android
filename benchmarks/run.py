#!/usr/bin/env python3
"""Build and run the actual blur core; no Gradle or app UI in the timings."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "6f1eda35454634912006676eabbbc4388e50044d"
PREVIOUS = "455f960"  # first optimized NEON version (fp32 strips), kept as a reference point

def run(args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, **kwargs)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", action="store_true", help="Run locally, without the Android-only baseline")
    parser.add_argument("--scalar", action="store_true", help="Force the portable backend")
    parser.add_argument("--sanitize", action="store_true", help="Host AddressSanitizer and UndefinedBehaviorSanitizer")
    parser.add_argument("--tsan", action="store_true", help="Host ThreadSanitizer")
    parser.add_argument("--ndk", type=Path)
    parser.add_argument("--serial", default=os.environ.get("ANDROID_SERIAL"))
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="-- --test, or -- [--only fast,draft] [--gap 16] width height sigma rounds")
    opt = parser.parse_args()
    if (opt.sanitize or opt.tsan) and not opt.host:
        parser.error("sanitizers are supported with --host")
    if opt.sanitize and opt.tsan:
        parser.error("choose one sanitizer")
    suffix = "host" if opt.host else "android"
    suffix += "-scalar" if opt.scalar else ""
    suffix += "-asan" if opt.sanitize else "-tsan" if opt.tsan else ""
    out = ROOT / "build" / "blur-bench" / suffix
    out.mkdir(parents=True, exist_ok=True)
    flags = ["-std=c++14", "-pthread", "-g"]
    if opt.host:
        compiler = os.environ.get("CXX", "clang++")
        if opt.sanitize:
            flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        if opt.tsan:
            flags += ["-fsanitize=thread", "-fno-omit-frame-pointer"]
    else:
        sdk = os.environ.get("ANDROID_SDK_ROOT") or os.environ.get("ANDROID_HOME")
        if not sdk:
            properties = (ROOT / "local.properties").read_text()
            sdk = re.search(r"^sdk.dir=(.+)$", properties, re.M).group(1).replace("\\ ", " ")
        sdk = Path(sdk)
        ndk = opt.ndk or Path(os.environ.get("ANDROID_NDK_HOME", sdk / "ndk" / "21.4.7075529"))
        host = "darwin-x86_64" if sys.platform == "darwin" else "linux-x86_64"
        compiler = ndk / "toolchains" / "llvm" / "prebuilt" / host / "bin" / "aarch64-linux-android21-clang++"

    objects = []
    if not opt.host:
        # Keep the original optimizer flags, allocations and threading policy.
        # Only rename the colliding ThreadPool types and expose a raw entry point.
        for label, filename in [("legacy", "IIRBlurNeon.cpp"), ("fp16", "IIRBlurNeonFp16.cpp")]:
            source = subprocess.check_output(["git", "show", f"{BASELINE}:app/jni/{filename}"], cwd=ROOT, text=True)
            source = source.split('extern "C"')[0].replace("ThreadPool", label + "ThreadPool")
            source += f'\nextern "C" void {label}(unsigned char* p,unsigned w,unsigned h,float sigma) {{ iir_gauss_blur_u8_rgba_parallel(p,nullptr,w,h,sigma,0.f); }}\n'
            path = out / (label + ".cpp")
            path.write_text(source)
            obj = out / (label + ".o")
            run([compiler, *flags, "-Os", "-ffast-math", "-funroll-loops", "-fno-strict-aliasing", "-c", path, "-o", obj])
            objects.append(obj)
        # The previous optimized core is compiled unchanged under another namespace.
        source = subprocess.check_output(["git", "show", f"{PREVIOUS}:app/jni/IIRBlurFast.cpp"], cwd=ROOT, text=True)
        source = source.replace('#include "IIRBlur.h"', "#include <cstddef>\n#include <cstdint>\nnamespace iirblur_prev { bool blur(uint8_t *, unsigned, unsigned, size_t, float); }")
        source = source.replace("namespace iirblur {", "namespace iirblur_prev {").replace("} // namespace iirblur", "} // namespace iirblur_prev")
        path = out / "previous.cpp"
        path.write_text(source)
        obj = out / "previous.o"
        run([compiler, *flags, "-O3", "-ffp-contract=fast", "-c", path, "-o", obj])
        objects.append(obj)
        flags += ["-DIIRBLUR_HAVE_BASELINE"]
    if opt.scalar:
        flags += ["-DIIRBLUR_FORCE_SCALAR"]
    binary = out / "blur-bench"
    run([compiler, *flags, *([] if opt.host else ["-static-libstdc++"]), "-O3", "-ffp-contract=fast", ROOT / "benchmarks" / "blur_bench.cpp", ROOT / "app" / "jni" / "IIRBlurFast.cpp", *objects, "-o", binary])
    if opt.build_only:
        print(binary)
        return
    args = opt.args[1:] if opt.args[:1] == ["--"] else opt.args
    if not args:
        args = ["--test"]
    if opt.host:
        run([binary, *args])
    else:
        adb = shutil.which("adb") or sdk / "platform-tools" / "adb"
        serial = opt.serial
        if not serial:
            devices = subprocess.check_output([str(adb), "devices"], text=True)
            ready = [line.split()[0] for line in devices.splitlines()[1:] if line.split()[1:] == ["device"]]
            if len(ready) != 1:
                parser.error("select a connected device using --serial")
            serial = ready[0]
        target = "/data/local/tmp/iirblur-bench-" + suffix
        run([adb, "-s", serial, "push", binary, target], stdout=sys.stderr)
        # Benchmark arguments are parsed numerically, except the two fixed test modes.
        import shlex
        run([adb, "-s", serial, "shell", shlex.join([target, *args])])

if __name__ == "__main__":
    main()
