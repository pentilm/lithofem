#!/usr/bin/env bash
# MFEM build script for LithoFEM (rationale and pitfalls: docs/building.md).
# cuDSS is an application-side pip wheel and does not affect this script.
#
# Usage:
#   tools/build_mfem.sh <mfem-4.9-source-dir> [options]
# Options:
#   --jobs N          parallel jobs (default: nproc-8)
#   --arch sm_XX      CUDA architecture (default sm_90; e.g. sm_80, sm_90, sm_120)
#   --cpu-only        build without CUDA (GPU tests then skip automatically)
#   --full            really compile the two pathologically slow files
#                     (bilininteg_convection_ea did not finish in 12h in our tests)
#   --skip-deps       skip apt dependencies and the OpenBLAS switch (no root, or already done)
#
# Idempotent: safe to re-run; make only rebuilds what is missing when the config is unchanged.

set -euo pipefail

die() { echo "error: $*" >&2; exit 1; }

MFEM_SRC="${1:-}"; [ -n "$MFEM_SRC" ] || die "usage: $0 <mfem-4.9-source-dir> [options]"
[ -f "$MFEM_SRC/makefile" ] || die "$MFEM_SRC is not an MFEM source directory"
shift

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STUB_SRC="$REPO_DIR/solver/mfem_missing_stubs.cpp"
[ -f "$STUB_SRC" ] || die "stub source $STUB_SRC not found (run this from inside the lithofem repository)"

JOBS=$(( $(nproc) > 12 ? $(nproc) - 8 : 4 ))
CUDA_ARCH="sm_90"
USE_CUDA=1
SKIP_SLOW=1
SKIP_DEPS=0
CUDA_DIR="${CUDA_HOME:-/usr/local/cuda}"

while [ $# -gt 0 ]; do
  case "$1" in
    --jobs) JOBS="$2"; shift 2 ;;
    --arch) CUDA_ARCH="$2"; shift 2 ;;
    --cpu-only) USE_CUDA=0; shift ;;
    --full) SKIP_SLOW=0; shift ;;
    --skip-deps) SKIP_DEPS=1; shift ;;
    *) die "unknown option: $1" ;;
  esac
done

echo "== MFEM build: src=$MFEM_SRC jobs=$JOBS cuda=$USE_CUDA arch=$CUDA_ARCH skip_slow=$SKIP_SLOW"

# ---------- 0) system dependencies ----------
if [ "$SKIP_DEPS" -eq 0 ]; then
  echo "== installing SuiteSparse + OpenBLAS and switching the system BLAS"
  apt-get install -y -qq libsuitesparse-dev libopenblas0-pthread libopenblas-dev
  update-alternatives --set libblas.so.3-x86_64-linux-gnu \
    /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 || true
  update-alternatives --set liblapack.so.3-x86_64-linux-gnu \
    /usr/lib/x86_64-linux-gnu/openblas-pthread/liblapack.so.3 || true
fi
[ -f /usr/include/suitesparse/umfpack.h ] || die "libsuitesparse-dev missing (umfpack.h not found)"

# ---------- 1) make config ----------
# Pitfall 1: an inherited MFEM_DIR makes the makefile look for defaults.mk in the
# wrong place -> always use env -u MFEM_DIR.
# Pitfall 2: when nvcc is not on PATH, config silently picks the clang-CUDA flag
# branch -> pass CUDA_CXX as an explicit full path.
SS_LIB="-lklu -lbtf -lumfpack -lcholmod -lcolamd -lamd -lcamd -lccolamd -lsuitesparseconfig -lrt -llapack -lblas"
cd "$MFEM_SRC"
if [ "$USE_CUDA" -eq 1 ]; then
  [ -x "$CUDA_DIR/bin/nvcc" ] || die "nvcc not found at $CUDA_DIR/bin/nvcc (set CUDA_HOME)"
  env -u MFEM_DIR make config \
    MFEM_USE_CUDA=YES CUDA_CXX="$CUDA_DIR/bin/nvcc" CUDA_ARCH="$CUDA_ARCH" \
    MFEM_USE_SUITESPARSE=YES \
    SUITESPARSE_OPT="-I/usr/include/suitesparse" \
    SUITESPARSE_LIB="$SS_LIB" >/dev/null
else
  env -u MFEM_DIR make config \
    MFEM_USE_SUITESPARSE=YES \
    SUITESPARSE_OPT="-I/usr/include/suitesparse" \
    SUITESPARSE_LIB="$SS_LIB" >/dev/null
fi

grep -q "MFEM_USE_SUITESPARSE *= *YES" config/config.mk || die "config did not enable SUITESPARSE"
if [ "$USE_CUDA" -eq 1 ]; then
  grep -q "MFEM_USE_CUDA *= *YES" config/config.mk || die "config did not enable CUDA"
  grep -q -- "-x=cu" config/config.mk || die "CXXFLAGS are not nvcc-style (likely the clang-CUDA branch; check CUDA_CXX)"
fi
echo "== configuration verified"

# ---------- 2) optional: stage stand-in objects for the slow files (after make config) ----------
# bilininteg_convection_ea.cpp does not finish compiling in a practical time on
# A stub object of the same name (providing ConvectionIntegrator::AssembleEA as
# MFEM_ABORT) is compiled in its place, so make treats it as already built.
# lor_batched.cpp takes hours and has no referenced symbols; an empty TU suffices.
if [ "$SKIP_SLOW" -eq 1 ]; then
  echo "== staging stand-in objects for the slow files (see docs/building.md)"
  if [ "$USE_CUDA" -eq 1 ]; then
    NVCC_FLAGS="-O1 -std=c++17 -x=cu --expt-extended-lambda --expt-relaxed-constexpr -arch=$CUDA_ARCH -isystem $CUDA_DIR/include -ccbin g++"
    CC_STUB=("$CUDA_DIR/bin/nvcc" $NVCC_FLAGS)
  else
    CC_STUB=(g++ -O2 -std=c++17)
  fi
  "${CC_STUB[@]}" -I. -I/usr/include/suitesparse \
    -c "$STUB_SRC" -o fem/integ/bilininteg_convection_ea.o
  TMP_EMPTY="$(mktemp --suffix=.cpp)"
  echo '// intentionally empty: LOR-batched kernels unused by LithoFEM (D-017)' > "$TMP_EMPTY"
  "${CC_STUB[@]}" -c "$TMP_EMPTY" -o fem/lor/lor_batched.o
  rm -f "$TMP_EMPTY"
  touch fem/integ/bilininteg_convection_ea.o fem/lor/lor_batched.o
fi

# ---------- 3) build ----------
echo "== make -j $JOBS (roughly 30-60 min on a many-core box; fem/tmop/tools/* dominate)"
env -u MFEM_DIR make -j "$JOBS"

# ---------- 4) verify ----------
nm -C libmfem.a | grep -q "ComplexUMFPackSolver::SetOperator" \
  || die "libmfem.a lacks ComplexUMFPackSolver (SuiteSparse integration failed)"
ar t libmfem.a | grep -q "bilininteg_convection_ea.o" \
  || die "archive is missing bilininteg_convection_ea.o"
echo "== done: $MFEM_SRC/libmfem.a"
echo "   next: run make solver && make test in the lithofem repository (point MFEM_DIR at $MFEM_SRC)"
if [ "$USE_CUDA" -eq 1 ]; then
  python3 -c "import importlib.util as u; exit(0 if u.find_spec('nvidia') else 1)" 2>/dev/null \
    && ls /opt/conda/lib/python3.1*/site-packages/nvidia/cu13/include/cudss.h >/dev/null 2>&1 \
    || echo "   note: the GPU direct solver also needs the cuDSS wheel: pip install nvidia-cudss-cu13==0.8.0.10"
fi
