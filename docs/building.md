# Building LithoFEM

LithoFEM has a Python layer (configuration, meshing, post-processing) and a C++
solver core built on MFEM. The Python side installs with pip; the C++ side needs
an MFEM build with specific options.

## 1. Python package

```bash
pip install -e .            # numpy, scipy, gmsh, shapely, h5py, pyyaml
pip install -e ".[dev]"     # additionally pytest, ruff, mypy, pyvista
```

Python ≥ 3.11 is required.

## 2. System dependencies

```bash
apt-get install -y libsuitesparse-dev libopenblas0-pthread libopenblas-dev

# Point the system BLAS/LAPACK at OpenBLAS. Without this the UMFPACK dense
# kernels run single-threaded and factorization is an order of magnitude slower.
update-alternatives --set libblas.so.3-x86_64-linux-gnu \
    /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3
update-alternatives --set liblapack.so.3-x86_64-linux-gnu \
    /usr/lib/x86_64-linux-gnu/openblas-pthread/liblapack.so.3
```

Gmsh arrives through pip, but its shared library needs OpenGL/X stubs even in
headless use:

```bash
apt-get install -y libgl1 libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1
```

## 3. MFEM

**A default MFEM build will not work.** Without SuiteSparse, the complex sparse
direct solver LithoFEM relies on (`ComplexUMFPackSolver`) is compiled out
entirely — the interface does not even exist.

| Option | Value | Why |
|---|---|---|
| MFEM version | 4.9 (source build via GNU make) | all validation was done against this version; its Gmsh reader behaviour and complex FEM API are what LithoFEM targets |
| `MFEM_USE_SUITESPARSE` | **YES (required)** | provides the complex sparse direct solver |
| `MFEM_USE_CUDA` + `CUDA_ARCH` | YES + your architecture | needed for the GPU paths; CPU-only machines can set NO and the GPU tests skip automatically |
| `CUDA_CXX` | explicit full path, e.g. `/usr/local/cuda/bin/nvcc` | see pitfall 2 below |
| `MFEM_USE_MPI` | NO | LithoFEM is single-node; MPI adds hypre/METIS for no benefit here |

The convenience script encodes everything below:

```bash
tools/build_mfem.sh /path/to/mfem-4.9 --arch sm_90     # or --cpu-only
```

Manual equivalent:

```bash
cd /path/to/mfem-4.9

env -u MFEM_DIR make config \
  MFEM_USE_CUDA=YES \
  CUDA_CXX=/usr/local/cuda/bin/nvcc \
  CUDA_ARCH=sm_90 \
  MFEM_USE_SUITESPARSE=YES \
  SUITESPARSE_OPT="-I/usr/include/suitesparse" \
  SUITESPARSE_LIB="-lklu -lbtf -lumfpack -lcholmod -lcolamd -lamd -lcamd -lccolamd -lsuitesparseconfig -lrt -llapack -lblas"

env -u MFEM_DIR make -j $(( $(nproc) - 8 ))
```

After configuring, `config/config.mk` must contain `MFEM_USE_SUITESPARSE = YES`,
`MFEM_USE_CUDA = YES`, `MFEM_CXX = <path>/nvcc`, and nvcc-style flags
(`-x=cu ... -arch=sm_XX -ccbin g++`) — *not* the clang-CUDA style
(`-xcuda --cuda-gpu-arch=...`).

### Pitfalls

1. **Always use `env -u MFEM_DIR`.** If `MFEM_DIR` points at an older install,
   `make config` looks for `defaults.mk` in the wrong place and fails.
2. **`CUDA_CXX` must be an explicit full path.** `nvcc` is often not on `PATH`;
   when it is missing, `make config` silently falls back to the clang-CUDA flag
   branch and produces a build incompatible with an nvcc-built application.
3. **Any config change is a full rebuild.** `make config` rewrites
   `config/_config.hpp`, which every source file includes. Get the flags right
   the first time.
4. **Build time.** Aside from the two files below, a parallel build takes
   roughly 30–60 minutes; the template-heavy `fem/tmop/tools/*` files dominate.

### Two pathologically slow files

`fem/lor/lor_batched.cpp` and `fem/integ/bilininteg_convection_ea.cpp` compile
extremely slowly under nvcc on recent architectures — the second one did not
finish in 12 hours in our measurements, at any optimization level.

LithoFEM uses neither (they serve convection problems and LOR preconditioning,
which a frequency-domain Maxwell solve never reaches). `tools/build_mfem.sh`
therefore stages stand-in object files before `make -j`, so make treats them as
already built. The stub for `ConvectionIntegrator::AssembleEA` is an
`MFEM_ABORT`: the symbol exists so linking succeeds, and if anything ever did
reach it the program would stop loudly rather than compute silently wrong
results. Pass `--full` to compile them for real.

## 4. LithoFEM solver binaries

```bash
make solver     # builds all binaries into solver/bin
make test       # full test suite (builds the solver first)
make test-fast  # quick tier only
```

`MFEM_DIR` at the top of the `Makefile` must point at your MFEM build.

## 5. cuDSS (GPU direct solver)

cuDSS is an application-side dependency; MFEM needs no changes for it.

```bash
pip install nvidia-cudss-cu13==0.8.0.10   # CUDA 13 wheel; use -cu12 on CUDA 12 machines
make solver                                # the Makefile auto-detects the wheel
```

The Makefile probes for the wheel and enables `-DLITHOFEM_HAVE_CUDSS`
automatically; override the location with `make solver CUDSS_DIR=/path/to/cu13`.
Without the wheel, LithoFEM still builds and `device: gpu` logs a message and
falls back to the CPU solver.

Two packaging details the Makefile already handles — preserve them if you change
the build logic:

1. The wheel's `include/` directory also ships a **complete CUDA toolkit header
   tree** (a pip dependency). Never pass that directory to nvcc with `-isystem`:
   mixing it with nvcc's own headers makes the preprocessor hang. Only the
   `cudss*.h` headers are symlinked into a private include directory.
2. MFEM's CXXFLAGS contain `-x=cu`, so a full `.so` path on the command line
   would be treated as CUDA *source*. The wheel also ships only
   `libcudss.so.0`, with no `.so` development symlink. The Makefile creates the
   symlink in a private directory and links via `-L/-l`, with the runtime SONAME
   resolved through an rpath.

## 6. GPU architectures and portability

The CUDA assembly kernels use only standard CUDA C++ (fp64 arithmetic, shared
memory, `atomicAdd(double)` — sm_60 and newer). There are no
architecture-specific code paths; the target is chosen entirely by the build
flag.

For a single architecture, build MFEM with `CUDA_ARCH=sm_XX` and LithoFEM
inherits it. For a binary that runs across GPU generations, replace the single
`-arch` with a `-gencode` list in CXXFLAGS:

```bash
-gencode arch=compute_80,code=sm_80 \
-gencode arch=compute_90,code=sm_90 \
-gencode arch=compute_120,code=sm_120 \
-gencode arch=compute_120,code=compute_120   # PTX fallback, JIT for future GPUs
```

Note that cubins are architecture-specific: binaries built for one architecture
will not run on another. After moving to a different GPU generation, rebuild and
re-run `pytest tests/test_asm_v25.py` — its criteria are hard numerical
thresholds and are architecture independent.

## 7. Verifying an installation

```bash
lithofem selftest            # quick tier
lithofem selftest --full     # everything, including the long validation cases
```

GPU-marked tests skip automatically when no CUDA device or no cuDSS is present.
