# LithoFEM top-level Makefile (local CI entry point: `make test`)

MFEM_DIR := /workspace/mfem-4.9
CONFIG_MK := $(MFEM_DIR)/config/config.mk

PYTEST ?= python3 -m pytest

.PHONY: test test-fast solver hello lint clean

# Full local CI: python unit tests (solver binaries are exercised via pytest wrappers)
test: solver
	$(PYTEST) -q

test-fast: solver
	$(PYTEST) -q -m fast

# --- C++ solver binaries -------------------------------------------------
include $(CONFIG_MK)

SOLVER_BIN := solver/bin

solver: hello mms pml main dipole asm

hello: $(SOLVER_BIN)/hello_mfem
mms: $(SOLVER_BIN)/mms_test
pml: $(SOLVER_BIN)/pml_test
main: $(SOLVER_BIN)/lithofem_solve
dipole: $(SOLVER_BIN)/dipole_test
asm: $(SOLVER_BIN)/asm_test

# MFEM_STUBS: temporary stubs while two never-used MFEM objects are absent
# from libmfem.a (see solver/mfem_missing_stubs.cpp); cleared automatically
# once the real objects are archived.
MFEM_STUBS := $(shell ar t $(MFEM_DIR)/libmfem.a 2>/dev/null | grep -q bilininteg_convection_ea.o || echo solver/mfem_missing_stubs.cpp)

# cuDSS (v2 GPU direct solve): auto-enabled when the pip wheel is present
# (see docs/gpu.md). The wheel ships only libcudss.so.0 (no .so dev link), so the
# library is passed to the linker by full path. The wheel's include dir also
# carries a FULL CUDA 13.3 toolkit header tree (pip dep) that must NOT be
# mixed with nvcc's own CUDA headers — only the cudss*.h headers are staged
# into a private include dir.
CUDSS_DIR ?= $(firstword $(wildcard /opt/conda/lib/python3.1[0-9]/site-packages/nvidia/cu13))
ifneq ($(wildcard $(CUDSS_DIR)/include/cudss.h),)
CUDSS_INC   := $(SOLVER_BIN)/.cudss_include
CUDSS_LIB   := $(SOLVER_BIN)/.cudss_lib
CUDSS_FLAGS  = -DLITHOFEM_HAVE_CUDSS -isystem $(CUDSS_INC)
# -x=cu is active in MFEM_FLAGS: the library must go through -L/-l (a full
# .so path on the command line would be treated as CUDA source). The dev
# symlink lives in CUDSS_LIB; the runtime SONAME resolves via the rpath.
CUDSS_LIBS  := -L$(CUDSS_LIB) -lcudss -Xlinker=-rpath,$(CUDSS_DIR)/lib
CUDSS_STAGE := $(CUDSS_INC)/cudss.h $(CUDSS_LIB)/libcudss.so
else
CUDSS_INC   :=
CUDSS_LIB   :=
CUDSS_FLAGS :=
CUDSS_LIBS  :=
CUDSS_STAGE :=
endif

$(CUDSS_INC)/cudss.h: $(CUDSS_DIR)/include/cudss.h
	@mkdir -p $(CUDSS_INC)
	ln -sf $(CUDSS_DIR)/include/cudss*.h $(CUDSS_INC)/

$(CUDSS_LIB)/libcudss.so: $(CUDSS_DIR)/lib/libcudss.so.0
	@mkdir -p $(CUDSS_LIB)
	ln -sf $(CUDSS_DIR)/lib/libcudss.so.0 $(CUDSS_LIB)/libcudss.so

$(SOLVER_BIN)/%: solver/%.cpp $(MFEM_DIR)/libmfem.a $(CUDSS_STAGE)
	@mkdir -p $(SOLVER_BIN)
	$(MFEM_CXX) $(MFEM_FLAGS) $(CUDSS_FLAGS) -Xcompiler=-Wall,-Wextra $< $(MFEM_STUBS) -o $@ $(MFEM_LIBS) $(CUDSS_LIBS)

$(SOLVER_BIN)/lithofem_solve: solver/cudss_solver.inc solver/problem_common.inc solver/asm_data.inc solver/asm_gpu.inc
$(SOLVER_BIN)/asm_test: solver/problem_common.inc solver/asm_data.inc solver/asm_gpu.inc

lint:
	ruff check src tests tools
	mypy src

clean:
	rm -rf $(SOLVER_BIN)
