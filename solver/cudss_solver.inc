// CuDssDirectSolver: full-GPU complex direct solve via NVIDIA cuDSS (v2,
// DESIGN V2 §2). Included by lithofem_solve.cpp inside its anonymous
// namespace, after ILinearSolver. Compiled only with -DLITHOFEM_HAVE_CUDSS.
//
// Contract: Solve() returns false on ANY cuDSS/CUDA failure (after logging);
// the caller falls back to UMFPACK. First Solve() factorizes and caches the
// factors on the device; later calls (per_source RHS sweeps) only run the
// backsolve. MTYPE is always GENERAL: the Bloch-envelope system is NOT
// complex-symmetric for kpar != 0 (see docs/gpu.md).

#ifdef LITHOFEM_HAVE_CUDSS

#define LFEM_CUDA_OK(call)                                                   \
   do {                                                                      \
      cudaError_t e_ = (call);                                               \
      if (e_ != cudaSuccess) {                                               \
         std::printf("cudss: CUDA error '%s' at %s:%d\n",                    \
                     cudaGetErrorString(e_), __FILE__, __LINE__);            \
         return false;                                                       \
      }                                                                      \
   } while (0)

#define LFEM_CUDSS_OK(call)                                                  \
   do {                                                                      \
      cudssStatus_t s_ = (call);                                             \
      if (s_ != CUDSS_STATUS_SUCCESS) {                                      \
         std::printf("cudss: cuDSS error %d at %s:%d\n", (int)s_,            \
                     __FILE__, __LINE__);                                    \
         return false;                                                       \
      }                                                                      \
   } while (0)

class CuDssDirectSolver : public ILinearSolver
{
public:
   // mem_limit_gb: VRAM cap for the factors (V2-M3). 0 = auto (free VRAM
   // with a 10% safety margin at analysis time).
   explicit CuDssDirectSolver(int gpu_id, double mem_limit_gb = 0.0)
      : gpu_id_(gpu_id), mem_limit_gb_(mem_limit_gb) {}
   ~CuDssDirectSolver() override { Release(); }

   bool Solve(ComplexSparseMatrix &A, const Vector &B, Vector &X) override
   {
      if (std::getenv("LITHOFEM_CUDSS_FORCE_FAIL"))
      {
         std::printf("cudss: forced failure injected "
                     "(LITHOFEM_CUDSS_FORCE_FAIL)\n");
         return false;
      }
      if (failed_) { return false; }   // don't retry a dead device path
      const int n = A.real().Height();
      if (!ready_)
      {
         if (!Factor(A))
         {
            failed_ = true;
            Release();
            return false;
         }
         ready_ = true;
      }
      else
      {
         std::printf("cudss: factorization reused\n");
      }
      if (!Backsolve(n, B, X))
      {
         failed_ = true;
         Release();
         return false;
      }
      return true;
   }

   // v2.5 zero-copy entry: the CSR (row ptr / col idx / interleaved complex
   // values) already lives on the device (GPU assembly, asm_gpu.inc). The
   // pointers are borrowed, not owned; factors are cached exactly like the
   // host-matrix path (per_source backsolves reuse them).
   bool SolveDeviceCsr(int n, int nnz, int *d_rp, int *d_ci, void *d_val,
                       const Vector &B, Vector &X)
   {
      if (std::getenv("LITHOFEM_CUDSS_FORCE_FAIL"))
      {
         std::printf("cudss: forced failure injected "
                     "(LITHOFEM_CUDSS_FORCE_FAIL)\n");
         return false;
      }
      if (failed_) { return false; }
      if (!ready_)
      {
         if (!FactorDevice(n, nnz, d_rp, d_ci, d_val))
         {
            failed_ = true;
            Release();
            return false;
         }
         ready_ = true;
      }
      else
      {
         std::printf("cudss: factorization reused\n");
      }
      if (!Backsolve(n, B, X))
      {
         failed_ = true;
         Release();
         return false;
      }
      return true;
   }
   const char *Name() const override { return "cudss-direct"; }

private:
   bool Factor(ComplexSparseMatrix &A)
   {
      StopWatch sw;
      sw.Start();
      const SparseMatrix &Ar = A.real();
      const SparseMatrix &Ai = A.imag();
      const int n = Ar.Height();
      const int nnz = Ar.NumNonZeroElems();
      if (Ai.NumNonZeroElems() != nnz)
      {
         std::printf("cudss: re/im sparsity mismatch (%d vs %d nnz)\n",
                     nnz, Ai.NumNonZeroElems());
         return false;
      }
      const int *I = Ar.HostReadI();
      const int *J = Ar.HostReadJ();
      const double *vr = Ar.HostReadData();
      const double *vi = Ai.HostReadData();
      const int *Ii = Ai.HostReadI();
      const int *Ji = Ai.HostReadJ();
      if (std::memcmp(I, Ii, (n + 1) * sizeof(int)) != 0 ||
          std::memcmp(J, Ji, (size_t)nnz * sizeof(int)) != 0)
      {
         std::printf("cudss: re/im sparsity patterns differ; cannot pack\n");
         return false;
      }

      // Interleaved complex CSR with per-row sorted columns (cuDSS wants
      // sorted CSR; MFEM's assembled CSR is unsorted).
      std::vector<int> col(J, J + nnz);
      std::vector<std::complex<double>> val(nnz);
      for (int k = 0; k < nnz; ++k) { val[k] = {vr[k], vi[k]}; }
      for (int r = 0; r < n; ++r)
      {
         const int b = I[r], e = I[r + 1];
         for (int i = b + 1; i < e; ++i)   // rows are short: insertion sort
         {
            const int c = col[i];
            const std::complex<double> v = val[i];
            int j = i - 1;
            while (j >= b && col[j] > c)
            {
               col[j + 1] = col[j];
               val[j + 1] = val[j];
               --j;
            }
            col[j + 1] = c;
            val[j + 1] = v;
         }
      }

      LFEM_CUDA_OK(cudaSetDevice(gpu_id_));
      {
         cudaDeviceProp prop;
         LFEM_CUDA_OK(cudaGetDeviceProperties(&prop, gpu_id_));
         std::printf("cudss_device %d %s (sm_%d%d)\n", gpu_id_, prop.name,
                     prop.major, prop.minor);
      }
      owns_csr_ = true;
      LFEM_CUDA_OK(cudaMalloc(&d_rp_, (size_t)(n + 1) * sizeof(int)));
      LFEM_CUDA_OK(cudaMalloc(&d_ci_, (size_t)nnz * sizeof(int)));
      LFEM_CUDA_OK(cudaMalloc(&d_val_, (size_t)nnz * 16));
      LFEM_CUDA_OK(cudaMemcpy(d_rp_, I, (size_t)(n + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));
      LFEM_CUDA_OK(cudaMemcpy(d_ci_, col.data(), (size_t)nnz * sizeof(int),
                              cudaMemcpyHostToDevice));
      LFEM_CUDA_OK(cudaMemcpy(d_val_, val.data(), (size_t)nnz * 16,
                              cudaMemcpyHostToDevice));
      return FactorCore(n, nnz, sw.RealTime());
   }

   // v2.5: factorize an externally owned device CSR (no host copy, no upload)
   bool FactorDevice(int n, int nnz, int *d_rp, int *d_ci, void *d_val)
   {
      StopWatch sw;
      sw.Start();
      LFEM_CUDA_OK(cudaSetDevice(gpu_id_));
      {
         cudaDeviceProp prop;
         LFEM_CUDA_OK(cudaGetDeviceProperties(&prop, gpu_id_));
         std::printf("cudss_device %d %s (sm_%d%d)\n", gpu_id_, prop.name,
                     prop.major, prop.minor);
      }
      owns_csr_ = false;
      d_rp_ = d_rp;
      d_ci_ = d_ci;
      d_val_ = d_val;
      std::printf("cudss: device-CSR zero-copy path (GPU assembly)\n");
      return FactorCore(n, nnz, sw.RealTime());
   }

   bool FactorCore(int n, int nnz, double t_pack)
   {
      StopWatch sw;
      sw.Start();
      LFEM_CUDA_OK(cudaMalloc(&d_b_, (size_t)n * 16));
      LFEM_CUDA_OK(cudaMalloc(&d_x_, (size_t)n * 16));
      LFEM_CUDSS_OK(cudssCreate(&handle_));
      have_handle_ = true;
      LFEM_CUDSS_OK(cudssConfigCreate(&config_));
      have_config_ = true;
      LFEM_CUDSS_OK(cudssDataCreate(handle_, &data_));
      have_data_ = true;
      LFEM_CUDSS_OK(cudssMatrixCreateCsr(
         &mA_, n, n, nnz, d_rp_, nullptr, d_ci_, d_val_, CUDSS_R_32I,
         CUDSS_R_32I, CUDSS_C_64F, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL,
         CUDSS_BASE_ZERO));
      have_mA_ = true;
      LFEM_CUDSS_OK(cudssMatrixCreateDn(&mb_, n, 1, n, d_b_, CUDSS_C_64F,
                                        CUDSS_LAYOUT_COL_MAJOR));
      have_mb_ = true;
      LFEM_CUDSS_OK(cudssMatrixCreateDn(&mx_, n, 1, n, d_x_, CUDSS_C_64F,
                                        CUDSS_LAYOUT_COL_MAJOR));
      have_mx_ = true;
      const double t_setup = t_pack + sw.RealTime();

      sw.Clear();
      sw.Start();
      LFEM_CUDSS_OK(cudssExecute(handle_, CUDSS_PHASE_ANALYSIS, config_,
                                 data_, mA_, mx_, mb_));
      LFEM_CUDA_OK(cudaDeviceSynchronize());
      const double t_analysis = sw.RealTime();

      int64_t mem_est[16] = {0};
      size_t swr = 0;
      cudssDataGet(handle_, data_, CUDSS_DATA_MEMORY_ESTIMATES, mem_est,
                   sizeof(mem_est), &swr);
      const double est_gb = mem_est[1] / 1e9;
      std::printf("cudss_peak_est_gb %.3f\n", est_gb);

      // V2-M3 VRAM guard: refuse (-> CPU fallback) when the analysis-phase
      // estimate exceeds the cap. design note: estimate vs measured within 7%.
      size_t free_b = 0, total_b = 0;
      LFEM_CUDA_OK(cudaMemGetInfo(&free_b, &total_b));
      const double vram_before_gb = (total_b - free_b) / 1e9;
      const double limit_gb =
         mem_limit_gb_ > 0.0 ? mem_limit_gb_ : 0.9 * free_b / 1e9;
      if (est_gb > limit_gb)
      {
         std::printf("GPU MEMORY LIMIT: cudss estimates %.2f GB for the "
                     "factors, %s limit is %.2f GB; falling back to CPU\n",
                     est_gb, mem_limit_gb_ > 0.0 ? "configured" : "free-VRAM",
                     limit_gb);
         return false;
      }

      sw.Clear();
      sw.Start();
      LFEM_CUDSS_OK(cudssExecute(handle_, CUDSS_PHASE_FACTORIZATION, config_,
                                 data_, mA_, mx_, mb_));
      LFEM_CUDA_OK(cudaDeviceSynchronize());
      const double t_factor = sw.RealTime();

      int info = 0;
      cudssDataGet(handle_, data_, CUDSS_DATA_INFO, &info, sizeof(info), &swr);
      if (info != 0)
      {
         std::printf("cudss: factorization reported info=%d\n", info);
         return false;
      }
      int64_t lu_nnz = 0;
      cudssDataGet(handle_, data_, CUDSS_DATA_LU_NNZ, &lu_nnz,
                   sizeof(lu_nnz), &swr);
      LFEM_CUDA_OK(cudaMemGetInfo(&free_b, &total_b));
      std::printf("cudss_setup_s %.3f\n", t_setup);
      std::printf("cudss_analysis_s %.3f\n", t_analysis);
      std::printf("cudss_factor_s %.3f\n", t_factor);
      std::printf("cudss_vram_used_gb %.3f\n",
                  (total_b - free_b) / 1e9 - vram_before_gb);
      std::printf("cudss_lu_nnz %lld\n", (long long)lu_nnz);
      return true;
   }

   bool Backsolve(int n, const Vector &B, Vector &X)
   {
      StopWatch sw;
      sw.Start();
      const double *b = B.HostRead();
      zbuf_.resize(n);
      for (int i = 0; i < n; ++i) { zbuf_[i] = {b[i], b[n + i]}; }
      LFEM_CUDA_OK(cudaMemcpy(d_b_, zbuf_.data(), (size_t)n * 16,
                              cudaMemcpyHostToDevice));
      LFEM_CUDSS_OK(cudssExecute(handle_, CUDSS_PHASE_SOLVE, config_, data_,
                                 mA_, mx_, mb_));
      LFEM_CUDA_OK(cudaDeviceSynchronize());
      LFEM_CUDA_OK(cudaMemcpy(zbuf_.data(), d_x_, (size_t)n * 16,
                              cudaMemcpyDeviceToHost));
      double *x = X.HostWrite();
      for (int i = 0; i < n; ++i)
      {
         x[i] = zbuf_[i].real();
         x[n + i] = zbuf_[i].imag();
      }
      std::printf("cudss_solve_s %.3f\n", sw.RealTime());
      return true;
   }

   void Release()
   {
      if (have_mA_) { cudssMatrixDestroy(mA_); have_mA_ = false; }
      if (have_mb_) { cudssMatrixDestroy(mb_); have_mb_ = false; }
      if (have_mx_) { cudssMatrixDestroy(mx_); have_mx_ = false; }
      if (have_data_) { cudssDataDestroy(handle_, data_); have_data_ = false; }
      if (have_config_) { cudssConfigDestroy(config_); have_config_ = false; }
      if (have_handle_) { cudssDestroy(handle_); have_handle_ = false; }
      auto free_dev = [](void *&p)
      { if (p) { cudaFree(p); p = nullptr; } };
      if (owns_csr_)
      {
         free_dev(d_val_);
         if (d_rp_) { cudaFree(d_rp_); }
         if (d_ci_) { cudaFree(d_ci_); }
      }
      d_val_ = nullptr;
      d_rp_ = nullptr;
      d_ci_ = nullptr;
      free_dev(d_b_);
      free_dev(d_x_);
      ready_ = false;
   }

   int gpu_id_ = 0;
   double mem_limit_gb_ = 0.0;
   bool ready_ = false, failed_ = false, owns_csr_ = true;
   bool have_handle_ = false, have_config_ = false, have_data_ = false;
   bool have_mA_ = false, have_mb_ = false, have_mx_ = false;
   cudssHandle_t handle_{};
   cudssConfig_t config_{};
   cudssData_t data_{};
   cudssMatrix_t mA_{}, mb_{}, mx_{};
   int *d_rp_ = nullptr, *d_ci_ = nullptr;
   void *d_val_ = nullptr, *d_b_ = nullptr, *d_x_ = nullptr;
   std::vector<std::complex<double>> zbuf_;
};

#endif // LITHOFEM_HAVE_CUDSS
