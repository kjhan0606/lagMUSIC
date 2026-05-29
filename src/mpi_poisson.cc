// MPI-distributed FFT-Poisson helpers + worker pump. See mpi_poisson.hh.

#include "mpi_poisson.hh"
#include "mpi_helper.hh"
#include "mesh.hh"
#include "mesh_distributed.hh"
#include "mpi_fft.hh"
#include "mg_dist.hh"
#include "poisson_hybrid_kernel.hh"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace MUSIC { namespace poisson {

#ifdef USE_MPI
// meta payload broadcast from rank 0 to workers in worker_pump:
//   [0]=op  [1]=gnx  [2]=gny  [3]=gnz  [4]=dir  [5]=deconvolve_cic
//   [6]=order (H.1 only; poisson_hybrid stencil order 2/4/6)
//   [7]=periodic (H.1 only; bool 0/1)
static const int META_LEN = 8;
#endif

// Diagnostic logging gated by MUSIC_TRACE_PHASE=1 in env.
// Each rank prints to stderr with a global call-step counter so logs from
// rank 0 and workers can be interleaved by step.
static bool trace_phase_enabled()
{
	static int cached = -1;
	if( cached < 0 ){
		const char * e = std::getenv("MUSIC_TRACE_PHASE");
		cached = (e && e[0] && e[0] != '0') ? 1 : 0;
	}
	return cached != 0;
}
static unsigned long g_phase_step = 0;
#define TRACE_PHASE(fmt, ...) do { \
	if( trace_phase_enabled() ){ \
		std::fprintf(stderr, "[trace rk=%d step=%lu] " fmt "\n", \
		             MUSIC::mpi::rank(), ++g_phase_step, ##__VA_ARGS__); \
		std::fflush(stderr); \
	} \
} while(0)

// ---------------------------------------------------------------------------
// Per-rank slab FFT body shared by rank0_dist_* and worker handlers.
// Caller passes root_buf (non-NULL on rank 0) and the operation parameters.
// ---------------------------------------------------------------------------
template<typename real_t>
static void run_dist_fft_op( int op, int dir, real_t* root_buf,
                             size_t gnx, size_t gny, size_t gnz,
                             bool deconvolve_cic )
{
#ifdef USE_MPI
	const int rk = MUSIC::mpi::rank();
	const int sz = MUSIC::mpi::size();

	Meshvar<real_t>* slab = MUSIC::dist::make_slab_meshvar<real_t>(gnx, gny, gnz, /*fftw_inplace_pad=*/true);
	const size_t local_nx    = slab->local_nx();
	const size_t local_x_off = slab->local_x_start();
	const size_t nz_complex  = gnz/2 + 1;
	const size_t nz_padded   = 2*nz_complex;
	const size_t local_count = local_nx * gny * nz_padded;

	std::vector<int> counts(sz), displs(sz);
	int my_count = (int)local_count;
	MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MUSIC::mpi::world());
	displs[0] = 0;
	for( int i=1; i<sz; ++i ) displs[i] = displs[i-1] + counts[i-1];

	MPI_Datatype dtype = (sizeof(real_t) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;

	MPI_Scatterv( (rk==0) ? root_buf : NULL, counts.data(), displs.data(), dtype,
	              slab->m_pdata, my_count, dtype,
	              0, MUSIC::mpi::world() );

	fftw_real    *data  = reinterpret_cast<fftw_real*>(slab->m_pdata);
	fftw_complex *cdata = reinterpret_cast<fftw_complex*>(data);

	MUSIC::fft::fft_plan_t plan  = MUSIC::fft::plan_r2c_3d_mpi(
		(ptrdiff_t)gnx, (ptrdiff_t)gny, (ptrdiff_t)gnz, data);
	MUSIC::fft::fft_plan_t iplan = MUSIC::fft::plan_c2r_3d_mpi(
		(ptrdiff_t)gnx, (ptrdiff_t)gny, (ptrdiff_t)gnz, data);

	MUSIC::fft::execute(plan);

	const double kfac = 2.0*M_PI;
	const double fac  = -1.0/(double)((size_t)gnx*(size_t)gny*(size_t)gnz);

	if( op == OP_SOLVE ){
		// divide by -k^2 over local x-slab (matches serial fft_poisson_plugin::solve)
		#pragma omp parallel for
		for( size_t lix = 0; lix < local_nx; ++lix ){
			const int gix = (int)(local_x_off + lix);
			int ii = gix; if( ii > (int)gnx/2 ) ii -= (int)gnx;
			for( int j = 0; j < (int)gny; ++j ){
				int jj = j; if( jj > (int)gny/2 ) jj -= (int)gny;
				for( int k = 0; k < (int)nz_complex; ++k ){
					double ki = (double)ii, kj = (double)jj, kk = (double)k;
					double kk2 = kfac*kfac*(ki*ki + kj*kj + kk*kk);
					size_t idx = (lix*gny + (size_t)j) * nz_complex + (size_t)k;
					RE(cdata[idx]) *= -1.0/kk2*fac;
					IM(cdata[idx]) *= -1.0/kk2*fac;
				}
			}
		}
	} else if( op == OP_GRADIENT ){
		// multiply by i*k_dir (matches serial fft_poisson_plugin::gradient)
		#pragma omp parallel for
		for( size_t lix = 0; lix < local_nx; ++lix ){
			const int gix = (int)(local_x_off + lix);
			int ii = gix; if( ii > (int)gnx/2 ) ii -= (int)gnx;
			for( int j = 0; j < (int)gny; ++j ){
				int jj = j; if( jj > (int)gny/2 ) jj -= (int)gny;
				for( int k = 0; k < (int)nz_complex; ++k ){
					double ki = (double)ii, kj = (double)jj, kk = (double)k;
					double kkdir[3] = { kfac*ki, kfac*kj, kfac*kk };
					double kdir = kkdir[dir];
					size_t idx = (lix*gny + (size_t)j) * nz_complex + (size_t)k;
					double re = RE(cdata[idx]);
					double im = IM(cdata[idx]);
					RE(cdata[idx]) =  fac*im*kdir;
					IM(cdata[idx]) = -fac*re*kdir;
					if( deconvolve_cic ){
						double dfx, dfy, dfz;
						dfx = M_PI*ki/(double)gnx; dfx = (gix!=0) ? sin(dfx)/dfx : 1.0;
						dfy = M_PI*kj/(double)gny; dfy = (j  !=0) ? sin(dfy)/dfy : 1.0;
						dfz = M_PI*kk/(double)gnz; dfz = (k  !=0) ? sin(dfz)/dfz : 1.0;
						double w = 1.0/(dfx*dfy*dfz); w = w*w;
						RE(cdata[idx]) *= w;
						IM(cdata[idx]) *= w;
					}
				}
			}
		}
	}

	// zero DC mode (only owned by rank with local_x_off==0)
	if( local_x_off == 0 && local_nx > 0 ){
		RE(cdata[0]) = 0.0;
		IM(cdata[0]) = 0.0;
	}

	MUSIC::fft::execute(iplan);
	MUSIC::fft::destroy(plan);
	MUSIC::fft::destroy(iplan);

	MPI_Gatherv( slab->m_pdata, my_count, dtype,
	             (rk==0) ? root_buf : NULL, counts.data(), displs.data(), dtype,
	             0, MUSIC::mpi::world() );

	delete slab;
#else
	(void)op; (void)dir; (void)root_buf; (void)gnx; (void)gny; (void)gnz; (void)deconvolve_cic;
#endif
}

// ---------------------------------------------------------------------------
// Phase E.2.2a: slab-direct FFT body (no Scatterv/Gatherv).
// Reads from src_slab (per-rank slab MeshvarBnd<real_t>*, no FFT padding),
// runs the FFTW MPI plan in a private padded scratch buffer, writes back
// to dst_slab. Both must have local_nx and local_x_start matching the
// FFTW MPI decomposition for (gnx,gny,gnz). All ranks call it; rank 0
// already broadcast OP_SOLVE_SLAB to wake workers.
// ---------------------------------------------------------------------------
template<typename real_t>
static void run_dist_fft_op_slab( int op, int dir,
                                  MeshvarBnd<real_t>* src_slab,
                                  MeshvarBnd<real_t>* dst_slab,
                                  size_t gnx, size_t gny, size_t gnz,
                                  bool deconvolve_cic )
{
#ifdef USE_MPI
	Meshvar<real_t>* scratch = MUSIC::dist::make_slab_meshvar<real_t>(gnx, gny, gnz, /*fftw_inplace_pad=*/true);
	const size_t local_nx    = scratch->local_nx();
	const size_t local_x_off = scratch->local_x_start();
	const size_t nz_complex  = gnz/2 + 1;
	const size_t nz_padded   = 2*nz_complex;

	if( !src_slab || !dst_slab ) {
		LOGERR("run_dist_fft_op_slab: NULL slab pointer (registered via set_slab_solve_inout?)");
		throw std::runtime_error("run_dist_fft_op_slab: NULL slab pointer");
	}
	if( src_slab->local_nx() != local_nx || src_slab->local_x_start() != local_x_off ) {
		LOGERR("run_dist_fft_op_slab: src slab geometry (%zu,%zu) != FFTW MPI (%zu,%zu)",
		       src_slab->local_x_start(), src_slab->local_nx(), local_x_off, local_nx);
		throw std::runtime_error("run_dist_fft_op_slab: src slab geometry mismatch");
	}
	if( dst_slab->local_nx() != local_nx || dst_slab->local_x_start() != local_x_off ) {
		LOGERR("run_dist_fft_op_slab: dst slab geometry (%zu,%zu) != FFTW MPI (%zu,%zu)",
		       dst_slab->local_x_start(), dst_slab->local_nx(), local_x_off, local_nx);
		throw std::runtime_error("run_dist_fft_op_slab: dst slab geometry mismatch");
	}

	// copy src slab -> padded scratch (skip k-pad bytes; mirrors how the
	// rank-0 full path packs nz onto nz_padded in fft_poisson_plugin::solve)
	{
		#pragma omp parallel for
		for( size_t lix = 0; lix < local_nx; ++lix )
			for( size_t j = 0; j < gny; ++j )
				for( size_t k = 0; k < gnz; ++k ) {
					size_t idx = (lix*gny + j) * nz_padded + k;
					scratch->m_pdata[idx] = (*src_slab)((int)lix, (int)j, (int)k);
				}
	}

	fftw_real    *data  = reinterpret_cast<fftw_real*>(scratch->m_pdata);
	fftw_complex *cdata = reinterpret_cast<fftw_complex*>(data);

	MUSIC::fft::fft_plan_t plan  = MUSIC::fft::plan_r2c_3d_mpi(
		(ptrdiff_t)gnx, (ptrdiff_t)gny, (ptrdiff_t)gnz, data);
	MUSIC::fft::fft_plan_t iplan = MUSIC::fft::plan_c2r_3d_mpi(
		(ptrdiff_t)gnx, (ptrdiff_t)gny, (ptrdiff_t)gnz, data);

	MUSIC::fft::execute(plan);

	const double kfac = 2.0*M_PI;
	const double fac  = -1.0/(double)((size_t)gnx*(size_t)gny*(size_t)gnz);

	if( op == OP_SOLVE_SLAB ){
		#pragma omp parallel for
		for( size_t lix = 0; lix < local_nx; ++lix ){
			const int gix = (int)(local_x_off + lix);
			int ii = gix; if( ii > (int)gnx/2 ) ii -= (int)gnx;
			for( int j = 0; j < (int)gny; ++j ){
				int jj = j; if( jj > (int)gny/2 ) jj -= (int)gny;
				for( int k = 0; k < (int)nz_complex; ++k ){
					double ki = (double)ii, kj = (double)jj, kk = (double)k;
					double kk2 = kfac*kfac*(ki*ki + kj*kj + kk*kk);
					size_t idx = (lix*gny + (size_t)j) * nz_complex + (size_t)k;
					RE(cdata[idx]) *= -1.0/kk2*fac;
					IM(cdata[idx]) *= -1.0/kk2*fac;
				}
			}
		}
	} else if( op == OP_GRADIENT_SLAB ){
		// multiply by i*k_dir, with optional CIC deconvolution
		// (matches kernel in run_dist_fft_op / fft_poisson_plugin::gradient)
		#pragma omp parallel for
		for( size_t lix = 0; lix < local_nx; ++lix ){
			const int gix = (int)(local_x_off + lix);
			int ii = gix; if( ii > (int)gnx/2 ) ii -= (int)gnx;
			for( int j = 0; j < (int)gny; ++j ){
				int jj = j; if( jj > (int)gny/2 ) jj -= (int)gny;
				for( int k = 0; k < (int)nz_complex; ++k ){
					double ki = (double)ii, kj = (double)jj, kk = (double)k;
					double kkdir[3] = { kfac*ki, kfac*kj, kfac*kk };
					double kdir = kkdir[dir];
					size_t idx = (lix*gny + (size_t)j) * nz_complex + (size_t)k;
					double re = RE(cdata[idx]);
					double im = IM(cdata[idx]);
					RE(cdata[idx]) =  fac*im*kdir;
					IM(cdata[idx]) = -fac*re*kdir;
					if( deconvolve_cic ){
						double dfx, dfy, dfz;
						dfx = M_PI*ki/(double)gnx; dfx = (gix!=0) ? sin(dfx)/dfx : 1.0;
						dfy = M_PI*kj/(double)gny; dfy = (j  !=0) ? sin(dfy)/dfy : 1.0;
						dfz = M_PI*kk/(double)gnz; dfz = (k  !=0) ? sin(dfz)/dfz : 1.0;
						double w = 1.0/(dfx*dfy*dfz); w = w*w;
						RE(cdata[idx]) *= w;
						IM(cdata[idx]) *= w;
					}
				}
			}
		}
	}

	if( local_x_off == 0 && local_nx > 0 ){
		RE(cdata[0]) = 0.0;
		IM(cdata[0]) = 0.0;
	}

	MUSIC::fft::execute(iplan);
	MUSIC::fft::destroy(plan);
	MUSIC::fft::destroy(iplan);

	// scratch -> dst slab (drop k-pad bytes)
	{
		#pragma omp parallel for
		for( size_t lix = 0; lix < local_nx; ++lix )
			for( size_t j = 0; j < gny; ++j )
				for( size_t k = 0; k < gnz; ++k ) {
					size_t idx = (lix*gny + j) * nz_padded + k;
					(*dst_slab)((int)lix, (int)j, (int)k) = scratch->m_pdata[idx];
				}
	}

	delete scratch;
#else
	(void)op; (void)dir; (void)src_slab; (void)dst_slab; (void)gnx; (void)gny; (void)gnz; (void)deconvolve_cic;
#endif
}

// ---------------------------------------------------------------------------
// Phase F2LPT.1: Scatterv-style distributed compute_2LPT_source_FFT body.
// Mirrors run_dist_fft_op (rank-0 holds the full padded buffers; Scatterv to
// per-rank slabs, FFT, kernel, IFFT, Gatherv back). Two root buffers:
//   phi_root  — input padded buffer (nx * ny * 2*(nz/2+1)) holding potential
//   dst_root  — output padded buffer (same shape) receiving the composite
//                (sum of 2x2 Hessian minors).
// Workers pass NULL/NULL for the root buffers.
// 6 intermediate Hessian-component scratch slabs are allocated per-rank only.
// ---------------------------------------------------------------------------
template<typename real_t>
static void run_dist_fft_op_lpt2( real_t* phi_root, real_t* dst_root,
                                   size_t gnx, size_t gny, size_t gnz )
{
#ifdef USE_MPI
	const int rk = MUSIC::mpi::rank();
	const int sz = MUSIC::mpi::size();

	// scratch[0]=phi (in/out for r2c), scratch[1..6]=h11,h12,h13,h22,h23,h33
	Meshvar<real_t>* scratch[7];
	for( int s=0; s<7; ++s )
		scratch[s] = MUSIC::dist::make_slab_meshvar<real_t>(gnx, gny, gnz, /*fftw_inplace_pad=*/true);

	const size_t local_nx    = scratch[0]->local_nx();
	const size_t local_x_off = scratch[0]->local_x_start();
	const size_t nz_complex  = gnz/2 + 1;
	const size_t nz_padded   = 2*nz_complex;
	const size_t local_count = local_nx * gny * nz_padded;

	std::vector<int> counts(sz), displs(sz);
	int my_count = (int)local_count;
	MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MUSIC::mpi::world());
	displs[0] = 0;
	for( int i=1; i<sz; ++i ) displs[i] = displs[i-1] + counts[i-1];

	MPI_Datatype dtype = (sizeof(real_t) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;

	MPI_Scatterv( (rk==0) ? phi_root : NULL, counts.data(), displs.data(), dtype,
	              scratch[0]->m_pdata, my_count, dtype,
	              0, MUSIC::mpi::world() );

	fftw_real    *data  = reinterpret_cast<fftw_real*>(scratch[0]->m_pdata);
	fftw_complex *cdata = reinterpret_cast<fftw_complex*>(data);
	fftw_complex *cdata_h[6];
	fftw_real    *data_h[6];
	for( int s=0; s<6; ++s ) {
		data_h[s]  = reinterpret_cast<fftw_real*>(scratch[s+1]->m_pdata);
		cdata_h[s] = reinterpret_cast<fftw_complex*>(data_h[s]);
	}

	MUSIC::fft::fft_plan_t plan = MUSIC::fft::plan_r2c_3d_mpi(
		(ptrdiff_t)gnx, (ptrdiff_t)gny, (ptrdiff_t)gnz, data);
	MUSIC::fft::fft_plan_t iplan_h[6];
	for( int s=0; s<6; ++s )
		iplan_h[s] = MUSIC::fft::plan_c2r_3d_mpi(
			(ptrdiff_t)gnx, (ptrdiff_t)gny, (ptrdiff_t)gnz, data_h[s]);

	MUSIC::fft::execute(plan);

	const double kfac = 2.0*M_PI;
	const double norm = 1.0/(double)((size_t)gnx*(size_t)gny*(size_t)gnz);

	// k-space kernel: h_ab_k = -k_a*k_b * phi_k * norm, with Nyquist-zero
	// (matches serial compute_2LPT_source_FFT cdata_ab[idx][0/1] = -k[a]*k[b]*cdata[idx][0/1]*norm,
	//  plus explicit Nyquist zero when gix==gnx/2 || j==gny/2 || k==gnz/2).
	#pragma omp parallel for
	for( size_t lix = 0; lix < local_nx; ++lix ){
		const int gix = (int)(local_x_off + lix);
		int ii = gix; if( ii > (int)gnx/2 ) ii -= (int)gnx;
		for( int j = 0; j < (int)gny; ++j ){
			int jj = j; if( jj > (int)gny/2 ) jj -= (int)gny;
			for( int k = 0; k < (int)nz_complex; ++k ){
				double kx = kfac * (double)ii;
				double ky = kfac * (double)jj;
				double kz = kfac * (double)k;
				size_t idx = (lix*gny + (size_t)j) * nz_complex + (size_t)k;
				const double pre = RE(cdata[idx]) * norm;
				const double pim = IM(cdata[idx]) * norm;
				const bool nyq = (gix == (int)gnx/2 || j == (int)gny/2 || k == (int)gnz/2);
				double kk_ab[6] = {
					-kx*kx, -kx*ky, -kx*kz,
					-ky*ky, -ky*kz, -kz*kz
				};
				for( int s=0; s<6; ++s ){
					if( nyq ){
						RE(cdata_h[s][idx]) = 0.0;
						IM(cdata_h[s][idx]) = 0.0;
					} else {
						RE(cdata_h[s][idx]) = kk_ab[s] * pre;
						IM(cdata_h[s][idx]) = kk_ab[s] * pim;
					}
				}
			}
		}
	}

	for( int s=0; s<6; ++s )
		MUSIC::fft::execute(iplan_h[s]);

	MUSIC::fft::destroy(plan);
	for( int s=0; s<6; ++s ) MUSIC::fft::destroy(iplan_h[s]);

	// composite (sum of 2x2 minors of the Hessian) -> scratch[0] (reusing as
	// the Gatherv source). data_h[0..5] = h11,h12,h13,h22,h23,h33.
	#pragma omp parallel for
	for( size_t lix = 0; lix < local_nx; ++lix )
		for( size_t j = 0; j < gny; ++j )
			for( size_t k = 0; k < gnz; ++k ) {
				size_t idx = (lix*gny + j) * nz_padded + k;
				const fftw_real h11 = data_h[0][idx];
				const fftw_real h12 = data_h[1][idx];
				const fftw_real h13 = data_h[2][idx];
				const fftw_real h22 = data_h[3][idx];
				const fftw_real h23 = data_h[4][idx];
				const fftw_real h33 = data_h[5][idx];
				scratch[0]->m_pdata[idx] =
					(h11*h22 - h12*h12) +
					(h11*h33 - h13*h13) +
					(h22*h33 - h23*h23);
			}

	MPI_Gatherv( scratch[0]->m_pdata, my_count, dtype,
	             (rk==0) ? dst_root : NULL, counts.data(), displs.data(), dtype,
	             0, MUSIC::mpi::world() );

	for( int s=0; s<7; ++s ) delete scratch[s];
#else
	(void)phi_root; (void)dst_root; (void)gnx; (void)gny; (void)gnz;
#endif
}

// ---------------------------------------------------------------------------
// Phase H.1: slab-direct hybrid Poisson body (PERIODIC ONLY for first ship).
//
// Drop-in slab replacement for poisson_hybrid<MeshvarBnd<real_t>> (the
// rank-0-only finest-grid hybrid Poisson correction in src/poisson.cc:1217).
// In-place on a single MeshvarBnd<real_t>* slab buffer (poisson_hybrid is
// declared as `void poisson_hybrid(T& f, ...)` and overwrites its input).
//
// PERIODIC branch only — at pillars-scale the call sites are inside
// with_pbox_distributed(...) on the finest level where levelmin==levelmax,
// matching the serial periodic branch (nxp=nx). The non-periodic 2x-padded
// branch (8x memory) is deferred to H.1.4.
// ---------------------------------------------------------------------------
template<typename real_t>
static void run_dist_fft_op_slab_hybrid( int idir, int order, bool periodic,
                                          bool deconvolve_cic,
                                          MeshvarBnd<real_t>* slab_buf,
                                          size_t gnx, size_t gny, size_t gnz )
{
#ifdef USE_MPI
	if( !periodic ){
		LOGERR("run_dist_fft_op_slab_hybrid: non-periodic 2x-padded path not implemented (H.1.4)");
		throw std::runtime_error("run_dist_fft_op_slab_hybrid: non-periodic not implemented");
	}
	if( order != 2 && order != 4 && order != 6 ){
		LOGERR("run_dist_fft_op_slab_hybrid: unsupported stencil order %d (expected 2/4/6)", order);
		throw std::runtime_error("run_dist_fft_op_slab_hybrid: unsupported order");
	}

	Meshvar<real_t>* scratch = MUSIC::dist::make_slab_meshvar<real_t>(gnx, gny, gnz, /*fftw_inplace_pad=*/true);
	const size_t local_nx    = scratch->local_nx();
	const size_t local_x_off = scratch->local_x_start();
	const size_t nz_complex  = gnz/2 + 1;
	const size_t nz_padded   = 2*nz_complex;

	if( !slab_buf ) {
		LOGERR("run_dist_fft_op_slab_hybrid: NULL slab pointer (registered via set_slab_hybrid_buf?)");
		throw std::runtime_error("run_dist_fft_op_slab_hybrid: NULL slab pointer");
	}
	if( slab_buf->local_nx() != local_nx || slab_buf->local_x_start() != local_x_off ) {
		LOGERR("run_dist_fft_op_slab_hybrid: slab geometry (%zu,%zu) != FFTW MPI (%zu,%zu)",
		       slab_buf->local_x_start(), slab_buf->local_nx(), local_x_off, local_nx);
		throw std::runtime_error("run_dist_fft_op_slab_hybrid: slab geometry mismatch");
	}

	// slab -> padded scratch (mirrors run_dist_fft_op_slab packing)
	{
		#pragma omp parallel for
		for( size_t lix = 0; lix < local_nx; ++lix )
			for( size_t j = 0; j < gny; ++j )
				for( size_t k = 0; k < gnz; ++k ) {
					size_t idx = (lix*gny + j) * nz_padded + k;
					scratch->m_pdata[idx] = (*slab_buf)((int)lix, (int)j, (int)k);
				}
	}

	fftw_real    *data  = reinterpret_cast<fftw_real*>(scratch->m_pdata);
	fftw_complex *cdata = reinterpret_cast<fftw_complex*>(data);

	MUSIC::fft::fft_plan_t plan  = MUSIC::fft::plan_r2c_3d_mpi(
		(ptrdiff_t)gnx, (ptrdiff_t)gny, (ptrdiff_t)gnz, data);
	MUSIC::fft::fft_plan_t iplan = MUSIC::fft::plan_c2r_3d_mpi(
		(ptrdiff_t)gnx, (ptrdiff_t)gny, (ptrdiff_t)gnz, data);

	MUSIC::fft::execute(plan);

	// k-space hybrid correction (matches do_poisson_hybrid<order> kernel)
	const double fftnorm = 1.0/((double)gnx*(double)gny*(double)gnz);
	const int nxp = (int)gnx, nyp = (int)gny, nzp = (int)gnz;

	#pragma omp parallel for
	for( size_t lix = 0; lix < local_nx; ++lix ){
		const int gix = (int)(local_x_off + lix);
		int ki = gix; if( ki > nxp/2 ) ki -= nxp;
		for( int j = 0; j < nyp; ++j ){
			int kj = j; if( kj > nyp/2 ) kj -= nyp;
			for( int k = 0; k < (int)nz_complex; ++k ){
				int kk = k;
				size_t idx = (lix*(size_t)nyp + (size_t)j) * nz_complex + (size_t)k;

				double dk;
				if( order == 2 )
					dk = poisson_hybrid_kernel<2>(idir, ki, kj, kk, nxp/2);
				else if( order == 4 )
					dk = poisson_hybrid_kernel<4>(idir, ki, kj, kk, nxp/2);
				else
					dk = poisson_hybrid_kernel<6>(idir, ki, kj, kk, nxp/2);

				fftw_real re = RE(cdata[idx]);
				fftw_real im = IM(cdata[idx]);
				RE(cdata[idx]) = -im*dk*fftnorm;
				IM(cdata[idx]) =  re*dk*fftnorm;

				if( deconvolve_cic ) {
					double dfx, dfy, dfz;
					dfx = M_PI*ki/(double)nxp; dfx = (gix!=0) ? sin(dfx)/dfx : 1.0;
					dfy = M_PI*kj/(double)nyp; dfy = (j  !=0) ? sin(dfy)/dfy : 1.0;
					dfz = M_PI*kk/(double)nzp; dfz = (k  !=0) ? sin(dfz)/dfz : 1.0;
					double w = 1.0/(dfx*dfy*dfz); w = w*w;
					RE(cdata[idx]) *= w;
					IM(cdata[idx]) *= w;
				}
			}
		}
	}

	if( local_x_off == 0 && local_nx > 0 ){
		RE(cdata[0]) = 0.0;
		IM(cdata[0]) = 0.0;
	}

	MUSIC::fft::execute(iplan);
	MUSIC::fft::destroy(plan);
	MUSIC::fft::destroy(iplan);

	// padded scratch -> slab (in-place overwrite)
	{
		#pragma omp parallel for
		for( size_t lix = 0; lix < local_nx; ++lix )
			for( size_t j = 0; j < gny; ++j )
				for( size_t k = 0; k < gnz; ++k ) {
					size_t idx = (lix*gny + j) * nz_padded + k;
					(*slab_buf)((int)lix, (int)j, (int)k) = scratch->m_pdata[idx];
				}
	}

	delete scratch;
#else
	(void)idir; (void)order; (void)periodic; (void)deconvolve_cic;
	(void)slab_buf; (void)gnx; (void)gny; (void)gnz;
#endif
}

// Module-private registered slab pointers (one set per build; build is single
// precision OR double, never both). All ranks must call set_slab_solve_inout
// before entering phase_scope so OP_SOLVE_SLAB dispatch finds the right buffers.
static MeshvarBnd<fftw_real>* g_slab_solve_src = NULL;
static MeshvarBnd<fftw_real>* g_slab_solve_dst = NULL;

// Phase H.1 in-place single-buffer registered pointer (poisson_hybrid is
// in-place; all ranks call set_slab_hybrid_buf before phase_scope).
static MeshvarBnd<fftw_real>* g_slab_hybrid_buf = NULL;

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

void worker_pump()
{
#ifdef USE_MPI
	if( MUSIC::mpi::size() <= 1 ) return;
	TRACE_PHASE("worker_pump enter");
	while( true ){
		int meta[META_LEN] = {0,0,0,0,0,0,0,0};
		MPI_Bcast(meta, META_LEN, MPI_INT, 0, MUSIC::mpi::world());
		const int op = meta[0];
		TRACE_PHASE("worker_pump recv op=%d", op);
		if( op == OP_DONE ){ TRACE_PHASE("worker_pump exit"); return; }
		const size_t gnx = (size_t)meta[1];
		const size_t gny = (size_t)meta[2];
		const size_t gnz = (size_t)meta[3];
		const int    dir = meta[4];
		const bool   dec = (meta[5] != 0);
		if( op == OP_SOLVE || op == OP_GRADIENT ){
			run_dist_fft_op<fftw_real>(op, dir, NULL, gnx, gny, gnz, dec);
		} else if( op == OP_SOLVE_SLAB || op == OP_GRADIENT_SLAB ){
			run_dist_fft_op_slab<fftw_real>(op, dir, g_slab_solve_src, g_slab_solve_dst,
			                                gnx, gny, gnz, dec);
		} else if( op == OP_LPT2_FFT ){
			run_dist_fft_op_lpt2<fftw_real>(NULL, NULL, gnx, gny, gnz);
		} else if( op == OP_POISSON_HYBRID ){
			const int   order    = meta[6];
			const bool  periodic = (meta[7] != 0);
			run_dist_fft_op_slab_hybrid<fftw_real>( dir, order, periodic, dec,
			                                         g_slab_hybrid_buf,
			                                         gnx, gny, gnz );
		} else if( op >= MUSIC::mg::OP_MG_BEGIN && op <= MUSIC::mg::OP_MG_INTERP_CF ){
			MUSIC::mg::worker_handle(op);
		} else {
			LOGERR("MUSIC::poisson::worker_pump: unknown op %d", op);
			throw std::runtime_error("worker_pump: unknown op");
		}
	}
#endif
}

void broadcast_done()
{
#ifdef USE_MPI
	if( MUSIC::mpi::size() <= 1 ) return;
	TRACE_PHASE("broadcast_done begin");
	int meta[META_LEN] = { OP_DONE, 0,0,0,0,0,0,0 };
	MPI_Bcast(meta, META_LEN, MPI_INT, 0, MUSIC::mpi::world());
	TRACE_PHASE("broadcast_done end");
#endif
}

phase_scope::phase_scope()
{
#ifdef USE_MPI
	if( MUSIC::mpi::size() > 1 && !MUSIC::mpi::is_root() ){
		TRACE_PHASE("phase_scope ctor (worker → pump)");
		worker_pump();
	} else if( MUSIC::mpi::size() > 1 ){
		TRACE_PHASE("phase_scope ctor (root → body)");
	}
#endif
}

phase_scope::~phase_scope()
{
#ifdef USE_MPI
	if( MUSIC::mpi::size() > 1 && MUSIC::mpi::is_root() ){
		TRACE_PHASE("phase_scope dtor (root → bcast DONE)");
		broadcast_done();
	} else if( MUSIC::mpi::size() > 1 ){
		TRACE_PHASE("phase_scope dtor (worker noop)");
	}
#endif
}

template<typename real_t>
void rank0_dist_solve( real_t* root_buf, size_t gnx, size_t gny, size_t gnz )
{
#ifdef USE_MPI
	if( MUSIC::mpi::size() <= 1 ){
		LOGERR("MUSIC::poisson::rank0_dist_solve called with size<=1");
		throw std::runtime_error("rank0_dist_solve called outside MPI");
	}
	int meta[META_LEN] = { OP_SOLVE, (int)gnx, (int)gny, (int)gnz, 0, 0, 0, 0 };
	MPI_Bcast(meta, META_LEN, MPI_INT, 0, MUSIC::mpi::world());
	run_dist_fft_op<real_t>(OP_SOLVE, 0, root_buf, gnx, gny, gnz, false);
#else
	(void)root_buf; (void)gnx; (void)gny; (void)gnz;
#endif
}

template<typename real_t>
void rank0_dist_gradient( int dir, real_t* root_buf, size_t gnx, size_t gny, size_t gnz,
                          bool deconvolve_cic )
{
#ifdef USE_MPI
	if( MUSIC::mpi::size() <= 1 ){
		LOGERR("MUSIC::poisson::rank0_dist_gradient called with size<=1");
		throw std::runtime_error("rank0_dist_gradient called outside MPI");
	}
	int meta[META_LEN] = { OP_GRADIENT, (int)gnx, (int)gny, (int)gnz, dir, deconvolve_cic ? 1 : 0, 0, 0 };
	MPI_Bcast(meta, META_LEN, MPI_INT, 0, MUSIC::mpi::world());
	run_dist_fft_op<real_t>(OP_GRADIENT, dir, root_buf, gnx, gny, gnz, deconvolve_cic);
#else
	(void)dir; (void)root_buf; (void)gnx; (void)gny; (void)gnz; (void)deconvolve_cic;
#endif
}

// ----- Phase E.2.2a public API ---------------------------------------------
template<typename real_t>
void set_slab_solve_inout( MeshvarBnd<real_t>* src, MeshvarBnd<real_t>* dst )
{
	// fftw_real is the build precision; this overload only valid for that type.
	g_slab_solve_src = reinterpret_cast<MeshvarBnd<fftw_real>*>(src);
	g_slab_solve_dst = reinterpret_cast<MeshvarBnd<fftw_real>*>(dst);
}

template<typename real_t>
void rank0_dist_solve_slab( size_t gnx, size_t gny, size_t gnz )
{
#ifdef USE_MPI
	if( MUSIC::mpi::size() <= 1 ){
		LOGERR("MUSIC::poisson::rank0_dist_solve_slab called with size<=1");
		throw std::runtime_error("rank0_dist_solve_slab called outside MPI");
	}
	int meta[META_LEN] = { OP_SOLVE_SLAB, (int)gnx, (int)gny, (int)gnz, 0, 0, 0, 0 };
	MPI_Bcast(meta, META_LEN, MPI_INT, 0, MUSIC::mpi::world());
	run_dist_fft_op_slab<real_t>(OP_SOLVE_SLAB, 0,
	                              reinterpret_cast<MeshvarBnd<real_t>*>(g_slab_solve_src),
	                              reinterpret_cast<MeshvarBnd<real_t>*>(g_slab_solve_dst),
	                              gnx, gny, gnz, false);
#else
	(void)gnx; (void)gny; (void)gnz;
#endif
}

// ----- Phase E.2.3 public API ----------------------------------------------
// Reuses g_slab_solve_src/dst (the in/out plumbing is shared across slab
// ops; caller calls set_slab_solve_inout before each phase_scope).
template<typename real_t>
void rank0_dist_gradient_slab( int dir, size_t gnx, size_t gny, size_t gnz,
                                bool deconvolve_cic )
{
#ifdef USE_MPI
	if( MUSIC::mpi::size() <= 1 ){
		LOGERR("MUSIC::poisson::rank0_dist_gradient_slab called with size<=1");
		throw std::runtime_error("rank0_dist_gradient_slab called outside MPI");
	}
	int meta[META_LEN] = { OP_GRADIENT_SLAB, (int)gnx, (int)gny, (int)gnz,
	                        dir, deconvolve_cic ? 1 : 0, 0, 0 };
	MPI_Bcast(meta, META_LEN, MPI_INT, 0, MUSIC::mpi::world());
	run_dist_fft_op_slab<real_t>(OP_GRADIENT_SLAB, dir,
	                              reinterpret_cast<MeshvarBnd<real_t>*>(g_slab_solve_src),
	                              reinterpret_cast<MeshvarBnd<real_t>*>(g_slab_solve_dst),
	                              gnx, gny, gnz, deconvolve_cic);
#else
	(void)dir; (void)gnx; (void)gny; (void)gnz; (void)deconvolve_cic;
#endif
}

// ----- Phase F2LPT.1 public API --------------------------------------------
template<typename real_t>
void rank0_dist_lpt2( real_t* phi_root, real_t* dst_root,
                      size_t gnx, size_t gny, size_t gnz )
{
#ifdef USE_MPI
	if( MUSIC::mpi::size() <= 1 ){
		LOGERR("MUSIC::poisson::rank0_dist_lpt2 called with size<=1");
		throw std::runtime_error("rank0_dist_lpt2 called outside MPI");
	}
	int meta[META_LEN] = { OP_LPT2_FFT, (int)gnx, (int)gny, (int)gnz, 0, 0, 0, 0 };
	MPI_Bcast(meta, META_LEN, MPI_INT, 0, MUSIC::mpi::world());
	run_dist_fft_op_lpt2<real_t>(phi_root, dst_root, gnx, gny, gnz);
#else
	(void)phi_root; (void)dst_root; (void)gnx; (void)gny; (void)gnz;
#endif
}

// ----- Phase H.1 public API -----------------------------------------------
template<typename real_t>
void set_slab_hybrid_buf( MeshvarBnd<real_t>* buf )
{
	// fftw_real is the build precision; this overload only valid for that type.
	g_slab_hybrid_buf = reinterpret_cast<MeshvarBnd<fftw_real>*>(buf);
}

template<typename real_t>
void rank0_dist_poisson_hybrid_slab( int idir, int order, bool periodic,
                                     bool deconvolve_cic,
                                     size_t gnx, size_t gny, size_t gnz )
{
#ifdef USE_MPI
	if( MUSIC::mpi::size() <= 1 ){
		LOGERR("MUSIC::poisson::rank0_dist_poisson_hybrid_slab called with size<=1");
		throw std::runtime_error("rank0_dist_poisson_hybrid_slab called outside MPI");
	}
	int meta[META_LEN] = { OP_POISSON_HYBRID,
	                       (int)gnx, (int)gny, (int)gnz,
	                       idir, deconvolve_cic ? 1 : 0,
	                       order, periodic ? 1 : 0 };
	MPI_Bcast(meta, META_LEN, MPI_INT, 0, MUSIC::mpi::world());
	run_dist_fft_op_slab_hybrid<real_t>( idir, order, periodic, deconvolve_cic,
	                                      reinterpret_cast<MeshvarBnd<real_t>*>(g_slab_hybrid_buf),
	                                      gnx, gny, gnz );
#else
	(void)idir; (void)order; (void)periodic; (void)deconvolve_cic;
	(void)gnx; (void)gny; (void)gnz;
#endif
}

// explicit instantiations for the build's precision (matches the FFTW plans)
#ifdef SINGLE_PRECISION
template void rank0_dist_solve<float >(float *, size_t, size_t, size_t);
template void rank0_dist_gradient<float >(int, float *, size_t, size_t, size_t, bool);
template void set_slab_solve_inout<float >(MeshvarBnd<float >*, MeshvarBnd<float >*);
template void rank0_dist_solve_slab<float >(size_t, size_t, size_t);
template void rank0_dist_gradient_slab<float >(int, size_t, size_t, size_t, bool);
template void rank0_dist_lpt2<float >(float *, float *, size_t, size_t, size_t);
template void set_slab_hybrid_buf<float >(MeshvarBnd<float >*);
template void rank0_dist_poisson_hybrid_slab<float >(int, int, bool, bool, size_t, size_t, size_t);
#else
template void rank0_dist_solve<double>(double*, size_t, size_t, size_t);
template void rank0_dist_gradient<double>(int, double*, size_t, size_t, size_t, bool);
template void set_slab_solve_inout<double>(MeshvarBnd<double>*, MeshvarBnd<double>*);
template void rank0_dist_solve_slab<double>(size_t, size_t, size_t);
template void rank0_dist_gradient_slab<double>(int, size_t, size_t, size_t, bool);
template void rank0_dist_lpt2<double>(double*, double*, size_t, size_t, size_t);
template void set_slab_hybrid_buf<double>(MeshvarBnd<double>*);
template void rank0_dist_poisson_hybrid_slab<double>(int, int, bool, bool, size_t, size_t, size_t);
#endif

}} // namespace MUSIC::poisson
