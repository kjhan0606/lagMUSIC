// MPI-distributed FFT-Poisson helpers + worker pump. See mpi_poisson.hh.

#include "mpi_poisson.hh"
#include "mpi_helper.hh"
#include "mesh.hh"
#include "mesh_distributed.hh"
#include "mpi_fft.hh"
#include "mg_dist.hh"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace MUSIC { namespace poisson {

#ifdef USE_MPI
// meta payload broadcast from rank 0 to workers in worker_pump:
//   [0]=op  [1]=gnx  [2]=gny  [3]=gnz  [4]=dir  [5]=deconvolve_cic
static const int META_LEN = 6;
#endif

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

// Module-private registered slab pointers (one set per build; build is single
// precision OR double, never both). All ranks must call set_slab_solve_inout
// before entering phase_scope so OP_SOLVE_SLAB dispatch finds the right buffers.
static MeshvarBnd<fftw_real>* g_slab_solve_src = NULL;
static MeshvarBnd<fftw_real>* g_slab_solve_dst = NULL;

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

void worker_pump()
{
#ifdef USE_MPI
	if( MUSIC::mpi::size() <= 1 ) return;
	while( true ){
		int meta[META_LEN] = {0,0,0,0,0,0};
		MPI_Bcast(meta, META_LEN, MPI_INT, 0, MUSIC::mpi::world());
		const int op = meta[0];
		if( op == OP_DONE ) return;
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
	int meta[META_LEN] = { OP_DONE, 0,0,0,0,0 };
	MPI_Bcast(meta, META_LEN, MPI_INT, 0, MUSIC::mpi::world());
#endif
}

phase_scope::phase_scope()
{
#ifdef USE_MPI
	if( MUSIC::mpi::size() > 1 && !MUSIC::mpi::is_root() ){
		worker_pump();
	}
#endif
}

phase_scope::~phase_scope()
{
#ifdef USE_MPI
	if( MUSIC::mpi::size() > 1 && MUSIC::mpi::is_root() ){
		broadcast_done();
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
	int meta[META_LEN] = { OP_SOLVE, (int)gnx, (int)gny, (int)gnz, 0, 0 };
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
	int meta[META_LEN] = { OP_GRADIENT, (int)gnx, (int)gny, (int)gnz, dir, deconvolve_cic ? 1 : 0 };
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
	int meta[META_LEN] = { OP_SOLVE_SLAB, (int)gnx, (int)gny, (int)gnz, 0, 0 };
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
	                        dir, deconvolve_cic ? 1 : 0 };
	MPI_Bcast(meta, META_LEN, MPI_INT, 0, MUSIC::mpi::world());
	run_dist_fft_op_slab<real_t>(OP_GRADIENT_SLAB, dir,
	                              reinterpret_cast<MeshvarBnd<real_t>*>(g_slab_solve_src),
	                              reinterpret_cast<MeshvarBnd<real_t>*>(g_slab_solve_dst),
	                              gnx, gny, gnz, deconvolve_cic);
#else
	(void)dir; (void)gnx; (void)gny; (void)gnz; (void)deconvolve_cic;
#endif
}

// explicit instantiations for the build's precision (matches the FFTW plans)
#ifdef SINGLE_PRECISION
template void rank0_dist_solve<float >(float *, size_t, size_t, size_t);
template void rank0_dist_gradient<float >(int, float *, size_t, size_t, size_t, bool);
template void set_slab_solve_inout<float >(MeshvarBnd<float >*, MeshvarBnd<float >*);
template void rank0_dist_solve_slab<float >(size_t, size_t, size_t);
template void rank0_dist_gradient_slab<float >(int, size_t, size_t, size_t, bool);
#else
template void rank0_dist_solve<double>(double*, size_t, size_t, size_t);
template void rank0_dist_gradient<double>(int, double*, size_t, size_t, size_t, bool);
template void set_slab_solve_inout<double>(MeshvarBnd<double>*, MeshvarBnd<double>*);
template void rank0_dist_solve_slab<double>(size_t, size_t, size_t);
template void rank0_dist_gradient_slab<double>(int, size_t, size_t, size_t, bool);
#endif

}} // namespace MUSIC::poisson
