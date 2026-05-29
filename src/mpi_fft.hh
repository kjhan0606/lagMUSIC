#ifndef __MPI_FFT_HH
#define __MPI_FFT_HH

// Thin wrappers for FFTW3 r2c/c2r plans, both single-rank and MPI-distributed.
// All functions are templated/overloaded on real_t so single- and double-precision
// builds compile cleanly with the same call sites.
//
// Naming:
//   fft_real_t  := MUSIC's fftw_real (float or double, set in general.hh)
//   fft_plan_t  := fftw_plan or fftwf_plan
//   fft_cplx_t  := fftw_complex or fftwf_complex
//
// The MPI variants require src to be allocated with the FFTW r2c in-place layout
// (inner dim padded to 2*(gnz/2+1)). Use MUSIC::dist::make_slab_meshvar to
// allocate the buffer correctly.

#include "general.hh"

#ifdef USE_MPI
#include <fftw3-mpi.h>
#include "mpi_helper.hh"
#endif

namespace MUSIC { namespace fft {

#ifdef SINGLE_PRECISION
typedef fftwf_plan    fft_plan_t;
typedef fftwf_complex fft_cplx_t;
#else
typedef fftw_plan     fft_plan_t;
typedef fftw_complex  fft_cplx_t;
#endif

//! In-place serial r2c plan for an nx*ny*nz real grid (padded inner dim).
inline fft_plan_t plan_r2c_3d_serial( int nx, int ny, int nz, fftw_real* data )
{
	fft_cplx_t* cdata = reinterpret_cast<fft_cplx_t*>(data);
#ifdef SINGLE_PRECISION
	return fftwf_plan_dft_r2c_3d(nx, ny, nz, data, cdata, FFTW_ESTIMATE);
#else
	return fftw_plan_dft_r2c_3d(nx, ny, nz, data, cdata, FFTW_ESTIMATE);
#endif
}

//! In-place serial c2r plan for an nx*ny*nz real grid (padded inner dim).
inline fft_plan_t plan_c2r_3d_serial( int nx, int ny, int nz, fftw_real* data )
{
	fft_cplx_t* cdata = reinterpret_cast<fft_cplx_t*>(data);
#ifdef SINGLE_PRECISION
	return fftwf_plan_dft_c2r_3d(nx, ny, nz, cdata, data, FFTW_ESTIMATE);
#else
	return fftw_plan_dft_c2r_3d(nx, ny, nz, cdata, data, FFTW_ESTIMATE);
#endif
}

#ifdef USE_MPI
//! In-place MPI r2c plan on an explicit communicator (default-comm uses world).
inline fft_plan_t plan_r2c_3d_mpi_comm( MPI_Comm comm,
                                        ptrdiff_t gnx, ptrdiff_t gny, ptrdiff_t gnz,
                                        fftw_real* data )
{
	fft_cplx_t* cdata = reinterpret_cast<fft_cplx_t*>(data);
#ifdef SINGLE_PRECISION
	return fftwf_mpi_plan_dft_r2c_3d(gnx, gny, gnz, data, cdata,
	                                 comm, FFTW_ESTIMATE);
#else
	return fftw_mpi_plan_dft_r2c_3d(gnx, gny, gnz, data, cdata,
	                                comm, FFTW_ESTIMATE);
#endif
}

//! In-place MPI c2r plan on an explicit communicator (default-comm uses world).
inline fft_plan_t plan_c2r_3d_mpi_comm( MPI_Comm comm,
                                        ptrdiff_t gnx, ptrdiff_t gny, ptrdiff_t gnz,
                                        fftw_real* data )
{
	fft_cplx_t* cdata = reinterpret_cast<fft_cplx_t*>(data);
#ifdef SINGLE_PRECISION
	return fftwf_mpi_plan_dft_c2r_3d(gnx, gny, gnz, cdata, data,
	                                 comm, FFTW_ESTIMATE);
#else
	return fftw_mpi_plan_dft_c2r_3d(gnx, gny, gnz, cdata, data,
	                                comm, FFTW_ESTIMATE);
#endif
}

//! In-place MPI r2c plan for an gnx*gny*gnz real grid (padded inner dim), world comm.
inline fft_plan_t plan_r2c_3d_mpi( ptrdiff_t gnx, ptrdiff_t gny, ptrdiff_t gnz, fftw_real* data )
{
	return plan_r2c_3d_mpi_comm(MUSIC::mpi::world(), gnx, gny, gnz, data);
}

//! In-place MPI c2r plan for an gnx*gny*gnz real grid (padded inner dim), world comm.
inline fft_plan_t plan_c2r_3d_mpi( ptrdiff_t gnx, ptrdiff_t gny, ptrdiff_t gnz, fftw_real* data )
{
	return plan_c2r_3d_mpi_comm(MUSIC::mpi::world(), gnx, gny, gnz, data);
}
#endif

inline void execute( fft_plan_t p )
{
#ifdef SINGLE_PRECISION
	fftwf_execute(p);
#else
	fftw_execute(p);
#endif
}

inline void destroy( fft_plan_t p )
{
#ifdef SINGLE_PRECISION
	fftwf_destroy_plan(p);
#else
	fftw_destroy_plan(p);
#endif
}

}} // namespace MUSIC::fft

#endif // __MPI_FFT_HH
