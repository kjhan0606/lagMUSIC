/*
 
 convolution_kernel.cc - This file is part of MUSIC -
 a code to generate multi-scale initial conditions 
 for cosmological simulations 
 
 Copyright (C) 2010-19  Oliver Hahn
 
*/

#include "general.hh"
#include "densities.hh"
#include "convolution_kernel.hh"
#include "mesh.hh"
#include "mpi_helper.hh"
#include "mpi_fft.hh"
#include "mesh_distributed.hh"
#include <vector>
#include <cstring>
#include <climits>
#include <cerrno>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>

#if defined(FFTW3) && defined(SINGLE_PRECISION)
//#define fftw_complex fftwf_complex
typedef fftw_complex fftwf_complex;
#endif

double T0 = 1.0;

namespace convolution
{

std::map<std::string, kernel_creator *> &
get_kernel_map()
{
	static std::map<std::string, kernel_creator *> kernel_map;
	return kernel_map;
}

template <typename real_t>
void perform(kernel *pk, void *pd, bool shift, bool fix, bool flip)
{
	//return;

	parameters cparam_ = pk->cparam_;
	double fftnormp = 1.0/sqrt((double)cparam_.nx * (double)cparam_.ny * (double)cparam_.nz);
	double fftnorm = pow(2.0 * M_PI, 1.5) / sqrt(cparam_.lx * cparam_.ly * cparam_.lz) * fftnormp;

	fftw_complex *cdata, *ckernel;
	fftw_real *data;

	data = reinterpret_cast<fftw_real *>(pd);
	cdata = reinterpret_cast<fftw_complex *>(data);
	ckernel = reinterpret_cast<fftw_complex *>(pk->get_ptr());

	std::cout << "   - Performing density convolution... ("
			  << cparam_.nx << ", " << cparam_.ny << ", " << cparam_.nz << ")\n";

	LOGUSER("Performing kernel convolution on (%5d,%5d,%5d) grid", cparam_.nx, cparam_.ny, cparam_.nz);
	LOGUSER("Performing forward FFT...");
#ifdef FFTW3
#ifdef SINGLE_PRECISION
	fftwf_plan plan, iplan;
	plan = fftwf_plan_dft_r2c_3d(cparam_.nx, cparam_.ny, cparam_.nz, data, cdata, FFTW_ESTIMATE);
	iplan = fftwf_plan_dft_c2r_3d(cparam_.nx, cparam_.ny, cparam_.nz, cdata, data, FFTW_ESTIMATE);

	fftwf_execute(plan);
#else
	fftw_plan plan, iplan;
	plan = fftw_plan_dft_r2c_3d(cparam_.nx, cparam_.ny, cparam_.nz, data, cdata, FFTW_ESTIMATE);
	iplan = fftw_plan_dft_c2r_3d(cparam_.nx, cparam_.ny, cparam_.nz, cdata, data, FFTW_ESTIMATE);

	fftw_execute(plan);
#endif
#else
	rfftwnd_plan iplan, plan;

	plan = rfftw3d_create_plan(cparam_.nx, cparam_.ny, cparam_.nz,
							   FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE | FFTW_IN_PLACE);

	iplan = rfftw3d_create_plan(cparam_.nx, cparam_.ny, cparam_.nz,
								FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE | FFTW_IN_PLACE);

#ifndef SINGLETHREAD_FFTW
	rfftwnd_threads_one_real_to_complex(omp_get_max_threads(), plan, data, NULL);
#else
	rfftwnd_one_real_to_complex(plan, data, NULL);
#endif

#endif
	//..... need a phase shift for baryons for SPH
	double dstag = 0.0;

	if (shift)
	{
		double boxlength = pk->pcf_->getValue<double>("setup", "boxlength");
		double stagfact = pk->pcf_->getValueSafe<double>("setup", "baryon_staggering", 0.5);
		int lmax = pk->pcf_->getValue<int>("setup", "levelmax");
		double dxmax = boxlength / (1 << lmax);
		double dxcur = cparam_.lx / cparam_.nx;
		//std::cerr << "Performing staggering shift for SPH\n";
		LOGUSER("Performing staggering shift for SPH");
		dstag = stagfact * 2.0 * M_PI / cparam_.nx * dxmax / dxcur;
	}

	//.............................................

	std::complex<double> dcmode(RE(cdata[0]), IM(cdata[0]));

	if (!pk->is_ksampled())
	{

#pragma omp parallel for
		for (int i = 0; i < cparam_.nx; ++i)
			for (int j = 0; j < cparam_.ny; ++j)
				for (int k = 0; k < cparam_.nz / 2 + 1; ++k)
				{
					size_t ii = (size_t)(i * cparam_.ny + j) * (size_t)(cparam_.nz / 2 + 1) + (size_t)k;

					double kx, ky, kz;

					kx = (double)i;
					ky = (double)j;
					kz = (double)k;

					if (kx > cparam_.nx / 2)
						kx -= cparam_.nx;
					if (ky > cparam_.ny / 2)
						ky -= cparam_.ny;

					double arg = (kx + ky + kz) * dstag;
					std::complex<double> carg(cos(arg), sin(arg));

					std::complex<double>
						ccdata(RE(cdata[ii]), IM(cdata[ii])),
						cckernel(RE(ckernel[ii]), IM(ckernel[ii]));

					if( fix ){
						ccdata = ccdata / std::abs(ccdata);
					}
					if( flip ){
						ccdata = -ccdata;
					}

					ccdata = ccdata * cckernel * fftnorm * carg;

					RE(cdata[ii]) = ccdata.real();
					IM(cdata[ii]) = ccdata.imag();
				}
	}
	else
	{

#pragma omp parallel
		{

			const size_t veclen = cparam_.nz / 2 + 1;

			double *kvec = new double[veclen];
			double *Tkvec = new double[veclen];
			double *argvec = new double[veclen];

#pragma omp for
			for (int i = 0; i < cparam_.nx; ++i)
				for (int j = 0; j < cparam_.ny; ++j)
				{

					for (int k = 0; k < cparam_.nz / 2 + 1; ++k)
					{
						double kx, ky, kz;

						kx = (double)i;
						ky = (double)j;
						kz = (double)k;

						if (kx > cparam_.nx / 2)
							kx -= cparam_.nx;
						if (ky > cparam_.ny / 2)
							ky -= cparam_.ny;

						kvec[k] = sqrt(kx * kx + ky * ky + kz * kz);
						argvec[k] = (kx + ky + kz) * dstag;
					}

					pk->at_k(veclen, kvec, Tkvec);

					for (int k = 0; k < cparam_.nz / 2 + 1; ++k)
					{
						size_t ii = (size_t)(i * cparam_.ny + j) * (size_t)(cparam_.nz / 2 + 1) + (size_t)k;
						std::complex<double> carg(cos(argvec[k]), sin(argvec[k]));

						std::complex<double> ccdata(RE(cdata[ii]), IM(cdata[ii]));

						if( fix ){
							ccdata = ccdata / std::abs(ccdata) / fftnormp;
						}
						if( flip ){
							ccdata = -ccdata;
						}

						ccdata = ccdata * Tkvec[k] * fftnorm * carg;

						RE(cdata[ii]) = ccdata.real();
						IM(cdata[ii]) = ccdata.imag();
					}
				}

			delete[] kvec;
			delete[] Tkvec;
			delete[] argvec;
		}

		// we now set the correct DC mode below...
		RE(cdata[0]) = 0.0;
		IM(cdata[0]) = 0.0;
	}

	LOGUSER("Performing backward FFT...");

#ifdef FFTW3
#ifdef SINGLE_PRECISION
	fftwf_execute(iplan);
	fftwf_destroy_plan(plan);
	fftwf_destroy_plan(iplan);
#else
	fftw_execute(iplan);
	fftw_destroy_plan(plan);
	fftw_destroy_plan(iplan);

#endif
#else
#ifndef SINGLETHREAD_FFTW
	rfftwnd_threads_one_complex_to_real(omp_get_max_threads(), iplan, cdata, NULL);
#else
	rfftwnd_one_complex_to_real(iplan, cdata, NULL);
#endif

	rfftwnd_destroy_plan(plan);
	rfftwnd_destroy_plan(iplan);
#endif

	// set the DC mode here to avoid a possible truncation error in single precision
	if (pk->is_ksampled())
	{
		size_t nelem = (size_t)cparam_.nx * (size_t)cparam_.ny * (size_t)cparam_.nz;
		real_t mean = dcmode.real() * fftnorm / (real_t)nelem;

#pragma omp parallel for
		for (size_t i = 0; i < nelem; ++i)
			data[i] += mean;
	}
}

template void perform<double>(kernel *pk, void *pd, bool shift, bool fix, bool flip);
template void perform<float>(kernel *pk, void *pd, bool shift, bool fix, bool flip);

/*****************************************************************************************/
/***    MPI / SLAB-DISTRIBUTED VARIANT                                                  ***/
/*****************************************************************************************/

template <typename real_t>
void perform_mpi(kernel *pk, Meshvar<real_t> *pdist, bool shift, bool fix, bool flip)
{
	if( !pdist->is_distributed() ){
		LOGERR("convolution::perform_mpi requires a slab-distributed Meshvar (see MUSIC::dist::make_slab_meshvar).");
		throw std::runtime_error("perform_mpi: Meshvar is not distributed");
	}
	const bool ksampled = pk->is_ksampled();

	const parameters cparam_ = pk->cparam_;
	const size_t gnx = (size_t)cparam_.nx;
	const size_t gny = (size_t)cparam_.ny;
	const size_t gnz = (size_t)cparam_.nz;
	if( pdist->global_size(0) != gnx || pdist->global_size(1) != gny || pdist->global_size(2) != gnz ){
		LOGERR("convolution::perform_mpi: kernel/buffer size mismatch (kernel=%zux%zux%zu, buffer=%zux%zux%zu).",
		       gnx, gny, gnz,
		       pdist->global_size(0), pdist->global_size(1), pdist->global_size(2));
		throw std::runtime_error("perform_mpi: size mismatch");
	}

	const size_t local_nx     = pdist->local_nx();
	const size_t local_x_off  = pdist->local_x_start();
	const size_t nz_complex   = gnz/2 + 1;

	const double fftnormp = 1.0 / sqrt((double)gnx * (double)gny * (double)gnz);
	const double fftnorm  = pow(2.0*M_PI, 1.5) / sqrt(cparam_.lx*cparam_.ly*cparam_.lz) * fftnormp;

	fftw_real    *data  = reinterpret_cast<fftw_real*>(pdist->m_pdata);
	fftw_complex *cdata = reinterpret_cast<fftw_complex*>(data);

	double dstag = 0.0;
	if( shift ){
		double boxlength = pk->pcf_->getValue<double>("setup","boxlength");
		double stagfact  = pk->pcf_->getValueSafe<double>("setup","baryon_staggering",0.5);
		int lmax         = pk->pcf_->getValue<int>("setup","levelmax");
		double dxmax     = boxlength / (1 << lmax);
		double dxcur     = cparam_.lx / cparam_.nx;
		LOGUSER("Performing staggering shift for SPH (MPI)");
		dstag = stagfact * 2.0 * M_PI / cparam_.nx * dxmax / dxcur;
	}

	if( MUSIC::mpi::is_root() ){
		std::cout << "   - Performing density convolution (MPI)... ("
		          << gnx << ", " << gny << ", " << gnz << ") on " << MUSIC::mpi::size() << " ranks\n";
	}
	LOGUSER("Performing kernel convolution (MPI) on (%zu,%zu,%zu) grid", gnx, gny, gnz);

#ifdef USE_MPI
	MUSIC::fft::fft_plan_t plan  = MUSIC::fft::plan_r2c_3d_mpi(
		(ptrdiff_t)gnx, (ptrdiff_t)gny, (ptrdiff_t)gnz, data);
	MUSIC::fft::fft_plan_t iplan = MUSIC::fft::plan_c2r_3d_mpi(
		(ptrdiff_t)gnx, (ptrdiff_t)gny, (ptrdiff_t)gnz, data);
#else
	MUSIC::fft::fft_plan_t plan  = MUSIC::fft::plan_r2c_3d_serial(
		(int)gnx, (int)gny, (int)gnz, data);
	MUSIC::fft::fft_plan_t iplan = MUSIC::fft::plan_c2r_3d_serial(
		(int)gnx, (int)gny, (int)gnz, data);
#endif

	LOGUSER("Performing forward FFT (MPI)...");
	MUSIC::fft::execute(plan);

	// DC mode capture (only needed by the ksampled path, which zeroes it and
	// restores it as a mean after the inverse FFT). The cached real-space path
	// preserves the DC mode through the multiply, matching the serial behaviour.
	std::complex<double> dcmode(0.0, 0.0);
	if( ksampled ){
		double dc_re_local = 0.0, dc_im_local = 0.0;
		int    dc_owner_local = -1;
		if( local_x_off == 0 && local_nx > 0 ){
			dc_re_local = (double)RE(cdata[0]);
			dc_im_local = (double)IM(cdata[0]);
			dc_owner_local = MUSIC::mpi::rank();
		}
		int    dc_owner = dc_owner_local;
		double dc_re = dc_re_local, dc_im = dc_im_local;
#ifdef USE_MPI
		MPI_Allreduce(&dc_owner_local, &dc_owner, 1, MPI_INT,    MPI_MAX, MUSIC::mpi::world());
		MPI_Bcast(&dc_re, 1, MPI_DOUBLE, dc_owner, MUSIC::mpi::world());
		MPI_Bcast(&dc_im, 1, MPI_DOUBLE, dc_owner, MUSIC::mpi::world());
#endif
		dcmode = std::complex<double>(dc_re, dc_im);
	}

	// K-space multiplication over the local x-slab. Global x index = local_x_off + ix.
	if( ksampled ){
#pragma omp parallel
		{
			const size_t veclen = nz_complex;
			double *kvec   = new double[veclen];
			double *Tkvec  = new double[veclen];
			double *argvec = new double[veclen];

#pragma omp for
			for( size_t ix = 0; ix < local_nx; ++ix ){
				const int gix = (int)(local_x_off + ix);
				int kx = gix;
				if( kx > (int)gnx/2 ) kx -= (int)gnx;
				for( int j = 0; j < (int)gny; ++j ){
					int ky = j;
					if( ky > (int)gny/2 ) ky -= (int)gny;
					for( int k = 0; k < (int)veclen; ++k ){
						int kz = k;
						kvec[k]   = sqrt((double)(kx*kx) + (double)(ky*ky) + (double)(kz*kz));
						argvec[k] = ((double)kx + (double)ky + (double)kz) * dstag;
					}
					pk->at_k(veclen, kvec, Tkvec);
					for( int k = 0; k < (int)veclen; ++k ){
						size_t ii = (ix*gny + (size_t)j) * veclen + (size_t)k;
						std::complex<double> carg(cos(argvec[k]), sin(argvec[k]));
						std::complex<double> ccdata(RE(cdata[ii]), IM(cdata[ii]));
						if( fix )  ccdata = ccdata / std::abs(ccdata) / fftnormp;
						if( flip ) ccdata = -ccdata;
						ccdata = ccdata * Tkvec[k] * fftnorm * carg;
						RE(cdata[ii]) = ccdata.real();
						IM(cdata[ii]) = ccdata.imag();
					}
				}
			}

			delete[] kvec;
			delete[] Tkvec;
			delete[] argvec;
		}
	} else {
		// Cached real-space kernel: multiply by the rank-local k-space kernel
		// slab. The kernel was loaded and FFT'd in fetch_kernel(.., distributed=true)
		// with the same global (gnx,gny,gnz) and the same FFTW MPI layout, so
		// the per-rank complex indexing matches the data buffer one-to-one.
		fftw_complex *ckernel = reinterpret_cast<fftw_complex *>(pk->get_ptr());
#pragma omp parallel for
		for( size_t ix = 0; ix < local_nx; ++ix ){
			const int gix = (int)(local_x_off + ix);
			int kx = gix;
			if( kx > (int)gnx/2 ) kx -= (int)gnx;
			for( int j = 0; j < (int)gny; ++j ){
				int ky = j;
				if( ky > (int)gny/2 ) ky -= (int)gny;
				for( int k = 0; k < (int)nz_complex; ++k ){
					int kz = k;
					double arg = ((double)kx + (double)ky + (double)kz) * dstag;
					std::complex<double> carg(cos(arg), sin(arg));
					size_t ii = (ix*gny + (size_t)j) * nz_complex + (size_t)k;
					std::complex<double> ccdata  (RE(cdata[ii]),   IM(cdata[ii]));
					std::complex<double> cckernel(RE(ckernel[ii]), IM(ckernel[ii]));
					if( fix )  ccdata = ccdata / std::abs(ccdata);
					if( flip ) ccdata = -ccdata;
					ccdata = ccdata * cckernel * fftnorm * carg;
					RE(cdata[ii]) = ccdata.real();
					IM(cdata[ii]) = ccdata.imag();
				}
			}
		}
	}

	// ksampled path zeroes & re-adds the DC mode around the inverse FFT
	// (single-precision truncation guard). Cached path leaves DC intact.
	if( ksampled && local_x_off == 0 && local_nx > 0 ){
		RE(cdata[0]) = 0.0;
		IM(cdata[0]) = 0.0;
	}

	LOGUSER("Performing backward FFT (MPI)...");
	MUSIC::fft::execute(iplan);

	MUSIC::fft::destroy(plan);
	MUSIC::fft::destroy(iplan);

	if( ksampled ){
		// Re-add the DC mean. Each rank touches only its own logical cells
		// (skipping the inner-dim r2c padding). This differs slightly from the
		// serial perform() which strides through the padded buffer linearly; the
		// MPI variant uses an explicit (ix,jy,kz) loop so the rank-collective
		// result is well-defined.
		const size_t nz_padded = 2 * nz_complex;
		const real_t mean = (real_t)(dcmode.real() * fftnorm / (double)(gnx*gny*gnz));
#pragma omp parallel for
		for( size_t ix = 0; ix < local_nx; ++ix )
			for( size_t jy = 0; jy < gny; ++jy )
				for( size_t kz = 0; kz < gnz; ++kz )
					data[(ix*gny + jy)*nz_padded + kz] += mean;
	}
}

// perform_mpi reinterpret-casts the Meshvar buffer to fftw_real*, so the
// template parameter must match the FFTW precision. Only one instantiation
// is provided per build to make a mismatched call a link error.
#ifdef SINGLE_PRECISION
template void perform_mpi<float >(kernel *pk, Meshvar<float > *pdist, bool shift, bool fix, bool flip);
#else
template void perform_mpi<double>(kernel *pk, Meshvar<double> *pdist, bool shift, bool fix, bool flip);
#endif

/*****************************************************************************************/
/***    SCATTER / PERFORM_MPI / GATHER WRAPPER AROUND ROOT-OWNED PADDED GRID            ***/
/*****************************************************************************************/
//
// Root owns a contiguous padded buffer of shape (gnx, gny, 2*(gnz/2+1)) — the
// in-place r2c FFTW layout that DensityGrid already uses. This wrapper:
//   1. allocates a slab Meshvar on every rank (FFTW MPI layout, same padding)
//   2. MPI_Scatterv the root buffer to the per-rank slabs
//   3. calls perform_mpi() to do the collective convolution in-place
//   4. MPI_Gatherv the slabs back to the root buffer
// Workers may pass root_data = NULL; only rank 0's pointer is dereferenced.
//
// Falls back to scalar perform() when USE_MPI is undefined or world size == 1.

template <typename real_t>
void perform_dist(kernel *pk, real_t *root_data, size_t gnx, size_t gny, size_t gnz,
                  bool shift, bool fix, bool flip)
{
#ifndef USE_MPI
	(void)gnx; (void)gny; (void)gnz;
	convolution::perform<real_t>(pk, reinterpret_cast<void*>(root_data), shift, fix, flip);
#else
	if( MUSIC::mpi::size() == 1 ){
		convolution::perform<real_t>(pk, reinterpret_cast<void*>(root_data), shift, fix, flip);
		return;
	}

	const int rk = MUSIC::mpi::rank();
	const int sz = MUSIC::mpi::size();

	Meshvar<real_t> *slab = MUSIC::dist::make_slab_meshvar<real_t>(gnx, gny, gnz, /*fftw_inplace_pad=*/true);
	const size_t local_nx     = slab->local_nx();
	const size_t nz_padded    = 2*(gnz/2+1);
	const size_t plane        = gny * nz_padded;

	// Scatter/gather one x-slice (plane = gny*nz_padded reals) at a time via
	// a derived MPI datatype. counts/displs become x-slice units (== local_nx
	// per rank), which fit in int easily even for huge grids. The naive
	// real-count form overflowed int around 1968^3 (level-10 zoom): for n=4
	// displs[2] = 2*1.9e9 reals wraps to negative, corrupting MPI_Scatterv
	// destination offsets on rank 0.
	if( plane > (size_t)INT_MAX ){
		LOGERR("convolution::perform_dist: gny*nz_padded (%zu) exceeds INT_MAX; need a deeper datatype factorization.", plane);
		throw std::runtime_error("perform_dist: y-z plane too large for int count");
	}

	MPI_Datatype dtype_base = (sizeof(real_t) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
	MPI_Datatype dtype_plane;
	MPI_Type_contiguous((int)plane, dtype_base, &dtype_plane);
	MPI_Type_commit(&dtype_plane);

	std::vector<int> counts(sz), displs(sz);
	int my_count = (int)local_nx;
	MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MUSIC::mpi::world());
	displs[0] = 0;
	for( int i=1; i<sz; ++i ) displs[i] = displs[i-1] + counts[i-1];

	MPI_Scatterv( (rk==0) ? root_data : NULL, counts.data(), displs.data(), dtype_plane,
	              slab->m_pdata, my_count, dtype_plane,
	              0, MUSIC::mpi::world() );

	convolution::perform_mpi<real_t>(pk, slab, shift, fix, flip);

	MPI_Gatherv( slab->m_pdata, my_count, dtype_plane,
	             (rk==0) ? root_data : NULL, counts.data(), displs.data(), dtype_plane,
	             0, MUSIC::mpi::world() );

	MPI_Type_free(&dtype_plane);
	delete slab;
#endif
}

#ifdef SINGLE_PRECISION
template void perform_dist<float >(kernel *pk, float  *root_data, size_t gnx, size_t gny, size_t gnz, bool shift, bool fix, bool flip);
#else
template void perform_dist<double>(kernel *pk, double *root_data, size_t gnx, size_t gny, size_t gnz, bool shift, bool fix, bool flip);
#endif

/*****************************************************************************************/
/***    PERFORM_MPI / GATHER WRAPPER AROUND A PRE-DISTRIBUTED SLAB                      ***/
/*****************************************************************************************/
//
// Use this when the per-rank slab is already populated (e.g. each rank loaded
// its own noise slab via random_number_generator::load_slab). Skips the
// scatter, runs perform_mpi() in place, then gathers the convolved slabs back
// to root_data on rank 0. Workers may pass root_data = NULL.
//
// Falls back to plain perform() on slab->m_pdata when USE_MPI is undefined or
// MPI size == 1 (and copies the result into root_data when different).

template <typename real_t>
void perform_dist_slab(kernel *pk, Meshvar<real_t> *slab, real_t *root_data,
                       size_t gnx, size_t gny, size_t gnz,
                       bool shift, bool fix, bool flip)
{
	if( slab == NULL ){
		LOGERR("convolution::perform_dist_slab: slab is NULL.");
		throw std::runtime_error("perform_dist_slab: slab is NULL");
	}
#ifndef USE_MPI
	(void)gnx; (void)gny; (void)gnz;
	convolution::perform<real_t>(pk, reinterpret_cast<void*>(slab->m_pdata), shift, fix, flip);
	if( root_data && root_data != slab->m_pdata ){
		const size_t nz_padded = 2*(gnz/2+1);
		std::memcpy(root_data, slab->m_pdata, gnx*gny*nz_padded*sizeof(real_t));
	}
#else
	if( MUSIC::mpi::size() == 1 ){
		convolution::perform<real_t>(pk, reinterpret_cast<void*>(slab->m_pdata), shift, fix, flip);
		if( root_data && root_data != slab->m_pdata ){
			const size_t nz_padded = 2*(gnz/2+1);
			std::memcpy(root_data, slab->m_pdata, gnx*gny*nz_padded*sizeof(real_t));
		}
		return;
	}

	const int rk = MUSIC::mpi::rank();
	const int sz = MUSIC::mpi::size();
	const size_t local_nx    = slab->local_nx();
	const size_t nz_padded   = 2*(gnz/2+1);
	const size_t plane       = gny * nz_padded;

	// H.4: bcast rank-0's "do we want a gather?" decision. When rank 0 passes
	// root_data=NULL the result stays distributed (per-rank slabs); skip the
	// final Gatherv entirely so we never materialize a gnx*gny*nz_padded buffer
	// on rank 0.
	int gather_wanted = (rk==0) ? (root_data ? 1 : 0) : 0;
	MPI_Bcast(&gather_wanted, 1, MPI_INT, 0, MUSIC::mpi::world());

	convolution::perform_mpi<real_t>(pk, slab, shift, fix, flip);

	if( !gather_wanted ){
		return;
	}

	// Same int-count fix as perform_dist: gather x-slices using a derived
	// datatype so counts/displs stay in the small (gnx-sized) range.
	if( plane > (size_t)INT_MAX ){
		LOGERR("convolution::perform_dist_slab: gny*nz_padded (%zu) exceeds INT_MAX; need a deeper datatype factorization.", plane);
		throw std::runtime_error("perform_dist_slab: y-z plane too large for int count");
	}

	MPI_Datatype dtype_base = (sizeof(real_t) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;
	MPI_Datatype dtype_plane;
	MPI_Type_contiguous((int)plane, dtype_base, &dtype_plane);
	MPI_Type_commit(&dtype_plane);

	std::vector<int> counts(sz), displs(sz);
	int my_count = (int)local_nx;
	MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MUSIC::mpi::world());
	displs[0] = 0;
	for( int i=1; i<sz; ++i ) displs[i] = displs[i-1] + counts[i-1];

	MPI_Gatherv( slab->m_pdata, my_count, dtype_plane,
	             (rk==0) ? root_data : NULL, counts.data(), displs.data(), dtype_plane,
	             0, MUSIC::mpi::world() );

	MPI_Type_free(&dtype_plane);
#endif
}

#ifdef SINGLE_PRECISION
template void perform_dist_slab<float >(kernel *pk, Meshvar<float > *slab, float  *root_data, size_t gnx, size_t gny, size_t gnz, bool shift, bool fix, bool flip);
#else
template void perform_dist_slab<double>(kernel *pk, Meshvar<double> *slab, double *root_data, size_t gnx, size_t gny, size_t gnz, bool shift, bool fix, bool flip);
#endif

/*****************************************************************************************/
/***    SPECIFIC KERNEL IMPLEMENTATIONS      *********************************************/
/*****************************************************************************************/

template <typename real_t>
class kernel_k : public kernel
{
protected:
	/**/
	double boxlength_, patchlength_, nspec_, pnorm_, volfac_, kfac_, kmax_;
	TransferFunction_k *tfk_;

public:
	kernel_k(config_file &cf, transfer_function *ptf, refinement_hierarchy &refh, tf_type type)
		: kernel(cf, ptf, refh, type)
	{
		boxlength_ = pcf_->getValue<double>("setup", "boxlength");
		nspec_ = pcf_->getValue<double>("cosmology", "nspec");
		pnorm_ = pcf_->getValue<double>("cosmology", "pnorm");
		volfac_ = 1.0; //pow(boxlength,3)/pow(2.0*M_PI,3);
		kfac_ = 2.0 * M_PI / boxlength_;
		kmax_ = kfac_ / 2;
		tfk_ = new TransferFunction_k(type_, ptf_, nspec_, pnorm_);

		cparam_.nx = 1;
		cparam_.ny = 1;
		cparam_.nz = 1;
		cparam_.lx = boxlength_;
		cparam_.ly = boxlength_;
		cparam_.lz = boxlength_;
		cparam_.pcf = pcf_;
		patchlength_ = boxlength_;
	}

	kernel *fetch_kernel(int ilevel, bool isolated = false, bool /*distributed*/ = false)
	{
		// ksampled kernels need no buffer, so the distributed flag is moot.
		if (!isolated)
		{
			cparam_.nx = prefh_->size(ilevel, 0);
			cparam_.ny = prefh_->size(ilevel, 1);
			cparam_.nz = prefh_->size(ilevel, 2);

			cparam_.lx = (double)cparam_.nx / (double)(1 << ilevel) * boxlength_;
			cparam_.ly = (double)cparam_.ny / (double)(1 << ilevel) * boxlength_;
			cparam_.lz = (double)cparam_.nz / (double)(1 << ilevel) * boxlength_;

			patchlength_ = cparam_.lx;
			kfac_ = 2.0 * M_PI / patchlength_;
			kmax_ = kfac_ * cparam_.nx / 2;
		}
		else
		{
			cparam_.nx = 2 * prefh_->size(ilevel, 0);
			cparam_.ny = 2 * prefh_->size(ilevel, 1);
			cparam_.nz = 2 * prefh_->size(ilevel, 2);

			cparam_.lx = (double)cparam_.nx / (double)(1 << ilevel) * boxlength_;
			cparam_.ly = (double)cparam_.ny / (double)(1 << ilevel) * boxlength_;
			cparam_.lz = (double)cparam_.nz / (double)(1 << ilevel) * boxlength_;

			patchlength_ = cparam_.lx;
			kfac_ = 2.0 * M_PI / patchlength_;
			kmax_ = kfac_ * cparam_.nx / 2;
		}

		return this;
	}

	void *get_ptr() { return NULL; }

	bool is_ksampled() { return true; }

	void at_k(size_t len, const double *in_k, double *out_Tk)
	{
		for (size_t i = 0; i < len; ++i)
		{
			double kk = kfac_ * in_k[i];
			out_Tk[i] = volfac_ * tfk_->compute(kk);
		}
	}

	~kernel_k() { delete tfk_; }

	void deallocate() {}
};

////////////////////////////////////////////////////////////////////////////

template <typename real_t>
class kernel_real_cached : public kernel
{
protected:
	std::vector<real_t> kdata_;
	void precompute_kernel(transfer_function *ptf, tf_type type, const refinement_hierarchy &refh);
	void precompute_kernel_slab(transfer_function *ptf, tf_type type, const refinement_hierarchy &refh);

public:
	kernel_real_cached(config_file &cf, transfer_function *ptf, refinement_hierarchy &refh, tf_type type)
		: kernel(cf, ptf, refh, type)
	{
		// H.2 SPMD path: when setup.precompute_kernel_slab=yes AND np>1,
		// all ranks cooperate to sample, FFT-deconvolve and pwrite the
		// finest-level kernel on an FFTW3-MPI x-slab. H.2b adds the
		// distributed 8-pt fine→coarse averaging so multibox (levelmin !=
		// levelmax) is now supported too: the first coarse level averages
		// from the distributed fine slab, lower levels cascade rank-0-only.
		// Default-off: root-only legacy path preserved bit-identical.
		const bool want_slab = cf.getValueSafe<bool>("setup", "precompute_kernel_slab", false);
#ifdef USE_MPI
		if (want_slab && MUSIC::mpi::size() > 1)
		{
			precompute_kernel_slab(ptf, type, refh);
		}
		else
		{
			if (MUSIC::mpi::is_root())
				precompute_kernel(ptf, type, refh);
		}
		MPI_Barrier(MUSIC::mpi::world());
#else
		(void)want_slab;
		if (MUSIC::mpi::is_root())
			precompute_kernel(ptf, type, refh);
#endif
	}

	kernel *fetch_kernel(int ilevel, bool isolated = false, bool distributed = false);

	void *get_ptr()
	{
		return reinterpret_cast<void *>(&kdata_[0]);
	}

	bool is_ksampled()
	{
		return false;
	}

	void at_k(size_t, const double *, double *) {}

	~kernel_real_cached()
	{
		deallocate();
	}

	void deallocate()
	{
		std::vector<real_t>().swap(kdata_);
	}
};

template <typename real_t>
kernel *kernel_real_cached<real_t>::fetch_kernel(int ilevel, bool isolated, bool distributed)
{
	(void)isolated;
	char cachefname[128];
	sprintf(cachefname, "temp_kernel_level%03d.tmp", ilevel);

	if (MUSIC::mpi::is_root())
		std::cout << " - Fetching kernel for level " << ilevel << std::endl;
	LOGUSER("Loading kernel for level %3d from file \'%s\'...", ilevel, cachefname);

	FILE *fp = fopen(cachefname, "r");
	if (fp == NULL)
	{
		LOGERR("Could not open kernel file \'%s\'.", cachefname);
		throw std::runtime_error("Internal error: cached convolution kernel does not exist on disk!");
	}

	unsigned nx, ny, nz_padded;
	size_t nread = 0;
	nread = fread(reinterpret_cast<void *>(&nx),        sizeof(unsigned), 1, fp);
	nread = fread(reinterpret_cast<void *>(&ny),        sizeof(unsigned), 1, fp);
	nread = fread(reinterpret_cast<void *>(&nz_padded), sizeof(unsigned), 1, fp);
	(void)nread;

	const size_t header_bytes = 3 * sizeof(unsigned);
	const int logical_nz = (int)nz_padded - 2;

	//... set parameters
	double boxlength = pcf_->getValue<double>("setup", "boxlength");
	double dx = boxlength / (1 << ilevel);

	cparam_.nx = nx;
	cparam_.ny = ny;
	cparam_.nz = logical_nz;
	cparam_.lx = dx * cparam_.nx;
	cparam_.ly = dx * cparam_.ny;
	cparam_.lz = dx * cparam_.nz;
	cparam_.pcf = pcf_;

#ifdef USE_MPI
	if (distributed && MUSIC::mpi::size() > 1)
	{
		// Each rank reads its own x-slab from the cached file. The on-disk
		// layout (nx rows of ny*nz_padded reals) is exactly the per-rank
		// real-space layout FFTW MPI uses for an in-place r2c plan, so we
		// can fread directly into the start of the slab buffer.
		MUSIC::dist::slab_layout slab = MUSIC::dist::compute_slab_layout(
			(size_t)nx, (size_t)ny, (size_t)logical_nz, /*fftw_inplace_pad=*/true);

		kdata_.assign(slab.alloc_real_count, 0.0);

		if (slab.local_nx > 0)
		{
			const size_t row_reals = (size_t)ny * (size_t)nz_padded;
			const off_t  offset    = (off_t)(header_bytes + slab.local_x_start * row_reals * sizeof(fftw_real));
			if (fseeko(fp, offset, SEEK_SET) != 0)
			{
				LOGERR("fseeko on cached kernel file \'%s\' failed.", cachefname);
				fclose(fp);
				throw std::runtime_error("fetch_kernel: fseek failure");
			}
			const size_t nreals = slab.local_nx * row_reals;
			const size_t got    = fread(reinterpret_cast<void *>(&kdata_[0]),
			                            sizeof(fftw_real), nreals, fp);
			if (got != nreals)
			{
				LOGERR("Short read on cached kernel slab (rank=%d): expected %zu reals, got %zu",
				       MUSIC::mpi::rank(), nreals, got);
				fclose(fp);
				throw std::runtime_error("fetch_kernel: short read");
			}
		}
		fclose(fp);

		fftw_real *rkernel = reinterpret_cast<fftw_real *>(&kdata_[0]);
		MUSIC::fft::fft_plan_t plan = MUSIC::fft::plan_r2c_3d_mpi(
			(ptrdiff_t)nx, (ptrdiff_t)ny, (ptrdiff_t)logical_nz, rkernel);
		MUSIC::fft::execute(plan);
		MUSIC::fft::destroy(plan);
		return this;
	}
#endif

	// Serial single-rank path (legacy behaviour).
	(void)distributed;
	kdata_.assign((size_t)nx * (size_t)ny * (size_t)nz_padded, 0.0);
	for (size_t ix = 0; ix < nx; ++ix)
	{
		const size_t sz = (size_t)ny * (size_t)nz_padded;
		nread = fread(reinterpret_cast<void *>(&kdata_[ix * sz]), sizeof(fftw_real), sz, fp);
		assert(nread == sz);
	}
	fclose(fp);

	fftw_real *rkernel = reinterpret_cast<fftw_real *>(&kdata_[0]);

#ifdef FFTW3
#ifdef SINGLE_PRECISION
	fftwf_complex *kkernel = reinterpret_cast<fftwf_complex *>(&rkernel[0]);
	fftwf_plan plan = fftwf_plan_dft_r2c_3d(cparam_.nx, cparam_.ny, cparam_.nz, rkernel, kkernel, FFTW_ESTIMATE);
	fftwf_execute(plan);
	fftwf_destroy_plan(plan);
#else
	fftw_complex *kkernel = reinterpret_cast<fftw_complex *>(&rkernel[0]);
	fftw_plan plan = fftw_plan_dft_r2c_3d(cparam_.nx, cparam_.ny, cparam_.nz, rkernel, kkernel, FFTW_ESTIMATE);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
#endif
#else
	rfftwnd_plan plan = rfftw3d_create_plan(cparam_.nx, cparam_.ny, cparam_.nz,
											FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE | FFTW_IN_PLACE);

#ifndef SINGLETHREAD_FFTW
	rfftwnd_threads_one_real_to_complex(omp_get_max_threads(), plan, rkernel, NULL);
#else
	rfftwnd_one_real_to_complex(plan, rkernel, NULL);
#endif

	rfftwnd_destroy_plan(plan);
#endif
	return this;
}

template <typename real_t>
inline real_t sqr(real_t x)
{
	return x * x;
}

template <typename real_t>
inline real_t eval_split_recurse(const TransferFunction_real *tfr, real_t *xmid, real_t dx, real_t prevval, int nsplit)
{
	const real_t abs_err = 1e-8, rel_err = 1e-6;
	const int nmaxsplits = 12;

	real_t dxnew = dx / 2, dxnew2 = dx / 4;
	real_t dV = dxnew * dxnew * dxnew;

	real_t xl[3] = {xmid[0] - dxnew2, xmid[1] - dxnew2, xmid[2] - dxnew2};
	real_t xr[3] = {xmid[0] + dxnew2, xmid[1] + dxnew2, xmid[2] + dxnew2};

	real_t xc[8][3] =
		{
			{xl[0], xl[1], xl[2]},
			{xl[0], xl[1], xr[2]},
			{xl[0], xr[1], xl[2]},
			{xl[0], xr[1], xr[2]},
			{xr[0], xl[1], xl[2]},
			{xr[0], xl[1], xr[2]},
			{xr[0], xr[1], xl[2]},
			{xr[0], xr[1], xr[2]},
		};

	real_t rr2, res[8], ressum = 0.;

	for (int i = 0; i < 8; ++i)
	{
		rr2 = sqr(xc[i][0]) + sqr(xc[i][1]) + sqr(xc[i][2]);
		res[i] = tfr->compute_real(rr2) * dV;
		if (res[i] != res[i])
		{
			LOGERR("NaN encountered at r=%f, dx=%f, dV=%f : TF = %f", sqrt(rr2), dx, dV, res[i]);
			abort();
		}
		ressum += res[i];
	}

	real_t ae = fabs((prevval - ressum));
	real_t re = fabs(ae / ressum);

	if (ae < abs_err || re < rel_err)
		return ressum;

	if (nsplit > nmaxsplits)
	{
		//LOGWARN("reached maximum number of supdivisions in eval_split_recurse. Ending recursion... : abs. err.=%f, rel. err.=%f",ae, re);
		return ressum;
	}

	//otherwise keep splitting
	ressum = 0;
	for (int i = 0; i < 8; ++i)
		ressum += eval_split_recurse(tfr, xc[i], dxnew, res[i], nsplit + 1);

	return ressum;
}

template <typename real_t>
inline real_t eval_split_recurse(const TransferFunction_real *tfr, real_t *xmid, real_t dx, int nsplit = 0)
{
	//sqr(xmid[0])+sqr(xmid[1])+sqr(xmid[2])

	real_t rr2 = sqr(xmid[0]) + sqr(xmid[1]) + sqr(xmid[2]);
	real_t prevval = tfr->compute_real(rr2) * dx * dx * dx;
	return eval_split_recurse(tfr, xmid, dx, prevval, nsplit);
}

#define OLD_KERNEL_SAMPLING

template <typename real_t>
void kernel_real_cached<real_t>::precompute_kernel(transfer_function *ptf, tf_type type, const refinement_hierarchy &refh)
{
	//... compute kernel for finest level
	int nx, ny, nz;
	real_t dx, lx, ly, lz;

	real_t
		boxlength = pcf_->getValue<double>("setup", "boxlength"),
		boxlength2 = 0.5 * boxlength;

	int
		levelmax = refh.levelmax(),
		levelmin = refh.levelmin();

	LOGUSER("Precomputing transfer function kernels...");

	nx = refh.size(refh.levelmax(), 0);
	ny = refh.size(refh.levelmax(), 1);
	nz = refh.size(refh.levelmax(), 2);

	if (levelmax != levelmin)
	{
		nx *= 2;
		ny *= 2;
		nz *= 2;
	}

	dx = boxlength / (1 << refh.levelmax());
	lx = dx * nx;
	ly = dx * ny;
	lz = dx * nz;

	real_t
		kny = M_PI / dx,
		fac = lx * ly * lz / pow(2.0 * M_PI, 3) / ((double)nx * (double)ny * (double)nz),
		nspec = pcf_->getValue<double>("cosmology", "nspec"),
		pnorm = pcf_->getValue<double>("cosmology", "pnorm");

	bool
		bperiodic = pcf_->getValueSafe<bool>("setup", "periodic_TF", true),
		deconv = pcf_->getValueSafe<bool>("setup", "deconvolve", true);
	//		bool deconv_baryons = true;//pcf_->getValueSafe<bool>("setup","deconvolve_baryons",false) || do_SPH;
	bool bsmooth_baryons = false; //type==baryon && !deconv_baryons;
	//bool bbaryons = type==baryon | type==vbaryon;
	bool kspacepoisson = ((pcf_->getValueSafe<bool>("poisson", "fft_fine", true) |
						   pcf_->getValueSafe<bool>("poisson", "kspace", false))); // & !(type==baryon&!do_SPH);//&!baryons ;

	std::cout << "   - Computing transfer function kernel...\n";

	TransferFunction_real *tfr = new TransferFunction_real(boxlength, 1 << levelmax, type, ptf, nspec, pnorm,
														   0.25 * dx, 2.0 * boxlength, kny, (int)pow(2, levelmax + 2));

	fftw_real *rkernel = new fftw_real[(size_t)nx * (size_t)ny * ((size_t)nz + 2)], *rkernel_coarse;

#pragma omp parallel for
	for (int i = 0; i < nx; ++i)
		for (int j = 0; j < ny; ++j)
			for (int k = 0; k < nz; ++k)
			{
				size_t q = ((size_t)(i)*ny + (size_t)(j)) * (size_t)(nz + 2) + (size_t)(k);
				rkernel[q] = 0.0;
			}

	LOGUSER("Computing fine kernel (level %d)...", levelmax);

#ifdef OLD_KERNEL_SAMPLING
	int ref_fac = (deconv && kspacepoisson) ? 2 : 0;
	const int ql = -ref_fac / 2 + 1, qr = ql + ref_fac;
	const double rf8 = pow(ref_fac, 3);
	const double dx05 = 0.5 * dx, dx025 = 0.25 * dx;
#endif

	if (bperiodic)
	{
#pragma omp parallel for
		for (int i = 0; i <= nx / 2; ++i)
			for (int j = 0; j <= ny / 2; ++j)
				for (int k = 0; k <= nz / 2; ++k)
				{
					int iix(i), iiy(j), iiz(k);
					real_t rr[3];

					if (iix > (int)nx / 2)
						iix -= nx;
					if (iiy > (int)ny / 2)
						iiy -= ny;
					if (iiz > (int)nz / 2)
						iiz -= nz;

					//... speed up 8x by copying data to other octants
					size_t idx[8];

					idx[0] = ((size_t)(i)*ny + (size_t)(j)) * 2 * (nz / 2 + 1) + (size_t)(k);
					idx[1] = ((size_t)(nx - i) * ny + (size_t)(j)) * 2 * (nz / 2 + 1) + (size_t)(k);
					idx[2] = ((size_t)(i)*ny + (size_t)(ny - j)) * 2 * (nz / 2 + 1) + (size_t)(k);
					idx[3] = ((size_t)(nx - i) * ny + (size_t)(ny - j)) * 2 * (nz / 2 + 1) + (size_t)(k);
					idx[4] = ((size_t)(i)*ny + (size_t)(j)) * 2 * (nz / 2 + 1) + (size_t)(nz - k);
					idx[5] = ((size_t)(nx - i) * ny + (size_t)(j)) * 2 * (nz / 2 + 1) + (size_t)(nz - k);
					idx[6] = ((size_t)(i)*ny + (size_t)(ny - j)) * 2 * (nz / 2 + 1) + (size_t)(nz - k);
					idx[7] = ((size_t)(nx - i) * ny + (size_t)(ny - j)) * 2 * (nz / 2 + 1) + (size_t)(nz - k);

					if (i == 0 || i == nx / 2)
					{
						idx[1] = idx[3] = idx[5] = idx[7] = (size_t)-1;
					}
					if (j == 0 || j == ny / 2)
					{
						idx[2] = idx[3] = idx[6] = idx[7] = (size_t)-1;
					}
					if (k == 0 || k == nz / 2)
					{
						idx[4] = idx[5] = idx[6] = idx[7] = (size_t)-1;
					}

					double val = 0.0;

					for (int ii = -1; ii <= 1; ++ii)
						for (int jj = -1; jj <= 1; ++jj)
							for (int kk = -1; kk <= 1; ++kk)
							{
								rr[0] = ((double)iix) * dx + ii * boxlength;
								rr[1] = ((double)iiy) * dx + jj * boxlength;
								rr[2] = ((double)iiz) * dx + kk * boxlength;

								if (rr[0] > -boxlength && rr[0] <= boxlength && rr[1] > -boxlength && rr[1] <= boxlength && rr[2] > -boxlength && rr[2] <= boxlength)
								{
#ifdef OLD_KERNEL_SAMPLING
									if (ref_fac > 0)
									{
										double rrr[3];
										register double rrr2[3];
										for (int iii = ql; iii < qr; ++iii)
										{
											rrr[0] = rr[0] + (double)iii * dx05 - dx025;
											rrr2[0] = rrr[0] * rrr[0];
											for (int jjj = ql; jjj < qr; ++jjj)
											{
												rrr[1] = rr[1] + (double)jjj * dx05 - dx025;
												rrr2[1] = rrr[1] * rrr[1];
												for (int kkk = ql; kkk < qr; ++kkk)
												{
													rrr[2] = rr[2] + (double)kkk * dx05 - dx025;
													rrr2[2] = rrr[2] * rrr[2];
													val += tfr->compute_real(rrr2[0] + rrr2[1] + rrr2[2]) / rf8;
												}
											}
										}
									}
									else
									{
										val += tfr->compute_real(rr[0] * rr[0] + rr[1] * rr[1] + rr[2] * rr[2]);
									}

#else // !OLD_KERNEL_SAMPLING
									val += eval_split_recurse(tfr, rr, dx) / (dx * dx * dx);
#endif
								}
							}

					val *= fac;

					for (int q = 0; q < 8; ++q)
						if (idx[q] != (size_t)-1)
							rkernel[idx[q]] = val;
				}
	}
	else
	{
#pragma omp parallel for
		for (int i = 0; i < nx; ++i)
			for (int j = 0; j < ny; ++j)
				for (int k = 0; k < nz; ++k)
				{
					int iix(i), iiy(j), iiz(k);
					real_t rr[3];

					if (iix > (int)nx / 2)
						iix -= nx;
					if (iiy > (int)ny / 2)
						iiy -= ny;
					if (iiz > (int)nz / 2)
						iiz -= nz;

					//size_t idx = ((size_t)i*ny + (size_t)j) * 2*(nz/2+1) + (size_t)k;

					rr[0] = ((double)iix) * dx;
					rr[1] = ((double)iiy) * dx;
					rr[2] = ((double)iiz) * dx;

					//rkernel[idx] = 0.0;

					//rr2 = rr[0]*rr[0]+rr[1]*rr[1]+rr[2]*rr[2];

					//... speed up 8x by copying data to other octants
					size_t idx[8];

					idx[0] = ((size_t)(i)*ny + (size_t)(j)) * 2 * (nz / 2 + 1) + (size_t)(k);
					idx[1] = ((size_t)(nx - i) * ny + (size_t)(j)) * 2 * (nz / 2 + 1) + (size_t)(k);
					idx[2] = ((size_t)(i)*ny + (size_t)(ny - j)) * 2 * (nz / 2 + 1) + (size_t)(k);
					idx[3] = ((size_t)(nx - i) * ny + (size_t)(ny - j)) * 2 * (nz / 2 + 1) + (size_t)(k);
					idx[4] = ((size_t)(i)*ny + (size_t)(j)) * 2 * (nz / 2 + 1) + (size_t)(nz - k);
					idx[5] = ((size_t)(nx - i) * ny + (size_t)(j)) * 2 * (nz / 2 + 1) + (size_t)(nz - k);
					idx[6] = ((size_t)(i)*ny + (size_t)(ny - j)) * 2 * (nz / 2 + 1) + (size_t)(nz - k);
					idx[7] = ((size_t)(nx - i) * ny + (size_t)(ny - j)) * 2 * (nz / 2 + 1) + (size_t)(nz - k);

					if (i == 0 || i == nx / 2)
					{
						idx[1] = idx[3] = idx[5] = idx[7] = (size_t)-1;
					}
					if (j == 0 || j == ny / 2)
					{
						idx[2] = idx[3] = idx[6] = idx[7] = (size_t)-1;
					}
					if (k == 0 || k == nz / 2)
					{
						idx[4] = idx[5] = idx[6] = idx[7] = (size_t)-1;
					}

					double val = 0.0; //(fftw_real)tfr->compute_real(rr2)*fac;

#ifdef OLD_KERNEL_SAMPLING

					if (ref_fac > 0)
					{
						double rrr[3];
						register double rrr2[3];
						for (int iii = ql; iii < qr; ++iii)
						{
							rrr[0] = rr[0] + (double)iii * dx05 - dx025;
							rrr2[0] = rrr[0] * rrr[0];
							for (int jjj = ql; jjj < qr; ++jjj)
							{
								rrr[1] = rr[1] + (double)jjj * dx05 - dx025;
								rrr2[1] = rrr[1] * rrr[1];
								for (int kkk = ql; kkk < qr; ++kkk)
								{
									rrr[2] = rr[2] + (double)kkk * dx05 - dx025;
									rrr2[2] = rrr[2] * rrr[2];
									val += tfr->compute_real(rrr2[0] + rrr2[1] + rrr2[2]) / rf8;
								}
							}
						}
					}
					else
					{
						val = tfr->compute_real(rr[0] * rr[0] + rr[1] * rr[1] + rr[2] * rr[2]);
					}

#else
					if (i == 0 && j == 0 && k == 0)
						continue;

					// use new exact volume integration scheme
					val = eval_split_recurse(tfr, rr, dx) / (dx * dx * dx);

#endif

					//if( rr2 <= boxlength2*boxlength2 )
					//rkernel[idx] += (fftw_real)tfr->compute_real(rr2)*fac;
					val *= fac;

					for (int q = 0; q < 8; ++q)
						if (idx[q] != (size_t)-1)
							rkernel[idx[q]] = val;
				}
	}
	{
#ifdef OLD_KERNEL_SAMPLING
		rkernel[0] = tfr->compute_real(0.0) * fac;
#else
		real_t xmid[3] = {0.0, 0.0, 0.0};
		rkernel[0] = fac * eval_split_recurse(tfr, xmid, dx) / (dx * dx * dx);
#endif
	}
	/*************************************************************************************/
	/*************************************************************************************/
	/******* perform deconvolution *******************************************************/

	//bool baryons = type==baryon||type==vbaryon;
	if (deconv)
	{

		LOGUSER("Deconvolving fine kernel...");
		std::cout << " - Deconvolving density kernel...\n";

		double fftnorm = 1.0 / ((size_t)nx * (size_t)ny * (size_t)nz);
		double k0 = rkernel[0];

		fftw_complex *kkernel = reinterpret_cast<fftw_complex *>(&rkernel[0]);

		//... subtract white noise component before deconvolution
		if (!bsmooth_baryons)
			rkernel[0] = 0.0;

#ifdef FFTW3
#ifdef SINGLE_PRECISION
		fftwf_plan
			plan = fftwf_plan_dft_r2c_3d(nx, ny, nz, rkernel, kkernel, FFTW_ESTIMATE),
			iplan = fftwf_plan_dft_c2r_3d(nx, ny, nz, kkernel, rkernel, FFTW_ESTIMATE);

		fftwf_execute(plan);
#else
		fftw_plan
			plan = fftw_plan_dft_r2c_3d(nx, ny, nz, rkernel, kkernel, FFTW_ESTIMATE),
			iplan = fftw_plan_dft_c2r_3d(nx, ny, nz, kkernel, rkernel, FFTW_ESTIMATE);

		fftw_execute(plan);
#endif
#else
		rfftwnd_plan plan = rfftw3d_create_plan(nx, ny, nz, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE | FFTW_IN_PLACE),
					 iplan = rfftw3d_create_plan(nx, ny, nz, FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE | FFTW_IN_PLACE);

#ifndef SINGLETHREAD_FFTW
		rfftwnd_threads_one_real_to_complex(omp_get_max_threads(), plan, rkernel, NULL);
#else
		rfftwnd_one_real_to_complex(plan, rkernel, NULL);
#endif
#endif

		if (deconv)
		{

			double ksum = 0.0;
			size_t kcount = 0;
			double kmax = 0.5 * M_PI / std::max(nx, std::max(ny, nz));

#pragma omp parallel for reduction(+ \
								   : ksum, kcount)
			for (int i = 0; i < nx; ++i)
				for (int j = 0; j < ny; ++j)
					for (int k = 0; k < nz / 2 + 1; ++k)
					{
						double kx, ky, kz;

						kx = (double)i;
						ky = (double)j;
						kz = (double)k;

						if (kx > nx / 2)
							kx -= nx;
						if (ky > ny / 2)
							ky -= ny;

						double kkmax = kmax;
						size_t q = ((size_t)i * ny + (size_t)j) * (size_t)(nz / 2 + 1) + (size_t)k;

						if (!bsmooth_baryons)
						{
							if (kspacepoisson)
							{
								//... Use child average response function to emulate sub-sampling
								double ipix = cos(kx * kkmax) * cos(ky * kkmax) * cos(kz * kkmax);

								RE(kkernel[q]) /= ipix;
								IM(kkernel[q]) /= ipix;
							}
							else
							{

								//... Use piecewise constant response function (NGP-kernel)
								//... for finite difference methods
								kkmax = kmax;
								double ipix = 1.0;
								if (i > 0)
									ipix /= sin(kx * 2.0 * kkmax) / (kx * 2.0 * kkmax);
								if (j > 0)
									ipix /= sin(ky * 2.0 * kkmax) / (ky * 2.0 * kkmax);
								if (k > 0)
									ipix /= sin(kz * 2.0 * kkmax) / (kz * 2.0 * kkmax);

								RE(kkernel[q]) *= ipix;
								IM(kkernel[q]) *= ipix;
							}
						}
#if 1
						else
						{
							//... if smooth==true, convolve with
							//... NGP kernel to get CIC smoothness

							//kkmax *= 2.0;

							double ipix = 1.0;
							if (i > 0)
								ipix /= sin(kx * 2.0 * kkmax) / (kx * 2.0 * kkmax);
							if (j > 0)
								ipix /= sin(ky * 2.0 * kkmax) / (ky * 2.0 * kkmax);
							if (k > 0)
								ipix /= sin(kz * 2.0 * kkmax) / (kz * 2.0 * kkmax);

							RE(kkernel[q]) /= ipix;
							IM(kkernel[q]) /= ipix;
						}
#endif

						//... store k-space average
						if (k == 0 || k == nz / 2)
						{
							ksum += RE(kkernel[q]);
							kcount++;
						}
						else
						{
							ksum += 2.0 * (RE(kkernel[q]));
							kcount += 2;
						}
					}

			double dk;

			//... re-add white noise component for finest grid
			dk = k0 - ksum / kcount;

			//... set white noise component to zero if smoothing is enabled
			//if( false )//cparam_.smooth )
			if (bsmooth_baryons)
				dk = 0.0;

				//... enforce the r=0 component by adjusting the k-space mean
#pragma omp parallel for reduction(+ \
								   : ksum, kcount)
			for (int i = 0; i < nx; ++i)
				for (int j = 0; j < ny; ++j)
					for (int k = 0; k < (nz / 2 + 1); ++k)
					{
						size_t q = ((size_t)i * ny + (size_t)j) * (nz / 2 + 1) + (size_t)k;

						RE(kkernel[q]) += dk;

						RE(kkernel[q]) *= fftnorm;
						IM(kkernel[q]) *= fftnorm;
					}
		}

#ifdef FFTW3
#ifdef SINGLE_PRECISION
		fftwf_execute(iplan);
		fftwf_destroy_plan(plan);
		fftwf_destroy_plan(iplan);
#else
		fftw_execute(iplan);
		fftw_destroy_plan(plan);
		fftw_destroy_plan(iplan);
#endif
#else
#ifndef SINGLETHREAD_FFTW
		rfftwnd_threads_one_complex_to_real(omp_get_max_threads(), iplan, kkernel, NULL);
#else
		rfftwnd_one_complex_to_real(iplan, kkernel, NULL);
#endif
		rfftwnd_destroy_plan(plan);
		rfftwnd_destroy_plan(iplan);
#endif
	}

	/*************************************************************************************/
	/*************************************************************************************/
	/*************************************************************************************/

	char cachefname[128];
	sprintf(cachefname, "temp_kernel_level%03d.tmp", levelmax);
	LOGUSER("Storing kernel in temp file \'%s\'.", cachefname);

	FILE *fp = fopen(cachefname, "w+");
	unsigned q = nx;
	fwrite(reinterpret_cast<void *>(&q), sizeof(unsigned), 1, fp);
	q = ny;
	fwrite(reinterpret_cast<void *>(&q), sizeof(unsigned), 1, fp);
	q = 2 * (nz / 2 + 1);
	fwrite(reinterpret_cast<void *>(&q), sizeof(unsigned), 1, fp);

	for (int ix = 0; ix < nx; ++ix)
	{
		size_t sz = ny * 2 * (nz / 2 + 1);
		//fwrite( reinterpret_cast<void*>(&rkernel[0]), sizeof(fftw_real), nx*ny*2*(nz/2+1), fp );
		fwrite(reinterpret_cast<void *>(&rkernel[(size_t)ix * sz]), sizeof(fftw_real), sz, fp);
	}

	fclose(fp);

	//... average and fill for other levels
	for (int ilevel = levelmax - 1; ilevel >= levelmin; ilevel--)
	{
		LOGUSER("Computing coarse kernel (level %d)...", ilevel);

		int nxc, nyc, nzc;
		real_t dxc, lxc, lyc, lzc;

		nxc = refh.size(ilevel, 0);
		nyc = refh.size(ilevel, 1);
		nzc = refh.size(ilevel, 2);

		if (ilevel != levelmin)
		{
			nxc *= 2;
			nyc *= 2;
			nzc *= 2;
		}

		dxc = boxlength / (1 << ilevel);
		lxc = dxc * nxc;
		lyc = dxc * nyc;
		lzc = dxc * nzc;

		rkernel_coarse = new fftw_real[(size_t)nxc * (size_t)nyc * 2 * ((size_t)nzc / 2 + 1)];
		fac = lxc * lyc * lzc / pow(2.0 * M_PI, 3) / ((double)nxc * (double)nyc * (double)nzc);

		if (bperiodic)
		{
#pragma omp parallel for
			for (int i = 0; i <= nxc / 2; ++i)
				for (int j = 0; j <= nyc / 2; ++j)
					for (int k = 0; k <= nzc / 2; ++k)
					{
						int iix(i), iiy(j), iiz(k);
						real_t rr[3], rr2;

						if (iix > (int)nxc / 2)
							iix -= nxc;
						if (iiy > (int)nyc / 2)
							iiy -= nyc;
						if (iiz > (int)nzc / 2)
							iiz -= nzc;

						//... speed up 8x by copying data to other octants
						size_t idx[8];

						idx[0] = ((size_t)(i)*nyc + (size_t)(j)) * 2 * (nzc / 2 + 1) + (size_t)(k);
						idx[1] = ((size_t)(nxc - i) * nyc + (size_t)(j)) * 2 * (nzc / 2 + 1) + (size_t)(k);
						idx[2] = ((size_t)(i)*nyc + (size_t)(nyc - j)) * 2 * (nzc / 2 + 1) + (size_t)(k);
						idx[3] = ((size_t)(nxc - i) * nyc + (size_t)(nyc - j)) * 2 * (nzc / 2 + 1) + (size_t)(k);
						idx[4] = ((size_t)(i)*nyc + (size_t)(j)) * 2 * (nzc / 2 + 1) + (size_t)(nzc - k);
						idx[5] = ((size_t)(nxc - i) * nyc + (size_t)(j)) * 2 * (nzc / 2 + 1) + (size_t)(nzc - k);
						idx[6] = ((size_t)(i)*nyc + (size_t)(nyc - j)) * 2 * (nzc / 2 + 1) + (size_t)(nzc - k);
						idx[7] = ((size_t)(nxc - i) * nyc + (size_t)(nyc - j)) * 2 * (nzc / 2 + 1) + (size_t)(nzc - k);

						if (i == 0 || i == nxc / 2)
						{
							idx[1] = idx[3] = idx[5] = idx[7] = (size_t)-1;
						}
						if (j == 0 || j == nyc / 2)
						{
							idx[2] = idx[3] = idx[6] = idx[7] = (size_t)-1;
						}
						if (k == 0 || k == nzc / 2)
						{
							idx[4] = idx[5] = idx[6] = idx[7] = (size_t)-1;
						}

						double val = 0.0;

						for (int ii = -1; ii <= 1; ++ii)
							for (int jj = -1; jj <= 1; ++jj)
								for (int kk = -1; kk <= 1; ++kk)
								{
									rr[0] = ((double)iix) * dxc + ii * boxlength;
									rr[1] = ((double)iiy) * dxc + jj * boxlength;
									rr[2] = ((double)iiz) * dxc + kk * boxlength;

									if (rr[0] > -boxlength && rr[0] < boxlength && rr[1] > -boxlength && rr[1] < boxlength && rr[2] > -boxlength && rr[2] < boxlength)
									{
#ifdef OLD_KERNEL_SAMPLING
										rr2 = rr[0] * rr[0] + rr[1] * rr[1] + rr[2] * rr[2];
										val += tfr->compute_real(rr2);
#else // ! OLD_KERNEL_SAMPLING
										val += eval_split_recurse(tfr, rr, dxc) / (dxc * dxc * dxc);
#endif
									}
								}

						val *= fac;

						for (int qq = 0; qq < 8; ++qq)
							if (idx[qq] != (size_t)-1)
								rkernel_coarse[idx[qq]] = val;
					}
		}
		else
		{
#pragma omp parallel for
			for (int i = 0; i < nxc; ++i)
				for (int j = 0; j < nyc; ++j)
					for (int k = 0; k < nzc; ++k)
					{
						int iix(i), iiy(j), iiz(k);
						real_t rr[3];

						if (iix > (int)nxc / 2)
							iix -= nxc;
						if (iiy > (int)nyc / 2)
							iiy -= nyc;
						if (iiz > (int)nzc / 2)
							iiz -= nzc;

						size_t idx = ((size_t)i * nyc + (size_t)j) * 2 * (nzc / 2 + 1) + (size_t)k;

						rr[0] = ((double)iix) * dxc;
						rr[1] = ((double)iiy) * dxc;
						rr[2] = ((double)iiz) * dxc;

#ifdef OLD_KERNEL_SAMPLING
						rkernel_coarse[idx] = 0.0;

						real_t rr2 = rr[0] * rr[0] + rr[1] * rr[1] + rr[2] * rr[2];
						if (fabs(rr[0]) <= boxlength2 || fabs(rr[1]) <= boxlength2 || fabs(rr[2]) <= boxlength2)
							rkernel_coarse[idx] += (fftw_real)tfr->compute_real(rr2) * fac;
#else

						rkernel_coarse[idx] = 0.0;

						//if( i==0 && j==0 && k==0 ) continue;
						real_t val = eval_split_recurse(tfr, rr, dxc) / (dxc * dxc * dxc);

						if (fabs(rr[0]) <= boxlength2 || fabs(rr[1]) <= boxlength2 || fabs(rr[2]) <= boxlength2)
							rkernel_coarse[idx] += val * fac;

#endif
					}
		}

#ifdef OLD_KERNEL_SAMPLING
		LOGUSER("Averaging fine kernel to coarse kernel...");

//... copy averaged and convolved fine kernel to coarse kernel
#pragma omp parallel for
		for (int ix = 0; ix < nx; ix += 2)
			for (int iy = 0; iy < ny; iy += 2)
				for (int iz = 0; iz < nz; iz += 2)
				{
					int iix(ix / 2), iiy(iy / 2), iiz(iz / 2);
					if (ix > nx / 2)
						iix += nxc - nx / 2;
					if (iy > ny / 2)
						iiy += nyc - ny / 2;
					if (iz > nz / 2)
						iiz += nzc - nz / 2;

					if (ix == nx / 2 || iy == ny / 2 || iz == nz / 2)
						continue;

					for (int i = 0; i <= 1; ++i)
						for (int j = 0; j <= 1; ++j)
							for (int k = 0; k <= 1; ++k)
								if (i == 0 && k == 0 && j == 0)
									rkernel_coarse[ACC_RC(iix, iiy, iiz)] =
										0.125 * (rkernel[ACC_RF(ix - i, iy - j, iz - k)] + rkernel[ACC_RF(ix - i + 1, iy - j, iz - k)] + rkernel[ACC_RF(ix - i, iy - j + 1, iz - k)] + rkernel[ACC_RF(ix - i, iy - j, iz - k + 1)] + rkernel[ACC_RF(ix - i + 1, iy - j + 1, iz - k)] + rkernel[ACC_RF(ix - i + 1, iy - j, iz - k + 1)] + rkernel[ACC_RF(ix - i, iy - j + 1, iz - k + 1)] + rkernel[ACC_RF(ix - i + 1, iy - j + 1, iz - k + 1)]);

								else
								{

									rkernel_coarse[ACC_RC(iix, iiy, iiz)] +=
										0.125 * (rkernel[ACC_RF(ix - i, iy - j, iz - k)] + rkernel[ACC_RF(ix - i + 1, iy - j, iz - k)] + rkernel[ACC_RF(ix - i, iy - j + 1, iz - k)] + rkernel[ACC_RF(ix - i, iy - j, iz - k + 1)] + rkernel[ACC_RF(ix - i + 1, iy - j + 1, iz - k)] + rkernel[ACC_RF(ix - i + 1, iy - j, iz - k + 1)] + rkernel[ACC_RF(ix - i, iy - j + 1, iz - k + 1)] + rkernel[ACC_RF(ix - i + 1, iy - j + 1, iz - k + 1)]);
								}
				}

#endif // #OLD_KERNEL_SAMPLING
		sprintf(cachefname, "temp_kernel_level%03d.tmp", ilevel);
		LOGUSER("Storing kernel in temp file \'%s\'.", cachefname);
		fp = fopen(cachefname, "w+");
		q = nxc;
		fwrite(reinterpret_cast<void *>(&q), sizeof(unsigned), 1, fp);
		q = nyc;
		fwrite(reinterpret_cast<void *>(&q), sizeof(unsigned), 1, fp);
		q = 2 * (nzc / 2 + 1);
		fwrite(reinterpret_cast<void *>(&q), sizeof(unsigned), 1, fp);

		for (int ix = 0; ix < nxc; ++ix)
		{
			size_t sz = nyc * 2 * (nzc / 2 + 1);
			//fwrite( reinterpret_cast<void*>(&rkernel_coarse[0]), sizeof(fftw_real), nxc*nyc*2*(nzc/2+1), fp );
			fwrite(reinterpret_cast<void *>(&rkernel_coarse[ix * sz]), sizeof(fftw_real), sz, fp);
		}

		fclose(fp);

		delete[] rkernel;

		//... prepare for next iteration
		nx = nxc;
		ny = nyc;
		nz = nzc;
		lx = lxc;
		ly = lyc;
		lz = lzc;
		dx = dxc;
		rkernel = rkernel_coarse;
	}

	//... clean up
	delete[] rkernel;
}

//------------------------------------------------------------------------------
// H.2b — slab-aware fine→coarse 8-pt averaging.
//
// The legacy OLD_KERNEL_SAMPLING path (precompute_kernel) produces each coarse
// kernel level by overwriting its central footprint with an 8-point average of
// the (deconvolved) fine kernel one level up. In the slab path (H.2) the finest
// kernel is FFTW3-MPI x-slab-distributed and never rank-0 resident, so that
// averaging step was deferred. These helpers perform the average distributed.
//
// average_fine_full_to_coarse_serial: bit-exact transcription of the legacy
//   averaging block (convolution_kernel.cc OLD_KERNEL_SAMPLING) on a full,
//   rank-0-resident fine array. Used as the smoke reference.
//
// average_fine_slab_to_coarse: each rank averages from its fine x-slab plus a
//   1-plane left/right x-halo (periodic ring), writing the coarse cells whose
//   2³ blocks start on an x-plane it owns. Each coarse cell is produced by
//   exactly one rank in the same per-cell summation order as the serial path,
//   so the MPI_SUM merge into a pre-zeroed rank-0 buffer is bit-identical to
//   the serial reference (no FFT is involved in this step).
//------------------------------------------------------------------------------
static void average_fine_full_to_coarse_serial(const fftw_real *rkernel,
                                                int nx, int ny, int nz,
                                                fftw_real *rkernel_coarse,
                                                int nxc, int nyc, int nzc)
{
#pragma omp parallel for
	for (int ix = 0; ix < nx; ix += 2)
		for (int iy = 0; iy < ny; iy += 2)
			for (int iz = 0; iz < nz; iz += 2)
			{
				int iix(ix / 2), iiy(iy / 2), iiz(iz / 2);
				if (ix > nx / 2) iix += nxc - nx / 2;
				if (iy > ny / 2) iiy += nyc - ny / 2;
				if (iz > nz / 2) iiz += nzc - nz / 2;

				if (ix == nx / 2 || iy == ny / 2 || iz == nz / 2)
					continue;

				for (int i = 0; i <= 1; ++i)
					for (int j = 0; j <= 1; ++j)
						for (int k = 0; k <= 1; ++k)
							if (i == 0 && k == 0 && j == 0)
								rkernel_coarse[ACC_RC(iix, iiy, iiz)] =
									0.125 * (rkernel[ACC_RF(ix - i, iy - j, iz - k)] + rkernel[ACC_RF(ix - i + 1, iy - j, iz - k)] + rkernel[ACC_RF(ix - i, iy - j + 1, iz - k)] + rkernel[ACC_RF(ix - i, iy - j, iz - k + 1)] + rkernel[ACC_RF(ix - i + 1, iy - j + 1, iz - k)] + rkernel[ACC_RF(ix - i + 1, iy - j, iz - k + 1)] + rkernel[ACC_RF(ix - i, iy - j + 1, iz - k + 1)] + rkernel[ACC_RF(ix - i + 1, iy - j + 1, iz - k + 1)]);
							else
								rkernel_coarse[ACC_RC(iix, iiy, iiz)] +=
									0.125 * (rkernel[ACC_RF(ix - i, iy - j, iz - k)] + rkernel[ACC_RF(ix - i + 1, iy - j, iz - k)] + rkernel[ACC_RF(ix - i, iy - j + 1, iz - k)] + rkernel[ACC_RF(ix - i, iy - j, iz - k + 1)] + rkernel[ACC_RF(ix - i + 1, iy - j + 1, iz - k)] + rkernel[ACC_RF(ix - i + 1, iy - j, iz - k + 1)] + rkernel[ACC_RF(ix - i, iy - j + 1, iz - k + 1)] + rkernel[ACC_RF(ix - i + 1, iy - j + 1, iz - k + 1)]);
			}
}

#ifdef USE_MPI
// Distributed 8-pt average. coarse_root is the full coarse buffer on rank 0
// (must be pre-zeroed by the caller for the MPI_SUM merge); NULL on workers.
// Returns true on success, false if the slab layout is unusable (a rank with
// an empty slab breaks the halo ring — caller should fall back to rank-0).
// touch_root (root-only buffer, NULL on workers) receives, per coarse cell, the
// number of ranks that produced it via averaging (0 or 1 — each touched cell is
// owned by exactly one rank). The production caller uses it to OVERWRITE only
// the averaged cells while preserving the direct-sampled gap cells; the smoke
// passes a throwaway buffer.
static bool average_fine_slab_to_coarse(const fftw_real *fine_slab,
                                        const MUSIC::dist::slab_layout &slab,
                                        int nx, int ny, int nz,
                                        fftw_real *coarse_root,
                                        int nxc, int nyc, int nzc,
                                        unsigned char *touch_root)
{
	const int rank = MUSIC::mpi::rank();
	const int np   = MUSIC::mpi::size();
	const MPI_Datatype T = (sizeof(fftw_real) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;

	// Require a non-empty slab on every rank (ascending-x block decomposition,
	// the FFTW3-MPI default) so rank±1 are the x-adjacent neighbors.
	int lnx_min_local = (int)slab.local_nx;
	int lnx_min = 0;
	MPI_Allreduce(&lnx_min_local, &lnx_min, 1, MPI_INT, MPI_MIN, MUSIC::mpi::world());
	if (lnx_min < 1)
		return false;

	const int x0  = (int)slab.local_x_start;
	const int lnx = (int)slab.local_nx;
	const int x1  = x0 + lnx;

	const size_t nz_padded_f = 2 * ((size_t)nz / 2 + 1);
	const size_t plane = (size_t)ny * nz_padded_f;

	std::vector<fftw_real> left_halo(plane), right_halo(plane);

	const int left  = (rank - 1 + np) % np;
	const int right = (rank + 1) % np;

	// right_halo = right neighbour's first plane (global x = x1, periodic).
	MPI_Sendrecv(const_cast<fftw_real *>(fine_slab), (int)plane, T, left, 27100,
	             right_halo.data(), (int)plane, T, right, 27100,
	             MUSIC::mpi::world(), MPI_STATUS_IGNORE);
	// left_halo = left neighbour's last plane (global x = x0-1, periodic).
	MPI_Sendrecv(const_cast<fftw_real *>(fine_slab) + (size_t)(lnx - 1) * plane,
	             (int)plane, T, right, 27101,
	             left_halo.data(), (int)plane, T, left, 27101,
	             MUSIC::mpi::world(), MPI_STATUS_IGNORE);

	// Fine accessor over [x0-1, x1] (slab + 2 halos); y,z wrap locally.
	auto Ffine = [&](int gx, int gy, int gz) -> fftw_real {
		gy = (gy % ny + ny) % ny;
		gz = (gz % nz + nz) % nz;
		const size_t yz = (size_t)gy * nz_padded_f + (size_t)gz;
		if (gx < x0)  return left_halo[yz];
		if (gx >= x1) return right_halo[yz];
		return fine_slab[(size_t)(gx - x0) * plane + yz];
	};

	const size_t nz_padded_c = 2 * ((size_t)nzc / 2 + 1);
	const size_t coarse_size = (size_t)nxc * (size_t)nyc * nz_padded_c;
	std::vector<fftw_real> coarse_local(coarse_size, (fftw_real)0);
	std::vector<unsigned char> touch_local(coarse_size, (unsigned char)0);

	const int ix0 = (x0 % 2 == 0) ? x0 : x0 + 1; // first even global x we own
	for (int ix = ix0; ix < x1; ix += 2)
	{
		if (ix == nx / 2) continue;
		int iix = ix / 2;
		if (ix > nx / 2) iix += nxc - nx / 2;
		for (int iy = 0; iy < ny; iy += 2)
		{
			if (iy == ny / 2) continue;
			int iiy = iy / 2;
			if (iy > ny / 2) iiy += nyc - ny / 2;
			for (int iz = 0; iz < nz; iz += 2)
			{
				if (iz == nz / 2) continue;
				int iiz = iz / 2;
				if (iz > nz / 2) iiz += nzc - nz / 2;

				for (int i = 0; i <= 1; ++i)
					for (int j = 0; j <= 1; ++j)
						for (int k = 0; k <= 1; ++k)
						{
							fftw_real fsum =
								Ffine(ix - i, iy - j, iz - k) + Ffine(ix - i + 1, iy - j, iz - k) + Ffine(ix - i, iy - j + 1, iz - k) + Ffine(ix - i, iy - j, iz - k + 1) + Ffine(ix - i + 1, iy - j + 1, iz - k) + Ffine(ix - i + 1, iy - j, iz - k + 1) + Ffine(ix - i, iy - j + 1, iz - k + 1) + Ffine(ix - i + 1, iy - j + 1, iz - k + 1);
							if (i == 0 && j == 0 && k == 0)
							{
								coarse_local[ACC_RC(iix, iiy, iiz)] = 0.125 * fsum;
								touch_local[ACC_RC(iix, iiy, iiz)] = 1;
							}
							else
								coarse_local[ACC_RC(iix, iiy, iiz)] += 0.125 * fsum;
						}
			}
		}
	}

	MPI_Reduce(coarse_local.data(), coarse_root,
	           (int)coarse_local.size(), T, MPI_SUM, 0, MUSIC::mpi::world());
	MPI_Reduce(touch_local.data(), touch_root,
	           (int)touch_local.size(), MPI_UNSIGNED_CHAR, MPI_SUM, 0, MUSIC::mpi::world());
	return true;
}
#endif // USE_MPI

// H.2b standalone smoke: deterministic separable pattern on a fine x-slab,
// distributed 8-pt average vs serial legacy average on rank 0. Expects a
// bit-identical match (no FFT in this step). Returns 0 on PASS.
int run_h2b_avg_smoke(size_t Nfine)
{
	if (Nfine == 0 || (Nfine & 1u))
	{
		if (MUSIC::mpi::is_root())
			std::cerr << "[h2b-avg-smoke] Nfine must be > 0 and even\n";
		return 1;
	}
	const int nx = (int)Nfine, ny = (int)Nfine, nz = (int)Nfine;
	const int nxc = (int)Nfine, nyc = (int)Nfine, nzc = (int)Nfine;
	const size_t nz_padded_f = 2 * ((size_t)nz / 2 + 1);
	const size_t nz_padded_c = 2 * ((size_t)nzc / 2 + 1);
	const bool is_root = MUSIC::mpi::is_root();
	const double TWO_PI = 2.0 * M_PI;

	auto pattern = [&](int gx, int gy, int gz) -> fftw_real {
		double sx = std::sin(TWO_PI * gx / nx) + 0.5 * std::cos(2.0 * TWO_PI * gx / nx);
		double sy = std::sin(TWO_PI * gy / ny) + 0.3 * std::cos(3.0 * TWO_PI * gy / ny);
		double sz = std::sin(TWO_PI * gz / nz) + 0.2 * std::cos(4.0 * TWO_PI * gz / nz);
		return (fftw_real)(sx * sy * sz);
	};

#ifdef USE_MPI
	MUSIC::dist::slab_layout slab = MUSIC::dist::compute_slab_layout(
		(size_t)nx, (size_t)ny, (size_t)nz, /*fftw_inplace_pad=*/true);

	std::vector<fftw_real> fine_slab(slab.alloc_real_count, (fftw_real)0);
#pragma omp parallel for
	for (int ixl = 0; ixl < (int)slab.local_nx; ++ixl)
	{
		int gx = (int)slab.local_x_start + ixl;
		for (int gy = 0; gy < ny; ++gy)
			for (int gz = 0; gz < nz; ++gz)
				fine_slab[((size_t)ixl * ny + gy) * nz_padded_f + gz] = pattern(gx, gy, gz);
	}

	std::vector<fftw_real> coarse_ref, coarse_dist, fine_full;
	if (is_root)
	{
		fine_full.assign((size_t)nx * ny * nz_padded_f, (fftw_real)0);
#pragma omp parallel for
		for (int gx = 0; gx < nx; ++gx)
			for (int gy = 0; gy < ny; ++gy)
				for (int gz = 0; gz < nz; ++gz)
					fine_full[((size_t)gx * ny + gy) * nz_padded_f + gz] = pattern(gx, gy, gz);
		coarse_ref.assign((size_t)nxc * nyc * nz_padded_c, (fftw_real)0);
		coarse_dist.assign((size_t)nxc * nyc * nz_padded_c, (fftw_real)0);
		average_fine_full_to_coarse_serial(fine_full.data(), nx, ny, nz,
		                                   coarse_ref.data(), nxc, nyc, nzc);
	}
	std::vector<unsigned char> touch_throwaway;
	if (is_root)
		touch_throwaway.assign((size_t)nxc * nyc * nz_padded_c, (unsigned char)0);

	bool ok = average_fine_slab_to_coarse(fine_slab.data(), slab, nx, ny, nz,
	                                      is_root ? coarse_dist.data() : NULL,
	                                      nxc, nyc, nzc,
	                                      is_root ? touch_throwaway.data() : NULL);
	int ok_all = ok ? 0 : 1;
	MPI_Allreduce(MPI_IN_PLACE, &ok_all, 1, MPI_INT, MPI_MAX, MUSIC::mpi::world());
	if (ok_all)
	{
		if (is_root)
			std::cerr << "[h2b-avg-smoke] slab layout unusable (empty slab on some rank); pick smaller np\n";
		return 1;
	}

	int rc = 0;
	if (is_root)
	{
		double max_abs_diff = 0.0, max_abs_ref = 0.0;
		for (size_t q = 0; q < coarse_ref.size(); ++q)
		{
			double a = (double)coarse_ref[q], b = (double)coarse_dist[q];
			double d = std::fabs(a - b);
			if (d > max_abs_diff) max_abs_diff = d;
			if (std::fabs(a) > max_abs_ref) max_abs_ref = std::fabs(a);
		}
		bool pass = (max_abs_diff == 0.0);
		std::cerr << "[h2b-avg-smoke] Nfine=" << Nfine
		          << " ranks=" << MUSIC::mpi::size()
		          << " max|ref|=" << max_abs_ref
		          << " max|dist-ref|=" << max_abs_diff
		          << " " << (pass ? "PASS" : "FAIL") << "\n";
		rc = pass ? 0 : 2;
	}
	MPI_Bcast(&rc, 1, MPI_INT, 0, MUSIC::mpi::world());
	return rc;
#else
	std::vector<fftw_real> fine_full((size_t)nx * ny * nz_padded_f, (fftw_real)0);
	for (int gx = 0; gx < nx; ++gx)
		for (int gy = 0; gy < ny; ++gy)
			for (int gz = 0; gz < nz; ++gz)
				fine_full[((size_t)gx * ny + gy) * nz_padded_f + gz] = pattern(gx, gy, gz);
	std::vector<fftw_real> coarse_ref((size_t)nxc * nyc * nz_padded_c, (fftw_real)0);
	average_fine_full_to_coarse_serial(fine_full.data(), nx, ny, nz, coarse_ref.data(), nxc, nyc, nzc);
	std::cerr << "[h2b-avg-smoke] Nfine=" << Nfine << " serial-only PASS\n";
	return 0;
#endif
}

// H.2 — SPMD slab-distributed finest-level kernel precompute.
// All ranks participate; each rank samples its FFTW3-MPI x-slab, runs
// a distributed r2c+c2r deconvolution and pwrites its slab to the
// temp_kernel_levelNNN.tmp cache file (rank 0 first writes the 3-unsigned
// header and pre-extends the file via ftruncate). Coarse levels are
// re-sampled rank-0-only (the OLD_KERNEL_SAMPLING 8-pt averaging path
// needs the fine kernel, which is no longer rank-0 resident here, so
// we use direct tfr->compute_real for coarse — same as the
// !OLD_KERNEL_SAMPLING branch already does for coarse levels).
template <typename real_t>
void kernel_real_cached<real_t>::precompute_kernel_slab(transfer_function *ptf, tf_type type, const refinement_hierarchy &refh)
{
#ifdef USE_MPI
	int nx, ny, nz;
	real_t dx, lx, ly, lz;

	real_t
		boxlength = pcf_->getValue<double>("setup", "boxlength"),
		boxlength2 = 0.5 * boxlength;

	int
		levelmax = refh.levelmax(),
		levelmin = refh.levelmin();

	LOGUSER("Precomputing transfer function kernels (slab path)...");

	nx = refh.size(refh.levelmax(), 0);
	ny = refh.size(refh.levelmax(), 1);
	nz = refh.size(refh.levelmax(), 2);

	if (levelmax != levelmin)
	{
		nx *= 2;
		ny *= 2;
		nz *= 2;
	}

	dx = boxlength / (1 << refh.levelmax());
	lx = dx * nx;
	ly = dx * ny;
	lz = dx * nz;

	real_t
		kny = M_PI / dx,
		fac = lx * ly * lz / pow(2.0 * M_PI, 3) / ((double)nx * (double)ny * (double)nz),
		nspec = pcf_->getValue<double>("cosmology", "nspec"),
		pnorm = pcf_->getValue<double>("cosmology", "pnorm");

	bool
		bperiodic = pcf_->getValueSafe<bool>("setup", "periodic_TF", true),
		deconv    = pcf_->getValueSafe<bool>("setup", "deconvolve", true);
	bool bsmooth_baryons = false;
	bool kspacepoisson = ((pcf_->getValueSafe<bool>("poisson", "fft_fine", true) |
						   pcf_->getValueSafe<bool>("poisson", "kspace", false)));

	if (MUSIC::mpi::is_root())
		std::cout << "   - Computing transfer function kernel (slab path; np=" << MUSIC::mpi::size() << ")...\n";

	TransferFunction_real *tfr = new TransferFunction_real(boxlength, 1 << levelmax, type, ptf, nspec, pnorm,
														   0.25 * dx, 2.0 * boxlength, kny, (int)pow(2, levelmax + 2));

	// FFTW3-MPI x-slab layout for the finest grid (r2c in-place padded inner dim).
	MUSIC::dist::slab_layout slab = MUSIC::dist::compute_slab_layout(
		(size_t)nx, (size_t)ny, (size_t)nz, /*fftw_inplace_pad=*/true);

	const size_t nz_padded  = 2 * ((size_t)nz / 2 + 1);
	const size_t nz_complex = (size_t)nz / 2 + 1;

	fftw_real *rkernel = new fftw_real[slab.alloc_real_count];
	std::memset(rkernel, 0, slab.alloc_real_count * sizeof(fftw_real));

	LOGUSER("Sampling fine kernel slab (level %d, local_nx=%zu @ x=%zu)...",
	        levelmax, slab.local_nx, slab.local_x_start);

#ifdef OLD_KERNEL_SAMPLING
	const int ref_fac_slab = (deconv && kspacepoisson) ? 2 : 0;
	const int ql_slab = -ref_fac_slab / 2 + 1, qr_slab = ql_slab + ref_fac_slab;
	const double rf8_slab = pow((double)ref_fac_slab, 3);
	const double dx05_slab = 0.5 * dx, dx025_slab = 0.25 * dx;
#endif

	// Sampling: each rank loops its full x-slab. We mirror within-slab
	// (j, k) octants only — the i↔(nx-i) mirror crosses ranks, so we
	// instead sample every i in the local slab directly.
	if (bperiodic)
	{
#pragma omp parallel for
		for (size_t ix_local = 0; ix_local < slab.local_nx; ++ix_local)
		{
			int i = (int)(slab.local_x_start + ix_local);
			for (int j = 0; j <= ny / 2; ++j)
				for (int k = 0; k <= nz / 2; ++k)
				{
					int iix(i), iiy(j), iiz(k);
					real_t rr[3];

					if (iix > nx / 2)
						iix -= nx;
					if (iiy > ny / 2)
						iiy -= ny;
					if (iiz > nz / 2)
						iiz -= nz;

					// 4-way (j, k) mirror within the local slab.
					size_t idx[4];
					idx[0] = (ix_local * (size_t)ny + (size_t)j)        * nz_padded + (size_t)k;
					idx[1] = (ix_local * (size_t)ny + (size_t)(ny - j)) * nz_padded + (size_t)k;
					idx[2] = (ix_local * (size_t)ny + (size_t)j)        * nz_padded + (size_t)(nz - k);
					idx[3] = (ix_local * (size_t)ny + (size_t)(ny - j)) * nz_padded + (size_t)(nz - k);

					if (j == 0 || j == ny / 2) idx[1] = idx[3] = (size_t)-1;
					if (k == 0 || k == nz / 2) idx[2] = idx[3] = (size_t)-1;

					double val = 0.0;
					for (int ii = -1; ii <= 1; ++ii)
						for (int jj = -1; jj <= 1; ++jj)
							for (int kk = -1; kk <= 1; ++kk)
							{
								rr[0] = ((double)iix) * dx + ii * boxlength;
								rr[1] = ((double)iiy) * dx + jj * boxlength;
								rr[2] = ((double)iiz) * dx + kk * boxlength;
								if (rr[0] > -boxlength && rr[0] <= boxlength &&
								    rr[1] > -boxlength && rr[1] <= boxlength &&
								    rr[2] > -boxlength && rr[2] <= boxlength)
								{
#ifdef OLD_KERNEL_SAMPLING
									if (ref_fac_slab > 0)
									{
										double rrr[3], rrr2[3];
										for (int iii = ql_slab; iii < qr_slab; ++iii)
										{
											rrr[0] = rr[0] + (double)iii * dx05_slab - dx025_slab;
											rrr2[0] = rrr[0] * rrr[0];
											for (int jjj = ql_slab; jjj < qr_slab; ++jjj)
											{
												rrr[1] = rr[1] + (double)jjj * dx05_slab - dx025_slab;
												rrr2[1] = rrr[1] * rrr[1];
												for (int kkk = ql_slab; kkk < qr_slab; ++kkk)
												{
													rrr[2] = rr[2] + (double)kkk * dx05_slab - dx025_slab;
													rrr2[2] = rrr[2] * rrr[2];
													val += tfr->compute_real(rrr2[0] + rrr2[1] + rrr2[2]) / rf8_slab;
												}
											}
										}
									}
									else
									{
										val += tfr->compute_real(rr[0]*rr[0] + rr[1]*rr[1] + rr[2]*rr[2]);
									}
#else
									val += eval_split_recurse(tfr, rr, dx) / (dx * dx * dx);
#endif
								}
							}

					val *= fac;
					for (int q = 0; q < 4; ++q)
						if (idx[q] != (size_t)-1)
							rkernel[idx[q]] = val;
				}
		}
	}
	else
	{
		// Non-periodic: sample every cell in the local slab directly.
#pragma omp parallel for
		for (size_t ix_local = 0; ix_local < slab.local_nx; ++ix_local)
		{
			int i = (int)(slab.local_x_start + ix_local);
			for (int j = 0; j < ny; ++j)
				for (int k = 0; k < nz; ++k)
				{
					int iix(i), iiy(j), iiz(k);
					real_t rr[3];

					if (iix > nx / 2) iix -= nx;
					if (iiy > ny / 2) iiy -= ny;
					if (iiz > nz / 2) iiz -= nz;

					rr[0] = (double)iix * dx;
					rr[1] = (double)iiy * dx;
					rr[2] = (double)iiz * dx;

					size_t idx = (ix_local * (size_t)ny + (size_t)j) * nz_padded + (size_t)k;
					double val = 0.0;
#ifdef OLD_KERNEL_SAMPLING
					val = tfr->compute_real(rr[0]*rr[0] + rr[1]*rr[1] + rr[2]*rr[2]);
#else
					if (i == 0 && j == 0 && k == 0)
						continue;
					val = eval_split_recurse(tfr, rr, dx) / (dx * dx * dx);
#endif
					val *= fac;
					rkernel[idx] = val;
				}
		}
	}

	// Pin the (0,0,0) cell on the rank that owns x=0.
	if (slab.local_x_start == 0 && slab.local_nx > 0)
	{
#ifdef OLD_KERNEL_SAMPLING
		rkernel[0] = tfr->compute_real(0.0) * fac;
#else
		real_t xmid[3] = {0.0, 0.0, 0.0};
		rkernel[0] = fac * eval_split_recurse(tfr, xmid, dx) / (dx * dx * dx);
#endif
	}

	// Deconvolution: distributed r2c, per-slab k-space loop, distributed c2r.
	if (deconv)
	{
		LOGUSER("Deconvolving fine kernel (slab path)...");
		if (MUSIC::mpi::is_root())
			std::cout << " - Deconvolving density kernel (slab path)...\n";

		const double fftnorm = 1.0 / ((size_t)nx * (size_t)ny * (size_t)nz);

		// k0 lives on the rank owning x=0 (and is rkernel[0]).
		double k0_local = 0.0;
		if (slab.local_x_start == 0 && slab.local_nx > 0)
			k0_local = (double)rkernel[0];
		double k0 = 0.0;
		MPI_Allreduce(&k0_local, &k0, 1, MPI_DOUBLE, MPI_SUM, MUSIC::mpi::world());

		// Subtract white-noise component prior to forward FFT.
		if (!bsmooth_baryons && slab.local_x_start == 0 && slab.local_nx > 0)
			rkernel[0] = 0.0;

		MUSIC::fft::fft_plan_t plan = MUSIC::fft::plan_r2c_3d_mpi(
			(ptrdiff_t)nx, (ptrdiff_t)ny, (ptrdiff_t)nz, rkernel);
		MUSIC::fft::execute(plan);
		MUSIC::fft::destroy(plan);

		MUSIC::fft::fft_cplx_t *kkernel = reinterpret_cast<MUSIC::fft::fft_cplx_t *>(rkernel);

		double ksum_local = 0.0;
		unsigned long long kcount_local = 0;
		const double kmax = 0.5 * M_PI / std::max(nx, std::max(ny, nz));

#pragma omp parallel for reduction(+ : ksum_local, kcount_local)
		for (size_t ix_local = 0; ix_local < slab.local_nx; ++ix_local)
		{
			int i = (int)(slab.local_x_start + ix_local);
			for (int j = 0; j < ny; ++j)
				for (int k = 0; k < (int)nz_complex; ++k)
				{
					double kx = (double)i, ky = (double)j, kz = (double)k;
					if (kx > nx / 2) kx -= nx;
					if (ky > ny / 2) ky -= ny;

					double kkmax = kmax;
					size_t q = (ix_local * (size_t)ny + (size_t)j) * nz_complex + (size_t)k;

					if (!bsmooth_baryons)
					{
						if (kspacepoisson)
						{
							double ipix = cos(kx * kkmax) * cos(ky * kkmax) * cos(kz * kkmax);
							RE(kkernel[q]) /= ipix;
							IM(kkernel[q]) /= ipix;
						}
						else
						{
							kkmax = kmax;
							double ipix = 1.0;
							if (i > 0) ipix /= sin(kx * 2.0 * kkmax) / (kx * 2.0 * kkmax);
							if (j > 0) ipix /= sin(ky * 2.0 * kkmax) / (ky * 2.0 * kkmax);
							if (k > 0) ipix /= sin(kz * 2.0 * kkmax) / (kz * 2.0 * kkmax);
							RE(kkernel[q]) *= ipix;
							IM(kkernel[q]) *= ipix;
						}
					}
					else
					{
						double ipix = 1.0;
						if (i > 0) ipix /= sin(kx * 2.0 * kkmax) / (kx * 2.0 * kkmax);
						if (j > 0) ipix /= sin(ky * 2.0 * kkmax) / (ky * 2.0 * kkmax);
						if (k > 0) ipix /= sin(kz * 2.0 * kkmax) / (kz * 2.0 * kkmax);
						RE(kkernel[q]) /= ipix;
						IM(kkernel[q]) /= ipix;
					}

					if (k == 0 || k == nz / 2)
					{
						ksum_local += RE(kkernel[q]);
						kcount_local++;
					}
					else
					{
						ksum_local += 2.0 * RE(kkernel[q]);
						kcount_local += 2;
					}
				}
		}

		double ksum = 0.0;
		unsigned long long kcount = 0;
		MPI_Allreduce(&ksum_local, &ksum, 1, MPI_DOUBLE, MPI_SUM, MUSIC::mpi::world());
		MPI_Allreduce(&kcount_local, &kcount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MUSIC::mpi::world());

		double dk = k0 - ksum / (double)kcount;
		if (bsmooth_baryons) dk = 0.0;

#pragma omp parallel for
		for (size_t ix_local = 0; ix_local < slab.local_nx; ++ix_local)
		{
			for (int j = 0; j < ny; ++j)
				for (int k = 0; k < (int)nz_complex; ++k)
				{
					size_t q = (ix_local * (size_t)ny + (size_t)j) * nz_complex + (size_t)k;
					RE(kkernel[q]) += dk;
					RE(kkernel[q]) *= fftnorm;
					IM(kkernel[q]) *= fftnorm;
				}
		}

		MUSIC::fft::fft_plan_t iplan = MUSIC::fft::plan_c2r_3d_mpi(
			(ptrdiff_t)nx, (ptrdiff_t)ny, (ptrdiff_t)nz, rkernel);
		MUSIC::fft::execute(iplan);
		MUSIC::fft::destroy(iplan);
	}

	// Write the cached kernel: rank 0 writes the 3-unsigned header and
	// ftruncate-extends to total bytes; all ranks then pwrite their slab.
	char cachefname[128];
	sprintf(cachefname, "temp_kernel_level%03d.tmp", levelmax);
	LOGUSER("Storing fine kernel in temp file '%s' (slab pwrite).", cachefname);

	const size_t header_bytes = 3 * sizeof(unsigned);
	const size_t row_reals    = (size_t)ny * nz_padded;

	if (MUSIC::mpi::is_root())
	{
		int fd_init = ::open(cachefname, O_WRONLY | O_CREAT | O_TRUNC, 0644);
		if (fd_init < 0)
		{
			LOGERR("precompute_kernel_slab: cannot create '%s': %s", cachefname, std::strerror(errno));
			throw std::runtime_error("precompute_kernel_slab: cache file create failed");
		}
		const unsigned hdr[3] = { (unsigned)nx, (unsigned)ny, (unsigned)nz_padded };
		ssize_t hw = ::write(fd_init, hdr, sizeof(hdr));
		if (hw != (ssize_t)sizeof(hdr))
		{
			::close(fd_init);
			LOGERR("precompute_kernel_slab: header write short on '%s'", cachefname);
			throw std::runtime_error("precompute_kernel_slab: header write failed");
		}
		const off_t total_bytes = (off_t)header_bytes + (off_t)nx * (off_t)row_reals * (off_t)sizeof(fftw_real);
		if (::ftruncate(fd_init, total_bytes) != 0)
		{
			::close(fd_init);
			LOGERR("precompute_kernel_slab: ftruncate failed on '%s': %s", cachefname, std::strerror(errno));
			throw std::runtime_error("precompute_kernel_slab: ftruncate failed");
		}
		::close(fd_init);
	}
	MPI_Barrier(MUSIC::mpi::world());

	if (slab.local_nx > 0)
	{
		int fd = ::open(cachefname, O_WRONLY, 0644);
		if (fd < 0)
		{
			LOGERR("precompute_kernel_slab: rank %d cannot open '%s': %s",
			       MUSIC::mpi::rank(), cachefname, std::strerror(errno));
			throw std::runtime_error("precompute_kernel_slab: rank open failed");
		}
		const off_t offset = (off_t)header_bytes + (off_t)slab.local_x_start * (off_t)row_reals * (off_t)sizeof(fftw_real);
		// pwrite slab one x-row at a time to skip the FFTW r2c tail-padding
		// columns when local_nx*row_reals would exceed alloc; but
		// alloc_real_count >= local_nx*row_reals, so a single pwrite is safe.
		const size_t nbytes = slab.local_nx * row_reals * sizeof(fftw_real);
		ssize_t got = ::pwrite(fd, rkernel, nbytes, offset);
		if (got < 0 || (size_t)got != nbytes)
		{
			::close(fd);
			LOGERR("precompute_kernel_slab: rank %d short pwrite on '%s' (got %zd of %zu): %s",
			       MUSIC::mpi::rank(), cachefname, got, nbytes, std::strerror(errno));
			throw std::runtime_error("precompute_kernel_slab: short pwrite");
		}
		::close(fd);
	}
	MPI_Barrier(MUSIC::mpi::world());

	// H.2b: the first coarse level (levelmax-1) is produced by the legacy
	// OLD_KERNEL_SAMPLING 8-pt average of the deconvolved fine kernel. That
	// fine kernel is the x-slab just computed, so the averaging is distributed
	// across all ranks and merged into a rank-0 coarse buffer; lower coarse
	// levels then cascade rank-0-only from the previous coarse kernel, exactly
	// like the legacy precompute_kernel. touch1 marks the averaged (overwritten)
	// cells so the direct-sampled gap cells are preserved.
	int nxc1 = refh.size(levelmax - 1, 0);
	int nyc1 = refh.size(levelmax - 1, 1);
	int nzc1 = refh.size(levelmax - 1, 2);
	if (levelmax - 1 != levelmin) { nxc1 *= 2; nyc1 *= 2; nzc1 *= 2; }
	const size_t avg1_size = (size_t)nxc1 * (size_t)nyc1 * 2 * ((size_t)nzc1 / 2 + 1);
	std::vector<fftw_real>     avg1;
	std::vector<unsigned char> touch1;
	if (MUSIC::mpi::is_root()) { avg1.assign(avg1_size, (fftw_real)0); touch1.assign(avg1_size, (unsigned char)0); }
#ifdef OLD_KERNEL_SAMPLING
	bool avg1_ok = average_fine_slab_to_coarse(rkernel, slab, nx, ny, nz,
	                   MUSIC::mpi::is_root() ? avg1.data() : NULL, nxc1, nyc1, nzc1,
	                   MUSIC::mpi::is_root() ? touch1.data() : NULL);
	if (!avg1_ok && MUSIC::mpi::is_root())
		LOGINFO("H.2b: fine slab unusable for 8-pt averaging (empty slab on some rank); coarse level %d falls back to direct sampling.", levelmax - 1);
#else
	bool avg1_ok = false; // !OLD_KERNEL_SAMPLING never averaged in the first place
#endif

	delete[] rkernel; // fine slab no longer needed

	// Coarse levels — rank-0-only after the distributed first-level average.
	if (MUSIC::mpi::is_root())
	{
		fftw_real *rkernel_prev = NULL; // previous coarse kernel (cascade source)
		int nx_prev = 0, ny_prev = 0, nz_prev = 0;
		for (int ilevel = levelmax - 1; ilevel >= levelmin; ilevel--)
		{
			LOGUSER("Computing coarse kernel (level %d) on rank 0 (slab path)...", ilevel);

			int nxc, nyc, nzc;
			real_t dxc, lxc, lyc, lzc;

			nxc = refh.size(ilevel, 0);
			nyc = refh.size(ilevel, 1);
			nzc = refh.size(ilevel, 2);

			if (ilevel != levelmin)
			{
				nxc *= 2;
				nyc *= 2;
				nzc *= 2;
			}

			dxc = boxlength / (1 << ilevel);
			lxc = dxc * nxc; lyc = dxc * nyc; lzc = dxc * nzc;

			fftw_real *rkernel_coarse = new fftw_real[(size_t)nxc * (size_t)nyc * 2 * ((size_t)nzc / 2 + 1)];
			real_t fac_c = lxc * lyc * lzc / pow(2.0 * M_PI, 3) / ((double)nxc * (double)nyc * (double)nzc);

			if (bperiodic)
			{
#pragma omp parallel for
				for (int i = 0; i <= nxc / 2; ++i)
					for (int j = 0; j <= nyc / 2; ++j)
						for (int k = 0; k <= nzc / 2; ++k)
						{
							int iix(i), iiy(j), iiz(k);
							real_t rr[3];
							if (iix > nxc / 2) iix -= nxc;
							if (iiy > nyc / 2) iiy -= nyc;
							if (iiz > nzc / 2) iiz -= nzc;

							size_t idx[8];
							idx[0] = ((size_t)i  *nyc + (size_t)j      ) * 2 * (nzc/2+1) + (size_t)k;
							idx[1] = ((size_t)(nxc-i)*nyc + (size_t)j  ) * 2 * (nzc/2+1) + (size_t)k;
							idx[2] = ((size_t)i  *nyc + (size_t)(nyc-j)) * 2 * (nzc/2+1) + (size_t)k;
							idx[3] = ((size_t)(nxc-i)*nyc + (size_t)(nyc-j)) * 2 * (nzc/2+1) + (size_t)k;
							idx[4] = ((size_t)i  *nyc + (size_t)j      ) * 2 * (nzc/2+1) + (size_t)(nzc-k);
							idx[5] = ((size_t)(nxc-i)*nyc + (size_t)j  ) * 2 * (nzc/2+1) + (size_t)(nzc-k);
							idx[6] = ((size_t)i  *nyc + (size_t)(nyc-j)) * 2 * (nzc/2+1) + (size_t)(nzc-k);
							idx[7] = ((size_t)(nxc-i)*nyc + (size_t)(nyc-j)) * 2 * (nzc/2+1) + (size_t)(nzc-k);

							if (i == 0 || i == nxc/2) idx[1]=idx[3]=idx[5]=idx[7]=(size_t)-1;
							if (j == 0 || j == nyc/2) idx[2]=idx[3]=idx[6]=idx[7]=(size_t)-1;
							if (k == 0 || k == nzc/2) idx[4]=idx[5]=idx[6]=idx[7]=(size_t)-1;

							double val = 0.0;
							for (int ii = -1; ii <= 1; ++ii)
								for (int jj = -1; jj <= 1; ++jj)
									for (int kk = -1; kk <= 1; ++kk)
									{
										rr[0] = ((double)iix) * dxc + ii * boxlength;
										rr[1] = ((double)iiy) * dxc + jj * boxlength;
										rr[2] = ((double)iiz) * dxc + kk * boxlength;
										if (rr[0] > -boxlength && rr[0] < boxlength &&
										    rr[1] > -boxlength && rr[1] < boxlength &&
										    rr[2] > -boxlength && rr[2] < boxlength)
										{
#ifdef OLD_KERNEL_SAMPLING
											val += tfr->compute_real(rr[0]*rr[0] + rr[1]*rr[1] + rr[2]*rr[2]);
#else
											val += eval_split_recurse(tfr, rr, dxc) / (dxc*dxc*dxc);
#endif
										}
									}
							val *= fac_c;
							for (int qq = 0; qq < 8; ++qq)
								if (idx[qq] != (size_t)-1)
									rkernel_coarse[idx[qq]] = val;
						}
			}
			else
			{
#pragma omp parallel for
				for (int i = 0; i < nxc; ++i)
					for (int j = 0; j < nyc; ++j)
						for (int k = 0; k < nzc; ++k)
						{
							int iix(i), iiy(j), iiz(k);
							real_t rr[3];
							if (iix > nxc/2) iix -= nxc;
							if (iiy > nyc/2) iiy -= nyc;
							if (iiz > nzc/2) iiz -= nzc;
							size_t idx = ((size_t)i*nyc + (size_t)j) * 2 * (nzc/2+1) + (size_t)k;
							rr[0] = (double)iix * dxc; rr[1] = (double)iiy * dxc; rr[2] = (double)iiz * dxc;
#ifdef OLD_KERNEL_SAMPLING
							rkernel_coarse[idx] = 0.0;
							real_t rr2 = rr[0]*rr[0] + rr[1]*rr[1] + rr[2]*rr[2];
							if (fabs(rr[0]) <= boxlength2 || fabs(rr[1]) <= boxlength2 || fabs(rr[2]) <= boxlength2)
								rkernel_coarse[idx] += (fftw_real)tfr->compute_real(rr2) * fac_c;
#else
							rkernel_coarse[idx] = 0.0;
							real_t val = eval_split_recurse(tfr, rr, dxc) / (dxc*dxc*dxc);
							if (fabs(rr[0]) <= boxlength2 || fabs(rr[1]) <= boxlength2 || fabs(rr[2]) <= boxlength2)
								rkernel_coarse[idx] += val * fac_c;
#endif
						}
			}

#ifdef OLD_KERNEL_SAMPLING
			// 8-pt averaging overwrite of the central footprint (H.2b).
			// First coarse level: merge the distributed average (touched cells
			// only, preserving direct-sampled gap cells). Lower levels cascade
			// from the previous coarse kernel exactly like precompute_kernel.
			if (ilevel == levelmax - 1)
			{
				if (avg1_ok)
				{
					const size_t csz = (size_t)nxc * (size_t)nyc * 2 * ((size_t)nzc / 2 + 1);
					for (size_t c = 0; c < csz; ++c)
						if (touch1[c]) rkernel_coarse[c] = avg1[c];
				}
			}
			else
			{
				average_fine_full_to_coarse_serial(rkernel_prev, nx_prev, ny_prev, nz_prev,
				                                   rkernel_coarse, nxc, nyc, nzc);
			}
#endif

			sprintf(cachefname, "temp_kernel_level%03d.tmp", ilevel);
			LOGUSER("Storing coarse kernel in temp file '%s'.", cachefname);
			FILE *fp = fopen(cachefname, "w+");
			unsigned q = (unsigned)nxc; fwrite(&q, sizeof(unsigned), 1, fp);
			q = (unsigned)nyc; fwrite(&q, sizeof(unsigned), 1, fp);
			q = (unsigned)(2 * (nzc/2 + 1)); fwrite(&q, sizeof(unsigned), 1, fp);
			for (int ix = 0; ix < nxc; ++ix)
			{
				size_t sz = (size_t)nyc * 2 * ((size_t)nzc/2 + 1);
				fwrite(reinterpret_cast<void *>(&rkernel_coarse[(size_t)ix * sz]), sizeof(fftw_real), sz, fp);
			}
			fclose(fp);

			// Cascade: keep this coarse kernel as the averaging source for the
			// next (coarser) level instead of freeing it immediately.
			if (rkernel_prev) delete[] rkernel_prev;
			rkernel_prev = rkernel_coarse;
			nx_prev = nxc; ny_prev = nyc; nz_prev = nzc;
		}
		if (rkernel_prev) delete[] rkernel_prev;
	}

	MPI_Barrier(MUSIC::mpi::world());
	delete tfr;
#else
	(void)ptf; (void)type; (void)refh;
	throw std::runtime_error("precompute_kernel_slab requires USE_MPI");
#endif
}

} // namespace convolution

/**************************************************************************************/
/**************************************************************************************/

namespace
{
convolution::kernel_creator_concrete<convolution::kernel_real_cached<double>> creator_d("tf_kernel_real_double");
convolution::kernel_creator_concrete<convolution::kernel_real_cached<float>> creator_f("tf_kernel_real_float");

convolution::kernel_creator_concrete<convolution::kernel_k<double>> creator_kd("tf_kernel_k_double");
convolution::kernel_creator_concrete<convolution::kernel_k<float>> creator_kf("tf_kernel_k_float");
} // namespace
