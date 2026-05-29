/*
 
 densities.cc - This file is part of MUSIC -
 a code to generate multi-scale initial conditions 
 for cosmological simulations 
 
 Copyright (C) 2010  Oliver Hahn
 
 */

#include <cstring>
#include <climits>
#include <vector>
#include <stdexcept>

#include "densities.hh"
#include "convolution_kernel.hh"
#include "mpi_helper.hh"
#include "mesh_distributed.hh"
#include "zoom_slab.hh"

//TODO: this should be a larger number by default, just to maintain consistency with old default
#define DEF_RAN_CUBE_SIZE 32

double blend_sharpness = 0.5;

double Blend_Function(double k, double kmax)
{
#if 0
    const static double pihalf = 0.5*M_PI;
    if( fabs(k)>kmax ) return 0.0;
    return cos(k/kmax * pihalf );
#else
	float kabs = fabs(k);
	double const eps = blend_sharpness;
	float kp = (1.0f - 2.0f * eps) * kmax;

	if (kabs >= kmax)
		return 0.;
	if (kabs > kp)
		return 1.0f / (expf((kp - kmax) / (k - kp) + (kp - kmax) / (k - kmax)) + 1.0f);
	return 1.0f;
#endif
}

template <typename m1, typename m2>
void fft_coarsen(m1 &v, m2 &V)
{
	size_t nxf = v.size(0), nyf = v.size(1), nzf = v.size(2), nzfp = nzf + 2;
	size_t nxF = V.size(0), nyF = V.size(1), nzF = V.size(2), nzFp = nzF + 2;

	fftw_real *rcoarse = new fftw_real[nxF * nyF * nzFp];
	fftw_complex *ccoarse = reinterpret_cast<fftw_complex *>(rcoarse);

	fftw_real *rfine = new fftw_real[nxf * nyf * nzfp];
	fftw_complex *cfine = reinterpret_cast<fftw_complex *>(rfine);

#ifdef FFTW3
#ifdef SINGLE_PRECISION
	fftwf_plan
		pf = fftwf_plan_dft_r2c_3d(nxf, nyf, nzf, rfine, cfine, FFTW_ESTIMATE),
		ipc = fftwf_plan_dft_c2r_3d(nxF, nyF, nzF, ccoarse, rcoarse, FFTW_ESTIMATE);
#else
	fftw_plan
		pf = fftw_plan_dft_r2c_3d(nxf, nyf, nzf, rfine, cfine, FFTW_ESTIMATE),
		ipc = fftw_plan_dft_c2r_3d(nxF, nyF, nzF, ccoarse, rcoarse, FFTW_ESTIMATE);
#endif

#else
	rfftwnd_plan
		pf = rfftw3d_create_plan(nxf, nyf, nzf, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE | FFTW_IN_PLACE),
		ipc = rfftw3d_create_plan(nxF, nyF, nzF, FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE | FFTW_IN_PLACE);
#endif

#pragma omp parallel for
	for (int i = 0; i < (int)nxf; i++)
		for (int j = 0; j < (int)nyf; j++)
			for (int k = 0; k < (int)nzf; k++)
			{
				size_t q = ((size_t)i * nyf + (size_t)j) * nzfp + (size_t)k;
				rfine[q] = v(i, j, k);
			}

#ifdef FFTW3
#ifdef SINGLE_PRECISION
	fftwf_execute(pf);
#else
	fftw_execute(pf);
#endif
#else
#ifndef SINGLETHREAD_FFTW
	rfftwnd_threads_one_real_to_complex(omp_get_max_threads(), pf, rfine, NULL);
#else
	rfftwnd_one_real_to_complex(pf, rfine, NULL);
#endif
#endif

	double fftnorm = 1.0 / ((double)nxF * (double)nyF * (double)nzF);

#pragma omp parallel for
	for (int i = 0; i < (int)nxF; i++)
		for (int j = 0; j < (int)nyF; j++)
			for (int k = 0; k < (int)nzF / 2 + 1; k++)
			{
				int ii(i), jj(j), kk(k);

				if (i > (int)nxF / 2)
					ii += (int)nxf / 2;
				if (j > (int)nyF / 2)
					jj += (int)nyf / 2;

				size_t qc, qf;

				double kx = (i <= (int)nxF / 2) ? (double)i : (double)(i - (int)nxF);
				double ky = (j <= (int)nyF / 2) ? (double)j : (double)(j - (int)nyF);
				double kz = (k <= (int)nzF / 2) ? (double)k : (double)(k - (int)nzF);

				qc = ((size_t)i * nyF + (size_t)j) * (nzF / 2 + 1) + (size_t)k;
				qf = ((size_t)ii * nyf + (size_t)jj) * (nzf / 2 + 1) + (size_t)kk;

				std::complex<double> val_fine(RE(cfine[qf]), IM(cfine[qf]));
				double phase = (kx / nxF + ky / nyF + kz / nzF) * 0.5 * M_PI;

				std::complex<double> val_phas(cos(phase), sin(phase));

				val_fine *= val_phas * fftnorm / 8.0; //sqrt(8.0);

				RE(ccoarse[qc]) = val_fine.real();
				IM(ccoarse[qc]) = val_fine.imag();
			}

	delete[] rfine;

#ifdef FFTW3
#ifdef SINGLE_PRECISION
	fftwf_execute(ipc);
#else
	fftw_execute(ipc);
#endif
#else
#ifndef SINGLETHREAD_FFTW
	rfftwnd_threads_one_complex_to_real(omp_get_max_threads(), ipc, ccoarse, NULL);
#else
	rfftwnd_one_complex_to_real(ipc, ccoarse, NULL);
#endif
#endif

#pragma omp parallel for
	for (int i = 0; i < (int)nxF; i++)
		for (int j = 0; j < (int)nyF; j++)
			for (int k = 0; k < (int)nzF; k++)
			{
				size_t q = ((size_t)i * nyF + (size_t)j) * nzFp + (size_t)k;
				V(i, j, k) = rcoarse[q];
			}

	delete[] rcoarse;

#ifdef FFTW3
#ifdef SINGLE_PRECISION
	fftwf_destroy_plan(pf);
	fftwf_destroy_plan(ipc);
#else
	fftw_destroy_plan(pf);
	fftw_destroy_plan(ipc);
#endif
#else
	rfftwnd_destroy_plan(pf);
	rfftwnd_destroy_plan(ipc);
#endif
}

//#define NO_COARSE_OVERLAP

template <typename m1, typename m2>
void fft_interpolate(m1 &V, m2 &v, bool from_basegrid = false)
{
	int oxf = v.offset(0), oyf = v.offset(1), ozf = v.offset(2);
	size_t nxf = v.size(0), nyf = v.size(1), nzf = v.size(2), nzfp = nzf + 2;
	size_t nxF = V.size(0), nyF = V.size(1), nzF = V.size(2);

	if (!from_basegrid)
	{
#ifdef NO_COARSE_OVERLAP
		oxf += nxF / 4;
		oyf += nyF / 4;
		ozf += nzF / 4;
#else
		oxf += nxF / 4 - nxf / 8;
		oyf += nyF / 4 - nyf / 8;
		ozf += nzF / 4 - nzf / 8;
	}
	else
	{
		oxf -= nxf / 8;
		oyf -= nyf / 8;
		ozf -= nzf / 8;
#endif
	}

	LOGUSER("FFT interpolate: offset=%d,%d,%d size=%d,%d,%d", oxf, oyf, ozf, nxf, nyf, nzf);

	// cut out piece of coarse grid that overlaps the fine:
	assert(nxf % 2 == 0 && nyf % 2 == 0 && nzf % 2 == 0);

	size_t nxc = nxf / 2, nyc = nyf / 2, nzc = nzf / 2, nzcp = nzf / 2 + 2;

	fftw_real *rcoarse = new fftw_real[nxc * nyc * nzcp];
	fftw_complex *ccoarse = reinterpret_cast<fftw_complex *>(rcoarse);

	fftw_real *rfine = new fftw_real[nxf * nyf * nzfp];
	fftw_complex *cfine = reinterpret_cast<fftw_complex *>(rfine);

	// copy coarse data to rcoarse[.]
	memset(rcoarse, 0, sizeof(fftw_real) * nxc * nyc * nzcp);

	// Periodic wrap for the coarse read: when the fine patch sits at the
	// corner of the coarse grid (e.g. multibox cubic-inflate places level
	// L+1 at offset 0 on the full domain), the -nxf/8 pre-roll makes
	// oxf+i negative on entry. The coarse grid is periodic (cosmological
	// IC, periodic_TF=yes), so wrap to the positive equivalent before
	// indexing — V::operator() takes size_t, so a raw negative value would
	// underflow into a huge unsigned and segfault.
	auto wrap = [](int x, size_t N) {
		int n = (int)N;
		x %= n;
		if (x < 0) x += n;
		return (size_t)x;
	};

#ifdef NO_COARSE_OVERLAP
#pragma omp parallel for
	for (int i = 0; i < (int)nxc / 2; ++i)
		for (int j = 0; j < (int)nyc / 2; ++j)
			for (int k = 0; k < (int)nzc / 2; ++k)
			{
				int ii(i + nxc / 4);
				int jj(j + nyc / 4);
				int kk(k + nzc / 4);
				size_t q = ((size_t)ii * nyc + (size_t)jj) * nzcp + (size_t)kk;
				rcoarse[q] = V(wrap(oxf + i, nxF), wrap(oyf + j, nyF), wrap(ozf + k, nzF));
			}
#else
#pragma omp parallel for
	for (int i = 0; i < (int)nxc; ++i)
		for (int j = 0; j < (int)nyc; ++j)
			for (int k = 0; k < (int)nzc; ++k)
			{
				int ii(i);
				int jj(j);
				int kk(k);
				size_t q = ((size_t)ii * nyc + (size_t)jj) * nzcp + (size_t)kk;
				rcoarse[q] = V(wrap(oxf + i, nxF), wrap(oyf + j, nyF), wrap(ozf + k, nzF));
			}
#endif

#pragma omp parallel for
	for (int i = 0; i < (int)nxf; ++i)
		for (int j = 0; j < (int)nyf; ++j)
			for (int k = 0; k < (int)nzf; ++k)
			{
				size_t q = ((size_t)i * nyf + (size_t)j) * nzfp + (size_t)k;
				rfine[q] = v(i, j, k);
			}

#ifdef FFTW3
#ifdef SINGLE_PRECISION
	fftwf_plan
		pc = fftwf_plan_dft_r2c_3d(nxc, nyc, nzc, rcoarse, ccoarse, FFTW_ESTIMATE),
		pf = fftwf_plan_dft_r2c_3d(nxf, nyf, nzf, rfine, cfine, FFTW_ESTIMATE),
		ipf = fftwf_plan_dft_c2r_3d(nxf, nyf, nzf, cfine, rfine, FFTW_ESTIMATE);
	fftwf_execute(pc);
	fftwf_execute(pf);
#else
	fftw_plan
		pc = fftw_plan_dft_r2c_3d(nxc, nyc, nzc, rcoarse, ccoarse, FFTW_ESTIMATE),
		pf = fftw_plan_dft_r2c_3d(nxf, nyf, nzf, rfine, cfine, FFTW_ESTIMATE),
		ipf = fftw_plan_dft_c2r_3d(nxf, nyf, nzf, cfine, rfine, FFTW_ESTIMATE);
	fftw_execute(pc);
	fftw_execute(pf);
#endif
#else
	rfftwnd_plan
		pc = rfftw3d_create_plan(nxc, nyc, nzc, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE | FFTW_IN_PLACE),
		pf = rfftw3d_create_plan(nxf, nyf, nzf, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE | FFTW_IN_PLACE),
		ipf = rfftw3d_create_plan(nxf, nyf, nzf, FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE | FFTW_IN_PLACE);

#ifndef SINGLETHREAD_FFTW
	rfftwnd_threads_one_real_to_complex(omp_get_max_threads(), pc, rcoarse, NULL);
	rfftwnd_threads_one_real_to_complex(omp_get_max_threads(), pf, rfine, NULL);
#else
	rfftwnd_one_real_to_complex(pc, rcoarse, NULL);
	rfftwnd_one_real_to_complex(pf, rfine, NULL);
#endif
#endif

	/*************************************************/
	//.. perform actual interpolation
	double fftnorm = 1.0 / ((double)nxf * (double)nyf * (double)nzf);
	double sqrt8 = 8.0; //sqrt(8.0);
	double phasefac = -0.5;

	// this enables filtered splicing of coarse and fine modes
#if 1
	for (int i = 0; i < (int)nxc; i++)
		for (int j = 0; j < (int)nyc; j++)
			for (int k = 0; k < (int)nzc / 2 + 1; k++)
			{
				int ii(i), jj(j), kk(k);

				if (i > (int)nxc / 2)
					ii += (int)nxf / 2;
				if (j > (int)nyc / 2)
					jj += (int)nyf / 2;
				if (k > (int)nzc / 2)
					kk += (int)nzf / 2;

				size_t qc, qf;
				qc = ((size_t)i * (size_t)nyc + (size_t)j) * (nzc / 2 + 1) + (size_t)k;
				qf = ((size_t)ii * (size_t)nyf + (size_t)jj) * (nzf / 2 + 1) + (size_t)kk;

				double kx = (i <= (int)nxc / 2) ? (double)i : (double)(i - (int)nxc);
				double ky = (j <= (int)nyc / 2) ? (double)j : (double)(j - (int)nyc);
				double kz = (k <= (int)nzc / 2) ? (double)k : (double)(k - (int)nzc);

				double phase = phasefac * (kx / nxc + ky / nyc + kz / nzc) * M_PI;

				std::complex<double> val_phas(cos(phase), sin(phase));

				std::complex<double> val(RE(ccoarse[qc]), IM(ccoarse[qc]));
				val *= sqrt8 * val_phas;

				double blend_coarse = Blend_Function(sqrt(kx * kx + ky * ky + kz * kz), nxc / 2);
				double blend_fine = 1.0 - blend_coarse;

				RE(cfine[qf]) = blend_fine * RE(cfine[qf]) + blend_coarse * val.real();
				IM(cfine[qf]) = blend_fine * IM(cfine[qf]) + blend_coarse * val.imag();
			}

#else

	// 0 0
#pragma omp parallel for
	for (int i = 0; i < (int)nxc / 2 + 1; i++)
		for (int j = 0; j < (int)nyc / 2 + 1; j++)
			for (int k = 0; k < (int)nzc / 2 + 1; k++)
			{
				int ii(i), jj(j), kk(k);
				size_t qc, qf;
				qc = ((size_t)i * (size_t)nyc + (size_t)j) * (nzc / 2 + 1) + (size_t)k;
				qf = ((size_t)ii * (size_t)nyf + (size_t)jj) * (nzf / 2 + 1) + (size_t)kk;

				double kx = (i <= (int)nxc / 2) ? (double)i : (double)(i - (int)nxc);
				double ky = (j <= (int)nyc / 2) ? (double)j : (double)(j - (int)nyc);
				double kz = (k <= (int)nzc / 2) ? (double)k : (double)(k - (int)nzc);

				double phase = phasefac * (kx / nxc + ky / nyc + kz / nzc) * M_PI;
				std::complex<double> val_phas(cos(phase), sin(phase));

				std::complex<double> val(RE(ccoarse[qc]), IM(ccoarse[qc]));
				val *= sqrt8 * val_phas;

				RE(cfine[qf]) = val.real();
				IM(cfine[qf]) = val.imag();
			}

// 1 0
#pragma omp parallel for
	for (int i = nxc / 2; i < (int)nxc; i++)
		for (int j = 0; j < (int)nyc / 2 + 1; j++)
			for (int k = 0; k < (int)nzc / 2 + 1; k++)
			{
				int ii(i + nxf / 2), jj(j), kk(k);
				size_t qc, qf;
				qc = ((size_t)i * (size_t)nyc + (size_t)j) * (nzc / 2 + 1) + (size_t)k;
				qf = ((size_t)ii * (size_t)nyf + (size_t)jj) * (nzf / 2 + 1) + (size_t)kk;

				double kx = (i <= (int)nxc / 2) ? (double)i : (double)(i - (int)nxc);
				double ky = (j <= (int)nyc / 2) ? (double)j : (double)(j - (int)nyc);
				double kz = (k <= (int)nzc / 2) ? (double)k : (double)(k - (int)nzc);

				double phase = phasefac * (kx / nxc + ky / nyc + kz / nzc) * M_PI;
				std::complex<double> val_phas(cos(phase), sin(phase));

				std::complex<double> val(RE(ccoarse[qc]), IM(ccoarse[qc]));
				val *= sqrt8 * val_phas;

				RE(cfine[qf]) = val.real();
				IM(cfine[qf]) = val.imag();
			}

// 0 1
#pragma omp parallel for
	for (int i = 0; i < (int)nxc / 2 + 1; i++)
		for (int j = nyc / 2; j < (int)nyc; j++)
			for (int k = 0; k < (int)nzc / 2 + 1; k++)
			{
				int ii(i), jj(j + nyf / 2), kk(k);
				size_t qc, qf;
				qc = ((size_t)i * (size_t)nyc + (size_t)j) * (nzc / 2 + 1) + (size_t)k;
				qf = ((size_t)ii * (size_t)nyf + (size_t)jj) * (nzf / 2 + 1) + (size_t)kk;

				double kx = (i <= (int)nxc / 2) ? (double)i : (double)(i - (int)nxc);
				double ky = (j <= (int)nyc / 2) ? (double)j : (double)(j - (int)nyc);
				double kz = (k <= (int)nzc / 2) ? (double)k : (double)(k - (int)nzc);

				double phase = phasefac * (kx / nxc + ky / nyc + kz / nzc) * M_PI;
				std::complex<double> val_phas(cos(phase), sin(phase));

				std::complex<double> val(RE(ccoarse[qc]), IM(ccoarse[qc]));
				val *= sqrt8 * val_phas;

				RE(cfine[qf]) = val.real();
				IM(cfine[qf]) = val.imag();
			}

// 1 1
#pragma omp parallel for
	for (int i = nxc / 2; i < (int)nxc; i++)
		for (int j = nyc / 2; j < (int)nyc; j++)
			for (int k = 0; k < (int)nzc / 2 + 1; k++)
			{
				int ii(i + nxf / 2), jj(j + nyf / 2), kk(k);
				size_t qc, qf;
				qc = ((size_t)i * (size_t)nyc + (size_t)j) * (nzc / 2 + 1) + (size_t)k;
				qf = ((size_t)ii * (size_t)nyf + (size_t)jj) * (nzf / 2 + 1) + (size_t)kk;

				double kx = (i <= (int)nxc / 2) ? (double)i : (double)(i - (int)nxc);
				double ky = (j <= (int)nyc / 2) ? (double)j : (double)(j - (int)nyc);
				double kz = (k <= (int)nzc / 2) ? (double)k : (double)(k - (int)nzc);

				double phase = phasefac * (kx / nxc + ky / nyc + kz / nzc) * M_PI;
				std::complex<double> val_phas(cos(phase), sin(phase));

				std::complex<double> val(RE(ccoarse[qc]), IM(ccoarse[qc]));
				val *= sqrt8 * val_phas;

				RE(cfine[qf]) = val.real();
				IM(cfine[qf]) = val.imag();
			}
#endif

	delete[] rcoarse;

	/*************************************************/

#ifdef FFTW3
#ifdef SINGLE_PRECISION
	fftwf_execute(ipf);
	fftwf_destroy_plan(pf);
	fftwf_destroy_plan(pc);
	fftwf_destroy_plan(ipf);
#else
	fftw_execute(ipf);
	fftw_destroy_plan(pf);
	fftw_destroy_plan(pc);
	fftw_destroy_plan(ipf);
#endif
#else
#ifndef SINGLETHREAD_FFTW
	rfftwnd_threads_one_complex_to_real(omp_get_max_threads(), ipf, cfine, NULL);
#else
	rfftwnd_one_complex_to_real(ipf, cfine, NULL);
#endif
	fftwnd_destroy_plan(pf);
	fftwnd_destroy_plan(pc);
	fftwnd_destroy_plan(ipf);
#endif

// copy back and normalize
#pragma omp parallel for
	for (int i = 0; i < (int)nxf; ++i)
		for (int j = 0; j < (int)nyf; ++j)
			for (int k = 0; k < (int)nzf; ++k)
			{
				size_t q = ((size_t)i * nyf + (size_t)j) * nzfp + (size_t)k;
				v(i, j, k) = rfine[q] * fftnorm;
			}

	delete[] rfine;
}

/*******************************************************************************************/
//
// Phase D.7.2: SPMD/MPI-distributed variant of fft_interpolate.
//
// Coarse FFT is computed redundantly on every rank (the coarse grid is 1/8
// the volume of fine, so replicate-and-compute is cheaper than building a
// separate distribution). Fine forward / inverse FFTs use FFTW3-MPI slab
// decomposition along x. The coarse buffer is packed and broadcast from
// rank 0; the fine buffer is packed on rank 0 and scattered, then the
// gathered slabs are written back into v with FFT normalization.
//
// V (coarse) and v (fine) are only meaningful on rank 0. Workers pass
// nullptr for both and never deref them — they only own the slab buffer.
// Falls back to the serial fft_interpolate when USE_MPI is undefined or
// MPI size == 1.
//
/*******************************************************************************************/

#ifdef USE_MPI
template <typename m1, typename m2>
void fft_interpolate_dist(m1 *V, m2 *v, bool from_basegrid = false)
{
	if (MUSIC::mpi::size() == 1) {
		fft_interpolate(*V, *v, from_basegrid);
		return;
	}
	const int rk = MUSIC::mpi::rank();
	const int sz = MUSIC::mpi::size();
	const bool is_root = (rk == 0);

	// -- Broadcast geometry (workers do not own V or v) --
	int dims[9] = {0};
	if (is_root) {
		dims[0] = (int)v->size(0);
		dims[1] = (int)v->size(1);
		dims[2] = (int)v->size(2);
		dims[3] = (int)V->size(0);
		dims[4] = (int)V->size(1);
		dims[5] = (int)V->size(2);
		int oxf = v->offset(0), oyf = v->offset(1), ozf = v->offset(2);
		if (!from_basegrid) {
#ifdef NO_COARSE_OVERLAP
			oxf += dims[3] / 4;
			oyf += dims[4] / 4;
			ozf += dims[5] / 4;
#else
			oxf += dims[3] / 4 - dims[0] / 8;
			oyf += dims[4] / 4 - dims[1] / 8;
			ozf += dims[5] / 4 - dims[2] / 8;
		} else {
			oxf -= dims[0] / 8;
			oyf -= dims[1] / 8;
			ozf -= dims[2] / 8;
#endif
		}
		dims[6] = oxf; dims[7] = oyf; dims[8] = ozf;
	}
	MPI_Bcast(dims, 9, MPI_INT, 0, MUSIC::mpi::world());

	const size_t nxf = (size_t)dims[0];
	const size_t nyf = (size_t)dims[1];
	const size_t nzf = (size_t)dims[2];
	const size_t nxF = (size_t)dims[3];
	const size_t nyF = (size_t)dims[4];
	const size_t nzF = (size_t)dims[5];
	const int    oxf = dims[6], oyf = dims[7], ozf = dims[8];
	const size_t nzfp = nzf + 2;

	assert(nxf % 2 == 0 && nyf % 2 == 0 && nzf % 2 == 0);
	const size_t nxc = nxf / 2, nyc = nyf / 2, nzc = nzf / 2, nzcp = nzc + 2;

	LOGUSER("FFT interpolate (SPMD): offset=%d,%d,%d size=%zu,%zu,%zu",
	        oxf, oyf, ozf, nxf, nyf, nzf);

	MPI_Datatype mpi_real = (sizeof(real_t) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;

	// -- Coarse path (replicated): pack on rank 0, Bcast, local r2c on all ranks --
	fftw_real *rcoarse = new fftw_real[nxc * nyc * nzcp];
	fftw_complex *ccoarse = reinterpret_cast<fftw_complex *>(rcoarse);
	std::memset(rcoarse, 0, sizeof(fftw_real) * nxc * nyc * nzcp);

	if (is_root) {
		auto wrap = [](int x, size_t N) {
			int n = (int)N;
			x %= n;
			if (x < 0) x += n;
			return (size_t)x;
		};
#ifdef NO_COARSE_OVERLAP
#pragma omp parallel for
		for (int i = 0; i < (int)nxc / 2; ++i)
			for (int j = 0; j < (int)nyc / 2; ++j)
				for (int k = 0; k < (int)nzc / 2; ++k) {
					int ii(i + (int)nxc / 4);
					int jj(j + (int)nyc / 4);
					int kk(k + (int)nzc / 4);
					size_t q = ((size_t)ii * nyc + (size_t)jj) * nzcp + (size_t)kk;
					rcoarse[q] = (*V)(wrap(oxf + i, nxF), wrap(oyf + j, nyF), wrap(ozf + k, nzF));
				}
#else
#pragma omp parallel for
		for (int i = 0; i < (int)nxc; ++i)
			for (int j = 0; j < (int)nyc; ++j)
				for (int k = 0; k < (int)nzc; ++k) {
					size_t q = ((size_t)i * nyc + (size_t)j) * nzcp + (size_t)k;
					rcoarse[q] = (*V)(wrap(oxf + i, nxF), wrap(oyf + j, nyF), wrap(ozf + k, nzF));
				}
#endif
	}

	{
		size_t total = nxc * nyc * nzcp;
		size_t plane = nyc * nzcp;
		if (total <= (size_t)INT_MAX) {
			MPI_Bcast(rcoarse, (int)total, mpi_real, 0, MUSIC::mpi::world());
		} else {
			if (plane > (size_t)INT_MAX)
				throw std::runtime_error("fft_interpolate_dist: coarse plane exceeds INT_MAX");
			MPI_Datatype dt;
			MPI_Type_contiguous((int)plane, mpi_real, &dt);
			MPI_Type_commit(&dt);
			MPI_Bcast(rcoarse, (int)nxc, dt, 0, MUSIC::mpi::world());
			MPI_Type_free(&dt);
		}
	}

#ifdef SINGLE_PRECISION
	fftwf_plan pc = fftwf_plan_dft_r2c_3d((int)nxc, (int)nyc, (int)nzc, rcoarse, ccoarse, FFTW_ESTIMATE);
	fftwf_execute(pc);
	fftwf_destroy_plan(pc);
#else
	fftw_plan pc = fftw_plan_dft_r2c_3d((int)nxc, (int)nyc, (int)nzc, rcoarse, ccoarse, FFTW_ESTIMATE);
	fftw_execute(pc);
	fftw_destroy_plan(pc);
#endif

	// -- Fine path (distributed): allocate slab, scatter from rank 0 --
	Meshvar<real_t> *slab = MUSIC::dist::make_slab_meshvar<real_t>(
	    nxf, nyf, nzf, /*fftw_inplace_pad=*/true);
	const size_t local_nx      = slab->local_nx();
	const size_t local_x_start = slab->local_x_start();
	const size_t plane_fine    = nyf * nzfp;
	if (plane_fine > (size_t)INT_MAX) {
		throw std::runtime_error("fft_interpolate_dist: fine y-z plane exceeds INT_MAX");
	}
	MPI_Datatype dtype_plane_fine;
	MPI_Type_contiguous((int)plane_fine, mpi_real, &dtype_plane_fine);
	MPI_Type_commit(&dtype_plane_fine);

	std::vector<int> counts(sz), displs(sz);
	int my_count = (int)local_nx;
	MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MUSIC::mpi::world());
	displs[0] = 0;
	for (int i = 1; i < sz; ++i) displs[i] = displs[i - 1] + counts[i - 1];

	real_t *fine_root = NULL;
	if (is_root) {
		fine_root = new real_t[nxf * nyf * nzfp];
#pragma omp parallel for
		for (int i = 0; i < (int)nxf; ++i)
			for (int j = 0; j < (int)nyf; ++j)
				for (int k = 0; k < (int)nzf; ++k) {
					size_t q = ((size_t)i * nyf + (size_t)j) * nzfp + (size_t)k;
					fine_root[q] = (*v)(i, j, k);
				}
	}

	MPI_Scatterv(is_root ? fine_root : (real_t *)NULL,
	             counts.data(), displs.data(), dtype_plane_fine,
	             slab->m_pdata, my_count, dtype_plane_fine,
	             0, MUSIC::mpi::world());

	fftw_complex *cslab = reinterpret_cast<fftw_complex *>(slab->m_pdata);

#ifdef SINGLE_PRECISION
	fftwf_plan p_fwd = fftwf_mpi_plan_dft_r2c_3d(
	    (ptrdiff_t)nxf, (ptrdiff_t)nyf, (ptrdiff_t)nzf,
	    slab->m_pdata, cslab, MUSIC::mpi::world(), FFTW_ESTIMATE);
	fftwf_plan p_inv = fftwf_mpi_plan_dft_c2r_3d(
	    (ptrdiff_t)nxf, (ptrdiff_t)nyf, (ptrdiff_t)nzf,
	    cslab, slab->m_pdata, MUSIC::mpi::world(), FFTW_ESTIMATE);
	fftwf_execute(p_fwd);
#else
	fftw_plan p_fwd = fftw_mpi_plan_dft_r2c_3d(
	    (ptrdiff_t)nxf, (ptrdiff_t)nyf, (ptrdiff_t)nzf,
	    slab->m_pdata, cslab, MUSIC::mpi::world(), FFTW_ESTIMATE);
	fftw_plan p_inv = fftw_mpi_plan_dft_c2r_3d(
	    (ptrdiff_t)nxf, (ptrdiff_t)nyf, (ptrdiff_t)nzf,
	    cslab, slab->m_pdata, MUSIC::mpi::world(), FFTW_ESTIMATE);
	fftw_execute(p_fwd);
#endif

	// -- K-space blend on owned x-slab --
	const double fftnorm = 1.0 / ((double)nxf * (double)nyf * (double)nzf);
	const double sqrt8 = 8.0;
	const double phasefac = -0.5;

	for (int i = 0; i < (int)nxc; ++i) {
		int ii = i;
		if (i > (int)nxc / 2) ii += (int)nxf / 2;
		if ((size_t)ii < local_x_start || (size_t)ii >= local_x_start + local_nx) continue;
		const size_t ii_local = (size_t)ii - local_x_start;

#pragma omp parallel for
		for (int j = 0; j < (int)nyc; ++j) {
			for (int k = 0; k < (int)nzc / 2 + 1; ++k) {
				int jj = j, kk = k;
				if (j > (int)nyc / 2) jj += (int)nyf / 2;
				if (k > (int)nzc / 2) kk += (int)nzf / 2;

				size_t qc = ((size_t)i * nyc + (size_t)j) * (nzc / 2 + 1) + (size_t)k;
				size_t qf = (ii_local * nyf + (size_t)jj) * (nzf / 2 + 1) + (size_t)kk;

				double kx = (i <= (int)nxc / 2) ? (double)i : (double)(i - (int)nxc);
				double ky = (j <= (int)nyc / 2) ? (double)j : (double)(j - (int)nyc);
				double kz = (k <= (int)nzc / 2) ? (double)k : (double)(k - (int)nzc);

				double phase = phasefac * (kx / (double)nxc + ky / (double)nyc + kz / (double)nzc) * M_PI;
				std::complex<double> val_phas(cos(phase), sin(phase));

				std::complex<double> val(RE(ccoarse[qc]), IM(ccoarse[qc]));
				val *= sqrt8 * val_phas;

				double blend_coarse = Blend_Function(sqrt(kx * kx + ky * ky + kz * kz), nxc / 2);
				double blend_fine = 1.0 - blend_coarse;

				RE(cslab[qf]) = blend_fine * RE(cslab[qf]) + blend_coarse * val.real();
				IM(cslab[qf]) = blend_fine * IM(cslab[qf]) + blend_coarse * val.imag();
			}
		}
	}

	delete[] rcoarse;

#ifdef SINGLE_PRECISION
	fftwf_execute(p_inv);
	fftwf_destroy_plan(p_fwd);
	fftwf_destroy_plan(p_inv);
#else
	fftw_execute(p_inv);
	fftw_destroy_plan(p_fwd);
	fftw_destroy_plan(p_inv);
#endif

	MPI_Gatherv(slab->m_pdata, my_count, dtype_plane_fine,
	            is_root ? fine_root : (real_t *)NULL,
	            counts.data(), displs.data(), dtype_plane_fine,
	            0, MUSIC::mpi::world());

	MPI_Type_free(&dtype_plane_fine);
	delete slab;

	if (is_root) {
#pragma omp parallel for
		for (int i = 0; i < (int)nxf; ++i)
			for (int j = 0; j < (int)nyf; ++j)
				for (int k = 0; k < (int)nzf; ++k) {
					size_t q = ((size_t)i * nyf + (size_t)j) * nzfp + (size_t)k;
					(*v)(i, j, k) = fine_root[q] * fftnorm;
				}
		delete[] fine_root;
	}
}
#else  // !USE_MPI
template <typename m1, typename m2>
void fft_interpolate_dist(m1 *V, m2 *v, bool from_basegrid = false)
{
	fft_interpolate(*V, *v, from_basegrid);
}
#endif // USE_MPI

/*******************************************************************************************/
//
// Phase H.3a: SPMD/MPI-distributed variant of fft_coarsen.
//
// Coarsen direction is inverse of fft_interpolate: fine (large, 60GB at pillars
// level 10) → coarse (1/8 size). Fine FFT is slab-distributed via FFTW3-MPI;
// coarse k-space buffer is allocated on every rank (1/8 of fine size, ~7.6GB
// at pillars). Each rank fills its disjoint contribution (coarse k-cells whose
// fine source ii is in this rank's x-slab) into its local ccoarse — other
// cells stay zero. MPI_Reduce(SUM) accumulates onto rank 0. Rank 0 runs serial
// c2r on the small (1/8) coarse buffer and writes back to V.
//
// v (fine) and V (coarse) are only meaningful on rank 0. Workers pass nullptr
// and never deref. Falls back to serial fft_coarsen when USE_MPI is undefined
// or MPI size == 1.
//
// Memory at pillars (level 10→9, nxf=1968, np=4):
//   - Now (serial): rank 0 = v(60GB) + rfine(60GB) + rcoarse(7.6GB) = 128GB
//   - H.3a:        rank 0 = v(60GB) + fine_root pack(60GB) + slab(15GB)
//                          + ccoarse(7.6GB) + rcoarse(7.6GB)
//                  workers = slab(15GB) + ccoarse temp(7.6GB → freed)
//   The 'fine_root pack' is a transient copy used for Scatterv (matches
//   fft_interpolate_dist's approach). H.3b will fuse pack with scatter.
//
/*******************************************************************************************/

#ifdef USE_MPI
template <typename m1, typename m2>
void fft_coarsen_dist(m1 *v, m2 *V)
{
	if (MUSIC::mpi::size() == 1) {
		fft_coarsen(*v, *V);
		return;
	}
	const int rk = MUSIC::mpi::rank();
	const int sz = MUSIC::mpi::size();
	const bool is_root = (rk == 0);

	// -- Broadcast geometry (workers do not own v or V) --
	int dims[6] = {0};
	if (is_root) {
		dims[0] = (int)v->size(0);
		dims[1] = (int)v->size(1);
		dims[2] = (int)v->size(2);
		dims[3] = (int)V->size(0);
		dims[4] = (int)V->size(1);
		dims[5] = (int)V->size(2);
	}
	MPI_Bcast(dims, 6, MPI_INT, 0, MUSIC::mpi::world());

	const size_t nxf = (size_t)dims[0];
	const size_t nyf = (size_t)dims[1];
	const size_t nzf = (size_t)dims[2];
	const size_t nxF = (size_t)dims[3];
	const size_t nyF = (size_t)dims[4];
	const size_t nzF = (size_t)dims[5];
	const size_t nzfp = nzf + 2;
	const size_t nzFp = nzF + 2;

	LOGUSER("FFT coarsen (SPMD): fine=%zu,%zu,%zu coarse=%zu,%zu,%zu",
	        nxf, nyf, nzf, nxF, nyF, nzF);

	MPI_Datatype mpi_real = (sizeof(real_t) == sizeof(float)) ? MPI_FLOAT : MPI_DOUBLE;

	// -- Fine path (distributed): allocate slab, pack on rank 0, scatter --
	Meshvar<real_t> *slab = MUSIC::dist::make_slab_meshvar<real_t>(
	    nxf, nyf, nzf, /*fftw_inplace_pad=*/true);
	const size_t local_nx      = slab->local_nx();
	const size_t local_x_start = slab->local_x_start();
	const size_t plane_fine    = nyf * nzfp;
	if (plane_fine > (size_t)INT_MAX) {
		throw std::runtime_error("fft_coarsen_dist: fine y-z plane exceeds INT_MAX");
	}
	MPI_Datatype dtype_plane_fine;
	MPI_Type_contiguous((int)plane_fine, mpi_real, &dtype_plane_fine);
	MPI_Type_commit(&dtype_plane_fine);

	std::vector<int> counts(sz), displs(sz);
	int my_count = (int)local_nx;
	MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MUSIC::mpi::world());
	displs[0] = 0;
	for (int i = 1; i < sz; ++i) displs[i] = displs[i - 1] + counts[i - 1];

	real_t *fine_root = NULL;
	if (is_root) {
		fine_root = new real_t[nxf * nyf * nzfp];
#pragma omp parallel for
		for (int i = 0; i < (int)nxf; ++i)
			for (int j = 0; j < (int)nyf; ++j)
				for (int k = 0; k < (int)nzf; ++k) {
					size_t q = ((size_t)i * nyf + (size_t)j) * nzfp + (size_t)k;
					fine_root[q] = (*v)(i, j, k);
				}
	}

	MPI_Scatterv(is_root ? fine_root : (real_t *)NULL,
	             counts.data(), displs.data(), dtype_plane_fine,
	             slab->m_pdata, my_count, dtype_plane_fine,
	             0, MUSIC::mpi::world());

	if (is_root) {
		delete[] fine_root;
		fine_root = NULL;
	}

	fftw_complex *cslab = reinterpret_cast<fftw_complex *>(slab->m_pdata);

#ifdef SINGLE_PRECISION
	fftwf_plan p_fwd = fftwf_mpi_plan_dft_r2c_3d(
	    (ptrdiff_t)nxf, (ptrdiff_t)nyf, (ptrdiff_t)nzf,
	    slab->m_pdata, cslab, MUSIC::mpi::world(), FFTW_ESTIMATE);
	fftwf_execute(p_fwd);
	fftwf_destroy_plan(p_fwd);
#else
	fftw_plan p_fwd = fftw_mpi_plan_dft_r2c_3d(
	    (ptrdiff_t)nxf, (ptrdiff_t)nyf, (ptrdiff_t)nzf,
	    slab->m_pdata, cslab, MUSIC::mpi::world(), FFTW_ESTIMATE);
	fftw_execute(p_fwd);
	fftw_destroy_plan(p_fwd);
#endif

	// -- Coarse path (replicated, sparse): each rank fills its k-cells --
	// Coarse buffer: nxF * nyF * nzFp reals (padded c2r layout)
	fftw_real *rcoarse = new fftw_real[nxF * nyF * nzFp];
	fftw_complex *ccoarse = reinterpret_cast<fftw_complex *>(rcoarse);
	std::memset(rcoarse, 0, sizeof(fftw_real) * nxF * nyF * nzFp);

	const double fftnorm = 1.0 / ((double)nxF * (double)nyF * (double)nzF);

	// Walk coarse k-space. Each (i,j,k) maps to ONE fine k-cell (ii,jj,k);
	// the rank owning ii in its x-slab writes the contribution. Other ranks'
	// ccoarse cells stay zero → MPI_Reduce(SUM) gives correct result.
	for (int i = 0; i < (int)nxF; ++i) {
		int ii = i;
		if (i > (int)nxF / 2) ii += (int)nxf / 2;
		if ((size_t)ii < local_x_start || (size_t)ii >= local_x_start + local_nx) continue;
		const size_t ii_local = (size_t)ii - local_x_start;

#pragma omp parallel for
		for (int j = 0; j < (int)nyF; ++j) {
			for (int k = 0; k < (int)nzF / 2 + 1; ++k) {
				int jj = j, kk = k;
				if (j > (int)nyF / 2) jj += (int)nyf / 2;
				// k stays the same (z is contiguous in FFTW r2c packing)

				size_t qc = ((size_t)i * nyF + (size_t)j) * (nzF / 2 + 1) + (size_t)k;
				size_t qf = (ii_local * nyf + (size_t)jj) * (nzf / 2 + 1) + (size_t)kk;

				double kx = (i <= (int)nxF / 2) ? (double)i : (double)(i - (int)nxF);
				double ky = (j <= (int)nyF / 2) ? (double)j : (double)(j - (int)nyF);
				double kz = (k <= (int)nzF / 2) ? (double)k : (double)(k - (int)nzF);

				std::complex<double> val_fine(RE(cslab[qf]), IM(cslab[qf]));
				double phase = (kx / (double)nxF + ky / (double)nyF + kz / (double)nzF) * 0.5 * M_PI;
				std::complex<double> val_phas(cos(phase), sin(phase));

				val_fine *= val_phas * fftnorm / 8.0;

				RE(ccoarse[qc]) = val_fine.real();
				IM(ccoarse[qc]) = val_fine.imag();
			}
		}
	}

	// Done with fine slab; free before allocating reduce buffer on rank 0
	delete slab;
	MPI_Type_free(&dtype_plane_fine);

	// MPI_Reduce(SUM) on the full ccoarse buffer onto rank 0. Workers'
	// contributions are sparse (most cells zero); rank 0 ends up with the
	// full coarse k-space buffer.
	{
		size_t total = nxF * nyF * nzFp;
		size_t plane_c = nyF * nzFp;
		MPI_Datatype dt_c = mpi_real;
		MPI_Datatype dt_c_tmp = MPI_DATATYPE_NULL;
		int reduce_count = 0;
		if (total <= (size_t)INT_MAX) {
			reduce_count = (int)total;
		} else {
			if (plane_c > (size_t)INT_MAX)
				throw std::runtime_error("fft_coarsen_dist: coarse plane exceeds INT_MAX");
			MPI_Type_contiguous((int)plane_c, mpi_real, &dt_c_tmp);
			MPI_Type_commit(&dt_c_tmp);
			dt_c = dt_c_tmp;
			reduce_count = (int)nxF;
		}
		if (is_root) {
			MPI_Reduce(MPI_IN_PLACE, rcoarse, reduce_count, dt_c,
			           MPI_SUM, 0, MUSIC::mpi::world());
		} else {
			MPI_Reduce(rcoarse, NULL, reduce_count, dt_c,
			           MPI_SUM, 0, MUSIC::mpi::world());
		}
		if (dt_c_tmp != MPI_DATATYPE_NULL) MPI_Type_free(&dt_c_tmp);
	}

	if (!is_root) {
		// Workers free their (zero) ccoarse and exit
		delete[] rcoarse;
		return;
	}

	// Rank 0: serial c2r on full coarse buffer, then unpack to V
#ifdef SINGLE_PRECISION
	fftwf_plan ipc = fftwf_plan_dft_c2r_3d((int)nxF, (int)nyF, (int)nzF,
	                                       ccoarse, rcoarse, FFTW_ESTIMATE);
	fftwf_execute(ipc);
	fftwf_destroy_plan(ipc);
#else
	fftw_plan ipc = fftw_plan_dft_c2r_3d((int)nxF, (int)nyF, (int)nzF,
	                                     ccoarse, rcoarse, FFTW_ESTIMATE);
	fftw_execute(ipc);
	fftw_destroy_plan(ipc);
#endif

#pragma omp parallel for
	for (int i = 0; i < (int)nxF; ++i)
		for (int j = 0; j < (int)nyF; ++j)
			for (int k = 0; k < (int)nzF; ++k) {
				size_t q = ((size_t)i * nyF + (size_t)j) * nzFp + (size_t)k;
				(*V)(i, j, k) = rcoarse[q];
			}

	delete[] rcoarse;
}
#else  // !USE_MPI
template <typename m1, typename m2>
void fft_coarsen_dist(m1 *v, m2 *V)
{
	fft_coarsen(*v, *V);
}
#endif // USE_MPI

/*******************************************************************************************/
/*******************************************************************************************/
/*******************************************************************************************/

//------------------------------------------------------------------------------
// H.3a smoke test: deterministic separable sin-product on a fine Meshvar,
// run both serial fft_coarsen and SPMD fft_coarsen_dist, compare on rank 0.
// Workers pass NULL for v/V to fft_coarsen_dist. Returns 0 on PASS.
//------------------------------------------------------------------------------
int run_h3a_coarsen_smoke(size_t Nfine)
{
	if (Nfine == 0 || (Nfine & 1u)) {
		if (MUSIC::mpi::is_root())
			std::cerr << "[h3a-coarsen-smoke] Nfine must be > 0 and even\n";
		return 1;
	}
	const size_t Ncoarse = Nfine / 2;
	const bool is_root = MUSIC::mpi::is_root();
	const double TWO_PI = 2.0 * M_PI;

	Meshvar<real_t> *v_serial = NULL, *V_serial = NULL;
	Meshvar<real_t> *v_dist   = NULL, *V_dist   = NULL;

	if (is_root) {
		v_serial = new Meshvar<real_t>((unsigned)Nfine, (unsigned)Nfine, (unsigned)Nfine,
		                                0, 0, 0);
		V_serial = new Meshvar<real_t>((unsigned)Ncoarse, (unsigned)Ncoarse, (unsigned)Ncoarse,
		                                0, 0, 0);
		v_dist   = new Meshvar<real_t>((unsigned)Nfine, (unsigned)Nfine, (unsigned)Nfine,
		                                0, 0, 0);
		V_dist   = new Meshvar<real_t>((unsigned)Ncoarse, (unsigned)Ncoarse, (unsigned)Ncoarse,
		                                0, 0, 0);
#pragma omp parallel for
		for (int i = 0; i < (int)Nfine; ++i) {
			double sx = std::sin(TWO_PI * (double)i / (double)Nfine)
			          + 0.5 * std::cos(2.0 * TWO_PI * (double)i / (double)Nfine);
			for (size_t j = 0; j < Nfine; ++j) {
				double sy = std::sin(TWO_PI * (double)j / (double)Nfine)
				          + 0.3 * std::cos(3.0 * TWO_PI * (double)j / (double)Nfine);
				for (size_t k = 0; k < Nfine; ++k) {
					double sz = std::sin(TWO_PI * (double)k / (double)Nfine)
					          + 0.2 * std::cos(4.0 * TWO_PI * (double)k / (double)Nfine);
					real_t val = (real_t)(sx * sy * sz);
					(*v_serial)(i, j, k) = val;
					(*v_dist)(i, j, k) = val;
				}
			}
		}
		// Zero coarse outputs so any unwritten cell is detectable
#pragma omp parallel for
		for (int i = 0; i < (int)Ncoarse; ++i)
			for (size_t j = 0; j < Ncoarse; ++j)
				for (size_t k = 0; k < Ncoarse; ++k) {
					(*V_serial)(i, j, k) = (real_t)0;
					(*V_dist)(i, j, k) = (real_t)0;
				}
	}

#ifdef USE_MPI
	if (is_root)
		fft_coarsen(*v_serial, *V_serial);
	fft_coarsen_dist(v_dist, V_dist);
#else
	fft_coarsen(*v_serial, *V_serial);
	fft_coarsen(*v_dist, *V_dist);
#endif

	int rc = 0;
	if (is_root) {
		double max_abs_diff = 0.0, max_abs_ref = 0.0;
		double sum_serial = 0.0, sum_dist = 0.0;
		for (size_t i = 0; i < Ncoarse; ++i)
			for (size_t j = 0; j < Ncoarse; ++j)
				for (size_t k = 0; k < Ncoarse; ++k) {
					double a = (double)(*V_serial)(i, j, k);
					double b = (double)(*V_dist)(i, j, k);
					sum_serial += a;
					sum_dist   += b;
					double d = std::fabs(a - b);
					if (d > max_abs_diff) max_abs_diff = d;
					if (std::fabs(a) > max_abs_ref) max_abs_ref = std::fabs(a);
				}
		double rel = max_abs_diff / std::max(max_abs_ref, 1e-300);
		// Accept ULP-level differences from FFTW plan divergence (serial vs MPI)
		const double tol = (sizeof(real_t) == sizeof(float)) ? 1e-4 : 1e-10;
		bool pass = (rel < tol);
		std::cerr << "[h3a-coarsen-smoke] Nfine=" << Nfine
		          << " Ncoarse=" << Ncoarse
		          << " ranks=" << MUSIC::mpi::size()
		          << " sum_serial=" << sum_serial
		          << " sum_dist=" << sum_dist
		          << " max|serial|=" << max_abs_ref
		          << " max|dist-serial|=" << max_abs_diff
		          << " rel=" << rel
		          << " tol=" << tol
		          << " " << (pass ? "PASS" : "FAIL")
		          << "\n";
		rc = pass ? 0 : 2;
	}

	delete v_serial; delete V_serial;
	delete v_dist;   delete V_dist;

#ifdef USE_MPI
	MPI_Bcast(&rc, 1, MPI_INT, 0, MUSIC::mpi::world());
#endif
	return rc;
}

//------------------------------------------------------------------------------
// H.4 smoke: exercises GridHierarchy::create_base_hierarchy_slab(lmax) and
// round-trips a deterministic global-index pattern through
// convert_level_slab_to_full(). Verifies that:
//   (a) the topmost level is a per-rank slab MeshvarBnd with the expected
//       FFTW3-MPI local x-range,
//   (b) each rank can write into its slab via global indices,
//   (c) gathering the slab back produces a rank-0 full MeshvarBnd that is
//       bit-identical to the serial reference pattern.
// Returns 0 on PASS.
//------------------------------------------------------------------------------
int run_h4_base_slab_smoke(size_t lmax_in)
{
	if (lmax_in == 0 || lmax_in > 12) {
		if (MUSIC::mpi::is_root())
			std::cerr << "[h4-base-slab-smoke] lmax must be in [1,12]\n";
		return 1;
	}
	const unsigned lmax = (unsigned)lmax_in;
	const size_t   ng   = (size_t)1 << lmax;
	const bool is_root = MUSIC::mpi::is_root();
	const int  nproc   = MUSIC::mpi::size();

	// Deterministic separable pattern (same one as h3a; gives a wide value
	// range so any indexing or strided-copy bug shows up).
	const double TWO_PI = 2.0 * M_PI;
	auto pattern = [&](size_t i, size_t j, size_t k) -> real_t {
		double sx = std::sin(TWO_PI * (double)i / (double)ng)
		          + 0.5 * std::cos(2.0 * TWO_PI * (double)i / (double)ng);
		double sy = std::sin(TWO_PI * (double)j / (double)ng)
		          + 0.3 * std::cos(3.0 * TWO_PI * (double)j / (double)ng);
		double sz = std::sin(TWO_PI * (double)k / (double)ng)
		          + 0.2 * std::cos(4.0 * TWO_PI * (double)k / (double)ng);
		return (real_t)(sx * sy * sz);
	};

	grid_hierarchy gh(1u);
	gh.create_base_hierarchy_slab(lmax);

	// Step (a): the top level must be a slab with expected geometry.
	int local_pass_a = 1;
	if (!gh.is_level_slab(lmax)) local_pass_a = 0;
	MeshvarBnd<real_t> *slab = gh.get_grid(lmax);
	if (!slab) local_pass_a = 0;
	if (slab && (slab->size(1) != ng || slab->size(2) != ng)) local_pass_a = 0;

	// Lower sub-level dims sanity (every rank holds them as full meshes).
	if (lmax >= 1) {
		MeshvarBnd<real_t> *sub = gh.get_grid(lmax - 1);
		if (!sub) local_pass_a = 0;
		else if (sub->size(0) != (ng / 2) || sub->size(1) != (ng / 2)
		         || sub->size(2) != (ng / 2)) local_pass_a = 0;
	}

	// Step (b): each rank fills its strip using global indices.
	long long my_lx = slab ? (long long)slab->local_x_start() : 0;
	long long my_nx = slab ? (long long)slab->local_nx()      : 0;
	if (slab) {
#pragma omp parallel for
		for (long long ii = 0; ii < my_nx; ++ii) {
			size_t g_i = (size_t)(my_lx + ii);
			for (size_t j = 0; j < ng; ++j)
				for (size_t k = 0; k < ng; ++k)
					(*slab)((int)ii, (int)j, (int)k) = pattern(g_i, j, k);
		}
	}

	// Step (c): gather → rank 0 full, compare to serial reference.
	gh.convert_level_slab_to_full(lmax);

	int rc = 0;
#ifdef USE_MPI
	int pass_a_all = local_pass_a;
	MPI_Allreduce(MPI_IN_PLACE, &pass_a_all, 1, MPI_INT, MPI_MIN,
	              MUSIC::mpi::world());
	local_pass_a = pass_a_all;
#endif

	if (is_root) {
		MeshvarBnd<real_t> *full = gh.get_grid(lmax);
		double max_abs_diff = 0.0, max_abs_ref = 0.0;
		bool dims_ok = (full != NULL)
		            && (full->size(0) == ng)
		            && (full->size(1) == ng)
		            && (full->size(2) == ng);
		if (full) {
			for (size_t i = 0; i < ng; ++i)
				for (size_t j = 0; j < ng; ++j)
					for (size_t k = 0; k < ng; ++k) {
						double ref = (double)pattern(i, j, k);
						double got = (double)(*full)((int)i, (int)j, (int)k);
						double d   = std::fabs(got - ref);
						if (d > max_abs_diff) max_abs_diff = d;
						if (std::fabs(ref) > max_abs_ref) max_abs_ref = std::fabs(ref);
					}
		}
		double rel = max_abs_diff / std::max(max_abs_ref, 1e-300);
		const double tol = (sizeof(real_t) == sizeof(float)) ? 1e-6 : 1e-14;
		bool pass = local_pass_a && dims_ok && (rel <= tol);
		std::cerr << "[h4-base-slab-smoke] lmax=" << lmax
		          << " ng=" << ng
		          << " ranks=" << nproc
		          << " geometry=" << (local_pass_a ? "ok" : "BAD")
		          << " gather_dims=" << (dims_ok ? "ok" : "BAD")
		          << " max|gather-serial|=" << max_abs_diff
		          << " rel=" << rel
		          << " tol=" << tol
		          << " " << (pass ? "PASS" : "FAIL")
		          << "\n";
		rc = pass ? 0 : 2;
	}

#ifdef USE_MPI
	MPI_Bcast(&rc, 1, MPI_INT, 0, MUSIC::mpi::world());
#endif
	return rc;
}

void GenerateDensityUnigrid(config_file &cf, transfer_function *ptf, tf_type type,
							refinement_hierarchy &refh, rand_gen &rand, grid_hierarchy &delta, bool smooth, bool shift)
{
	unsigned levelmin, levelmax, levelminPoisson;

	levelminPoisson = cf.getValue<unsigned>("setup", "levelmin");
	levelmin = cf.getValueSafe<unsigned>("setup", "levelmin_TF", levelminPoisson);
	levelmax = cf.getValue<unsigned>("setup", "levelmax");

	bool kspace = cf.getValue<bool>("setup", "kspace_TF");

	bool fix  = cf.getValueSafe<bool>("setup","fix_mode_amplitude",false);
	bool flip = cf.getValueSafe<bool>("setup","flip_mode_amplitude",false);

	unsigned nbase = 1 << levelmin;

	std::cerr << " - Running unigrid version\n";
	LOGUSER("Running unigrid density convolution...");

	//... select the transfer function to be used
	convolution::kernel_creator *the_kernel_creator;

	if (kspace)
	{
		std::cout << " - Using k-space transfer function kernel.\n";
		LOGUSER("Using k-space transfer function kernel.");

#ifdef SINGLE_PRECISION
		the_kernel_creator = convolution::get_kernel_map()["tf_kernel_k_float"];
#else
		the_kernel_creator = convolution::get_kernel_map()["tf_kernel_k_double"];
#endif
	}
	else
	{
		std::cout << " - Using real-space transfer function kernel.\n";
		LOGUSER("Using real-space transfer function kernel.");

#ifdef SINGLE_PRECISION
		the_kernel_creator = convolution::get_kernel_map()["tf_kernel_real_float"];
#else
		the_kernel_creator = convolution::get_kernel_map()["tf_kernel_real_double"];
#endif
	}

	//... initialize convolution kernel
	convolution::kernel *the_tf_kernel = the_kernel_creator->create(cf, ptf, refh, type);

	//...
	std::cout << " - Performing noise convolution on level " << std::setw(2) << levelmax << " ..." << std::endl;
	LOGUSER("Performing noise convolution on level %3d", levelmax);

	//... create convolution mesh
	DensityGrid<real_t> *top = new DensityGrid<real_t>(nbase, nbase, nbase);

	//... fill with random numbers
	rand.load(*top, levelmin);

	//... load convolution kernel
	the_tf_kernel->fetch_kernel(levelmin, false);

	//... perform convolution
	convolution::perform<real_t>(the_tf_kernel, reinterpret_cast<void *>(top->get_data_ptr()), shift, fix, flip);

	//... clean up kernel
	delete the_tf_kernel;

	//... create multi-grid hierarchy
	delta.create_base_hierarchy(levelmin);

	//... copy convolved field to multi-grid hierarchy
	top->copy(*delta.get_grid(levelmin));

	//... delete convolution grid
	delete top;
}

/*******************************************************************************************/
/*******************************************************************************************/
/*******************************************************************************************/

void GenerateDensityHierarchy(config_file &cf, transfer_function *ptf, tf_type type,
							  refinement_hierarchy &refh, rand_gen &rand,
							  grid_hierarchy &delta, bool smooth, bool shift)
{
	unsigned levelmin, levelmax, levelminPoisson;
	std::vector<long> rngseeds;
	std::vector<std::string> rngfnames;
	bool kspaceTF;

	double tstart, tend;

#ifndef SINGLETHREAD_FFTW
	tstart = omp_get_wtime();
#else
	tstart = (double)clock() / CLOCKS_PER_SEC;
#endif

	levelminPoisson = cf.getValue<unsigned>("setup", "levelmin");
	levelmin = cf.getValueSafe<unsigned>("setup", "levelmin_TF", levelminPoisson);
	levelmax = cf.getValue<unsigned>("setup", "levelmax");
	kspaceTF = cf.getValue<bool>("setup", "kspace_TF");

	// B.2b.2.2.1.d: when SPMD multigrid is enabled, workers must hold the same
	// union skeleton (m_pgrids[levelmin..levelmax]) as rank 0 so the V-cycle's
	// single-box-parent broadcasts have matching geometry on every rank. The
	// memory cost is the union mesh at each level on every worker — acceptable
	// at test scales; cluster-to-rank deferral (#80/#81) handles pillars scale.
	const bool spmd_mg_skel =
		cf.getValueSafe<bool>("setup", "zoom_slab_spmd_multigrid", false);
	const bool need_union_skel = MUSIC::mpi::is_root() || spmd_mg_skel;

	// Load-balance instrumentation: per-level wall time split into rank-0
	// serial (pre/post; workers idle in worker_pump) vs SPMD (conv/interp).
	// slot 0 = levelmin, slot i = levelmin+i.
	const int prof_nlev = (int)levelmax - (int)levelmin + 1;
	std::vector<double> prof_pre(prof_nlev, 0.0);
	std::vector<double> prof_conv(prof_nlev, 0.0);
	std::vector<double> prof_interp(prof_nlev, 0.0);
	std::vector<double> prof_post(prof_nlev, 0.0);
	// Sub-phase split of pre/post (rank-0 serial only). pre_alloc = new
	// PaddedDensitySubGrid; pre_load = rand.load (disk I/O); post_addpatch =
	// delta.add_patch; post_copy = fine->copy_unpad. Helps spot whether the
	// rank-0 bottleneck is disk I/O or the single-threaded unpad triple loop.
	std::vector<double> prof_pre_alloc(prof_nlev, 0.0);
	std::vector<double> prof_pre_load(prof_nlev, 0.0);
	std::vector<double> prof_post_addpatch(prof_nlev, 0.0);
	std::vector<double> prof_post_copy(prof_nlev, 0.0);
	double prof_tail = 0.0;  // populate_per_box / sync / scatter at end

	bool fix  = cf.getValueSafe<bool>("setup","fix_mode_amplitude",false);
	bool flip = cf.getValueSafe<bool>("setup","flip_mode_amplitude",false);

	if( fix && levelmin != levelmax ){
		LOGWARN("You have chosen mode fixing for a zoom. This is not well tested,\n please proceed at your own risk...");
	}

	blend_sharpness = cf.getValueSafe<double>("setup", "kspace_filter", blend_sharpness);

	unsigned nbase = 1 << levelmin;

	convolution::kernel_creator *the_kernel_creator;

	if (kspaceTF)
	{
		std::cout << " - Using k-space transfer function kernel.\n";
		LOGUSER("Using k-space transfer function kernel.");

#ifdef SINGLE_PRECISION
		the_kernel_creator = convolution::get_kernel_map()["tf_kernel_k_float"];
#else
		the_kernel_creator = convolution::get_kernel_map()["tf_kernel_k_double"];
#endif
	}
	else
	{
		std::cout << " - Using real-space transfer function kernel.\n";
		LOGUSER("Using real-space transfer function kernel.");
#ifdef SINGLE_PRECISION
		the_kernel_creator = convolution::get_kernel_map()["tf_kernel_real_float"];
#else
		the_kernel_creator = convolution::get_kernel_map()["tf_kernel_real_double"];
#endif
	}

	convolution::kernel *the_tf_kernel = the_kernel_creator->create(cf, ptf, refh, type);

	/***** PERFORM CONVOLUTIONS *****/
	if (kspaceTF)
	{

		//... create and initialize density grids with white noise
		DensityGrid<real_t> *top(NULL);
		PaddedDensitySubGrid<real_t> *coarse(NULL), *fine(NULL);
		int nlevels = (int)levelmax - (int)levelmin + 1;

		const bool is_root = MUSIC::mpi::is_root();
		// H.4 production wire: opt-in slab storage for the levelmin density.
		// Scoped to unigrid (levelmin == levelmax) + MPI; on a unigrid run the
		// refinement loop below does not execute and `top` is never consumed
		// downstream, so the rank-0 full DensityGrid materialization can be
		// skipped end-to-end. H.3b (coarsen_density) + H.3c (normalize_density)
		// are SPMD-safe on slab GH, so production consumers tolerate the flip.
		const bool h4_slab_active = (levelmin == levelmax)
		                         && (MUSIC::mpi::size() > 1)
		                         && cf.getValueSafe<bool>("setup", "slab_levelmin_density", false);
		Meshvar<real_t>* slab_keep = NULL;
		double __t_lvl = omp_get_wtime();
#ifdef USE_MPI
		if (MUSIC::mpi::size() > 1) {
			// SPMD coarse level: every rank loads its own slab of the
			// white noise from wnoise_NNNN.bin and participates in the
			// collective convolution. Only rank 0 allocates the full
			// top-level DensityGrid (used downstream by the zoom loop
			// and the delta hierarchy copy).
			LOGINFO("Performing noise convolution on level %3d (SPMD%s)",
			        levelmin, h4_slab_active ? "; H.4 slab-keep" : "");
			Meshvar<real_t>* slab = MUSIC::dist::make_slab_meshvar<real_t>(
				(size_t)nbase, (size_t)nbase, (size_t)nbase, /*fftw_inplace_pad=*/true);
			rand.load_slab(slab->m_pdata,
			               slab->local_x_start(), slab->local_nx(),
			               (size_t)nbase, (size_t)nbase,
			               (size_t)(2*(nbase/2+1)), levelmin);
			if (!h4_slab_active && is_root)
				top = new DensityGrid<real_t>(nbase, nbase, nbase);
			convolution::perform_dist_slab<real_t>(
				the_tf_kernel->fetch_kernel(levelmin, false),
				slab,
				h4_slab_active ? (real_t*)NULL
				               : (is_root ? top->get_data_ptr() : (real_t*)NULL),
				(size_t)nbase, (size_t)nbase, (size_t)nbase,
				shift, fix, flip);
			if (h4_slab_active) slab_keep = slab;
			else delete slab;
		} else
#endif
		{
			if (is_root) {
				top = new DensityGrid<real_t>(nbase, nbase, nbase);
				LOGINFO("Performing noise convolution on level %3d", levelmin);
				rand.load(*top, levelmin);
			}
			convolution::perform_dist<real_t>(
				the_tf_kernel->fetch_kernel(levelmin, false),
				is_root ? top->get_data_ptr() : (real_t*)NULL,
				(size_t)nbase, (size_t)nbase, (size_t)nbase,
				shift, fix, flip);
		}
		prof_conv[0] += omp_get_wtime() - __t_lvl;

		// Phase D.7.1: workers no longer early-return. They participate in
		// every collective perform_dist() inside the zoom loop below so the
		// per-level FFT convolution is slab-distributed across all ranks.

		__t_lvl = omp_get_wtime();
		if (h4_slab_active) {
			// H.4: every rank allocates full meshes for levels 0..levelmin-1
			// plus its own slab at levelmin. Strip FFT padding from the
			// convolved slab into delta's slab MeshvarBnd. No rank-0 full
			// materialization anywhere on this path.
			LOGINFO("H.4: create_base_hierarchy_slab(levelmin=%u) on all ranks",
			        (unsigned)levelmin);
			delta.create_base_hierarchy_slab((unsigned)levelmin);
			MeshvarBnd<real_t>* gslab = delta.get_grid(levelmin);
			const size_t lnx       = (size_t)slab_keep->local_nx();
			const size_t nz_padded = 2 * ((size_t)nbase/2 + 1);
#pragma omp parallel for
			for (int ix = 0; ix < (int)lnx; ++ix)
				for (size_t iy = 0; iy < (size_t)nbase; ++iy)
					for (size_t iz = 0; iz < (size_t)nbase; ++iz)
						(*gslab)(ix, (int)iy, (int)iz) =
							slab_keep->m_pdata[(ix * (size_t)nbase + iy) * nz_padded + iz];
			delete slab_keep;
			slab_keep = NULL;
		} else {
			if (need_union_skel)
				delta.create_base_hierarchy(levelmin);
			if (is_root)
				top->copy(*delta.get_grid(levelmin));
		}
		prof_post[0] += omp_get_wtime() - __t_lvl;

#ifdef USE_MPI
		// Phase E.2.1a smoke test: verify the E.2.0 slab gather/scatter plumbing
		// round-trips bit-identical on real density data. Opt-in via config
		// (default off so production runs are unchanged). Runs only on the
		// kspaceTF=true SPMD path because that's the first consumer site we
		// plan to migrate to slab storage in E.2.1b.
		if (MUSIC::mpi::size() > 1 && cf.getValueSafe<bool>("setup", "test_slab_roundtrip", false)) {
			delta.test_slab_roundtrip_at((unsigned)levelmin,
			                              (size_t)nbase, (size_t)nbase, (size_t)nbase);
		}

		// Phase E.2.1b: opt-in storage flip. After rank 0 has the union populated
		// from top->copy, scatter to per-rank slabs (workers' delta gets the slot
		// allocated), then immediately gather back to a rank-0 full union before
		// proceeding. Smoke-tests the real storage round-trip in the production
		// data path; future E.2.x widens the slab-active span between the converts.
		if (MUSIC::mpi::size() > 1 && cf.getValueSafe<bool>("setup", "slab_levelmin", false)) {
			LOGINFO("E.2.1b: levelmin storage flip full->slab->full (L=%u, %zu^3)",
			        (unsigned)levelmin, (size_t)nbase);
			delta.convert_level_full_to_slab((unsigned)levelmin,
			                                  (size_t)nbase, (size_t)nbase, (size_t)nbase);
			delta.convert_level_slab_to_full((unsigned)levelmin);
		}

		// Phase E.2.2a smoke test: verify the SPMD slab-solve primitive matches
		// the existing Scatterv/FFT/Gatherv full-path solve bit-by-bit. Opt-in
		// (default off). Internally flips storage full→slab, runs the slab
		// solve, gathers, compares, then flips slab→full so delta is restored.
		if (MUSIC::mpi::size() > 1 && cf.getValueSafe<bool>("setup", "test_slab_solve", false)) {
			delta.test_slab_solve_at((unsigned)levelmin,
			                          (size_t)nbase, (size_t)nbase, (size_t)nbase);
		}
#endif

		// density-boost: rank-0 pre/post blocks below are serial (workers idle in
		// MPI_Barrier inside perform_dist). pre_load's plane-copy loop, copy_unpad,
		// and MeshvarBnd::zero() are all #pragma omp parallel, so growing rank-0's
		// OMP thread pool to claim the workers' cores cuts those serial windows.
		// Mirrors the B2 pattern from random.cc dispatcher (no FFTW thread change
		// needed here — no plan creation in this scope).
		int dens_omp_saved = is_root ? omp_get_max_threads() : 0;
		int dens_omp_boost = dens_omp_saved;
		if (is_root) {
			const std::string boost_mode =
			    cf.getValueSafe<std::string>("setup", "density_boost_threads",
			                                  std::string("auto"));
			if (boost_mode != std::string("no")) {
				if (boost_mode == std::string("auto")) {
					const int avail = omp_get_num_procs();
					const int cap   = dens_omp_saved * std::max(1, MUSIC::mpi::size());
					dens_omp_boost = std::min(avail, cap);
				} else {
					try { dens_omp_boost = std::max(1, std::stoi(boost_mode)); }
					catch (...) { dens_omp_boost = dens_omp_saved; }
				}
			}
			if (dens_omp_boost > dens_omp_saved) {
				LOGINFO("density-boost: rank-0 OMP %d -> %d during refinement pre/post",
				        dens_omp_saved, dens_omp_boost);
			}
		}

		for (int i = 1; i < nlevels; ++i)
		{
			double __t_ph = omp_get_wtime();
			if (is_root) {
				LOGINFO("Performing noise convolution on level %3d... (SPMD)", levelmin + i);
				LOGUSER("Allocating refinement patch");
				LOGUSER("   offset=(%5d,%5d,%5d)", refh.offset(levelmin + i, 0),
						refh.offset(levelmin + i, 1), refh.offset(levelmin + i, 2));
				LOGUSER("   size  =(%5d,%5d,%5d)", refh.size(levelmin + i, 0),
						refh.size(levelmin + i, 1), refh.size(levelmin + i, 2));

				if (dens_omp_boost > dens_omp_saved)
					omp_set_num_threads(dens_omp_boost);
				double __t_sub = omp_get_wtime();
				fine = new PaddedDensitySubGrid<real_t>(refh.offset(levelmin + i, 0),
														refh.offset(levelmin + i, 1),
														refh.offset(levelmin + i, 2),
														refh.size(levelmin + i, 0),
														refh.size(levelmin + i, 1),
														refh.size(levelmin + i, 2));
				prof_pre_alloc[i] += omp_get_wtime() - __t_sub;

				// load white noise for patch
				__t_sub = omp_get_wtime();
				rand.load(*fine, levelmin + i);
				prof_pre_load[i] += omp_get_wtime() - __t_sub;
				if (dens_omp_boost > dens_omp_saved)
					omp_set_num_threads(dens_omp_saved);
			}
			prof_pre[i] += omp_get_wtime() - __t_ph;

			// Collective: PaddedDensitySubGrid stores 2*refh.size() per axis,
			// which is the FFT extent the kernel was sampled for. Workers don't
			// own `fine`; they create a local FFTW slab inside perform_dist and
			// participate in the scatter/FFT/gather. Only rank 0's data pointer
			// is meaningful.
			const size_t gnx = 2 * (size_t)refh.size(levelmin + i, 0);
			const size_t gny = 2 * (size_t)refh.size(levelmin + i, 1);
			const size_t gnz = 2 * (size_t)refh.size(levelmin + i, 2);
			__t_ph = omp_get_wtime();
			convolution::perform_dist<real_t>(
				the_tf_kernel->fetch_kernel(levelmin + i, /*isolated=*/true),
				is_root ? reinterpret_cast<real_t*>(fine->get_data_ptr())
				        : (real_t*)NULL,
				gnx, gny, gnz,
				shift, fix, flip);
			prof_conv[i] += omp_get_wtime() - __t_ph;

			// Phase D.7.2: collective FFT-based interpolate. Workers pass NULL
			// for V/v; the function broadcasts geometry, distributes the fine
			// FFT across ranks via FFTW3-MPI, and writes the result back into
			// *fine on rank 0.
			__t_ph = omp_get_wtime();
			if (i == 1)
				fft_interpolate_dist(top, fine, true);
			else
				fft_interpolate_dist(coarse, fine, false);
			prof_interp[i] += omp_get_wtime() - __t_ph;

			__t_ph = omp_get_wtime();
			if (need_union_skel) {
				if (is_root && dens_omp_boost > dens_omp_saved)
					omp_set_num_threads(dens_omp_boost);
				double __t_sub_ap = omp_get_wtime();
				delta.add_patch(refh.offset(levelmin + i, 0),
								refh.offset(levelmin + i, 1),
								refh.offset(levelmin + i, 2),
								refh.size(levelmin + i, 0),
								refh.size(levelmin + i, 1),
								refh.size(levelmin + i, 2));
				if (is_root)
					prof_post_addpatch[i] += omp_get_wtime() - __t_sub_ap;
			}
			if (is_root) {
				double __t_sub = omp_get_wtime();
				fine->copy_unpad(*delta.get_grid(levelmin + i));
				prof_post_copy[i] += omp_get_wtime() - __t_sub;

				if (i == 1)
					delete top;
				else
					delete coarse;

				coarse = fine;
				if (dens_omp_boost > dens_omp_saved)
					omp_set_num_threads(dens_omp_saved);
			}
			prof_post[i] += omp_get_wtime() - __t_ph;
		}

		if (is_root)
			delete coarse;
	}
	else
	{
		//... create and initialize density grids with white noise
		PaddedDensitySubGrid<real_t> *coarse(NULL), *fine(NULL);
		DensityGrid<real_t> *top(NULL);
		const bool is_root = MUSIC::mpi::is_root();

		if (levelmax == levelmin)
		{
			// H.4 production wire: opt-in slab storage for the levelmin density
			// (mirrors the kspaceTF=true branch above).
			const bool h4_slab_active = (MUSIC::mpi::size() > 1)
			    && cf.getValueSafe<bool>("setup", "slab_levelmin_density", false);
			Meshvar<real_t>* slab_keep = NULL;
#ifdef USE_MPI
			if (MUSIC::mpi::size() > 1) {
				// SPMD: every rank loads its own noise slab from
				// wnoise_NNNN.bin, fetches its kernel slab, and runs
				// the collective convolution. Only rank 0 gathers the
				// final padded grid for downstream copy into delta.
				if (is_root) {
					std::cout << " - Performing noise convolution on level "
							  << std::setw(2) << levelmax
							  << " (SPMD" << (h4_slab_active ? "; H.4 slab-keep" : "")
							  << ") ..." << std::endl;
				}
				LOGUSER("Performing noise convolution on level %3d... (SPMD%s)",
				        levelmax, h4_slab_active ? "; H.4 slab-keep" : "");

				Meshvar<real_t>* slab = MUSIC::dist::make_slab_meshvar<real_t>(
					(size_t)nbase, (size_t)nbase, (size_t)nbase, /*fftw_inplace_pad=*/true);
				rand.load_slab(slab->m_pdata,
				               slab->local_x_start(), slab->local_nx(),
				               (size_t)nbase, (size_t)nbase,
				               (size_t)(2*(nbase/2+1)), levelmin);
				if (!h4_slab_active && is_root)
					top = new DensityGrid<real_t>(nbase, nbase, nbase);
				convolution::perform_dist_slab<real_t>(
					the_tf_kernel->fetch_kernel(levelmax, /*isolated=*/false, /*distributed=*/true),
					slab,
					h4_slab_active ? (real_t*)NULL
					               : (is_root ? top->get_data_ptr() : (real_t*)NULL),
					(size_t)nbase, (size_t)nbase, (size_t)nbase,
					shift, fix, flip);
				the_tf_kernel->deallocate();
				if (h4_slab_active) slab_keep = slab;
				else delete slab;
			} else
#endif
			{
				if (is_root) {
					std::cout << " - Performing noise convolution on level "
							  << std::setw(2) << levelmax << " ..." << std::endl;
					LOGUSER("Performing noise convolution on level %3d...", levelmax);

					top = new DensityGrid<real_t>(nbase, nbase, nbase);
					rand.load(*top, levelmin);
				}

				// Collective: every rank participates in the slab-distributed
				// kernel load + FFT + multiply + inverse FFT. Workers pass a
				// NULL data pointer; only root owns the full padded grid.
				convolution::perform_dist<real_t>(
					the_tf_kernel->fetch_kernel(levelmax, /*isolated=*/false, /*distributed=*/true),
					is_root ? top->get_data_ptr() : (real_t*)NULL,
					(size_t)nbase, (size_t)nbase, (size_t)nbase,
					shift, fix, flip);
				the_tf_kernel->deallocate();
			}

			if (h4_slab_active) {
				LOGINFO("H.4: create_base_hierarchy_slab(levelmin=%u) on all ranks",
				        (unsigned)levelmin);
				delta.create_base_hierarchy_slab((unsigned)levelmin);
				MeshvarBnd<real_t>* gslab = delta.get_grid(levelmin);
				const size_t lnx       = (size_t)slab_keep->local_nx();
				const size_t nz_padded = 2 * ((size_t)nbase/2 + 1);
#pragma omp parallel for
				for (int ix = 0; ix < (int)lnx; ++ix)
					for (size_t iy = 0; iy < (size_t)nbase; ++iy)
						for (size_t iz = 0; iz < (size_t)nbase; ++iz)
							(*gslab)(ix, (int)iy, (int)iz) =
								slab_keep->m_pdata[(ix * (size_t)nbase + iy) * nz_padded + iz];
				delete slab_keep;
				slab_keep = NULL;
			} else {
				if (need_union_skel)
					delta.create_base_hierarchy(levelmin);
				if (is_root) {
					top->copy(*delta.get_grid(levelmin));
					delete top;
				}
			}
		}

		// Phase D.7.3: workers no longer early-return. They participate in
		// every collective perform_dist() inside the kspace_TF=false zoom loop
		// below. All rank-0-only state (top/coarse/fine pointers, delta
		// hierarchy, rand.load) stays gated by is_root; workers pass NULL
		// data pointers into perform_dist.

		for (int i = 0; i < (int)levelmax - (int)levelmin; ++i)
		{
			//.......................................................................................................//
			//... GENERATE/FILL WITH RANDOM NUMBERS .................................................................//
			//.......................................................................................................//

			if (is_root) {
				if (i == 0)
				{
					top = new DensityGrid<real_t>(nbase, nbase, nbase);
					rand.load(*top, levelmin);
				}

				fine = new PaddedDensitySubGrid<real_t>(refh.offset(levelmin + i + 1, 0),
														refh.offset(levelmin + i + 1, 1),
														refh.offset(levelmin + i + 1, 2),
														refh.size(levelmin + i + 1, 0),
														refh.size(levelmin + i + 1, 1),
														refh.size(levelmin + i + 1, 2));

				rand.load(*fine, levelmin + i + 1);
			}

			//.......................................................................................................//
			//... PERFORM CONVOLUTIONS ..............................................................................//
			//.......................................................................................................//
			if (i == 0)
			{
				/**********************************************************************************************************\
			 *	multi-grid: top-level grid grids .....
			 \**********************************************************************************************************/
				if (is_root) {
					std::cout << " - Performing noise convolution on level "
							  << std::setw(2) << levelmin + i << " ..." << std::endl;
				}
				LOGUSER("Performing noise convolution on level %3d", levelmin + i);

				if (is_root)
					LOGUSER("Creating base hierarchy...");
				if (need_union_skel)
					delta.create_base_hierarchy(levelmin);

				// top_save is a rank-0-only deep copy used to restore *top
				// between the two convolutions below.
				DensityGrid<real_t> *top_save = is_root ? new DensityGrid<real_t>(*top) : NULL;

				// Top-grid convolution is unigrid at levelmin; kernel and slab
				// both sized nbase. fetch_kernel(distributed=true) is collective.
				convolution::perform_dist<real_t>(
					the_tf_kernel->fetch_kernel(levelmin, /*isolated=*/false, /*distributed=*/true),
					is_root ? reinterpret_cast<real_t*>(top->get_data_ptr()) : (real_t*)NULL,
					(size_t)nbase, (size_t)nbase, (size_t)nbase,
					shift, fix, flip);

				if (is_root) {
					top->copy(*delta.get_grid(levelmin));

					//... 2) compute contribution to finer grids from non-refined region
					LOGUSER("Computing long-range component for finer grid.");
					*top = *top_save;
					top_save->clear();
					top->zero_subgrid(refh.offset(levelmin + i + 1, 0), refh.offset(levelmin + i + 1, 1), refh.offset(levelmin + i + 1, 2),
									  refh.size(levelmin + i + 1, 0) / 2, refh.size(levelmin + i + 1, 1) / 2, refh.size(levelmin + i + 1, 2) / 2);
				}

				convolution::perform_dist<real_t>(
					the_tf_kernel->fetch_kernel(levelmin, /*isolated=*/false, /*distributed=*/true),
					is_root ? reinterpret_cast<real_t*>(top->get_data_ptr()) : (real_t*)NULL,
					(size_t)nbase, (size_t)nbase, (size_t)nbase,
					shift, fix, flip);
				the_tf_kernel->deallocate();

				// Stash long-range slice on rank 0 before add_patch (which is now
				// SPMD under need_union_skel). Workers skip both the stash and the
				// prolong; their union mesh at levelmin+1 just holds zeros, which
				// the SPMD V-cycle later overwrites via per-box restrict + single-
				// box-parent broadcast.
				meshvar_bnd * delta_longrange = NULL;
				if (is_root) {
					delta_longrange = new meshvar_bnd(*delta.get_grid(levelmin));
					top->copy(*delta_longrange);
					delete top;
					delete top_save;

					LOGUSER("Allocating refinement patch");
					LOGUSER("   offset=(%5d,%5d,%5d)", refh.offset(levelmin + 1, 0), refh.offset(levelmin + 1, 1), refh.offset(levelmin + 1, 2));
					LOGUSER("   size  =(%5d,%5d,%5d)", refh.size(levelmin + 1, 0), refh.size(levelmin + 1, 1), refh.size(levelmin + 1, 2));
				}
				if (need_union_skel) {
					delta.add_patch(refh.offset(levelmin + 1, 0), refh.offset(levelmin + 1, 1), refh.offset(levelmin + 1, 2),
									refh.size(levelmin + 1, 0), refh.size(levelmin + 1, 1), refh.size(levelmin + 1, 2));
				}
				if (is_root) {
					LOGUSER("Injecting long range component");
					mg_cubic().prolong(*delta_longrange, *delta.get_grid(levelmin + 1));
					delete delta_longrange;
				}
			}
			else
			{
				/**********************************************************************************************************\
			 *	multi-grid: intermediate sub-grids .....
			 \**********************************************************************************************************/

				if (is_root) {
					std::cout << " - Performing noise convolution on level " << std::setw(2) << levelmin + i << " ..." << std::endl;
				}
				LOGUSER("Performing noise convolution on level %3d", levelmin + i);

				if (is_root) {
					//... add new refinement patch
					LOGUSER("Allocating refinement patch");
					LOGUSER("   offset=(%5d,%5d,%5d)", refh.offset(levelmin + i + 1, 0), refh.offset(levelmin + i + 1, 1), refh.offset(levelmin + i + 1, 2));
					LOGUSER("   size  =(%5d,%5d,%5d)", refh.size(levelmin + i + 1, 0), refh.size(levelmin + i + 1, 1), refh.size(levelmin + i + 1, 2));
				}
				if (need_union_skel) {
					delta.add_patch(refh.offset(levelmin + i + 1, 0), refh.offset(levelmin + i + 1, 1), refh.offset(levelmin + i + 1, 2),
									refh.size(levelmin + i + 1, 0), refh.size(levelmin + i + 1, 1), refh.size(levelmin + i + 1, 2));
				}
				if (is_root) {
					LOGUSER("Injecting long range component");
					mg_cubic().prolong(*delta.get_grid(levelmin + i), *delta.get_grid(levelmin + i + 1));
				}

				// All ranks compute zoom-level FFT extents from refh.
				const size_t gnx = 2 * (size_t)refh.size(levelmin + i, 0);
				const size_t gny = 2 * (size_t)refh.size(levelmin + i, 1);
				const size_t gnz = 2 * (size_t)refh.size(levelmin + i, 2);

				PaddedDensitySubGrid<real_t> *coarse_save = is_root ? new PaddedDensitySubGrid<real_t>(*coarse) : NULL;

				if (is_root) {
					//... 1) the inner region
					LOGUSER("Computing density self-contribution");
					coarse->subtract_boundary_oct_mean();
				}
				convolution::perform_dist<real_t>(
					the_tf_kernel->fetch_kernel(levelmin + i, /*isolated=*/false, /*distributed=*/true),
					is_root ? reinterpret_cast<real_t*>(coarse->get_data_ptr()) : (real_t*)NULL,
					gnx, gny, gnz,
					shift, fix, flip);
				if (is_root) {
					coarse->copy_add_unpad(*delta.get_grid(levelmin + i));

					//... 2) the 'BC' for the next finer grid
					LOGUSER("Computing long-range component for finer grid.");
					*coarse = *coarse_save;
					coarse->subtract_boundary_oct_mean();
					coarse->zero_subgrid(refh.offset(levelmin + i + 1, 0), refh.offset(levelmin + i + 1, 1), refh.offset(levelmin + i + 1, 2),
										 refh.size(levelmin + i + 1, 0) / 2, refh.size(levelmin + i + 1, 1) / 2, refh.size(levelmin + i + 1, 2) / 2);
				}

				convolution::perform_dist<real_t>(
					the_tf_kernel->fetch_kernel(levelmin + i, /*isolated=*/false, /*distributed=*/true),
					is_root ? reinterpret_cast<real_t*>(coarse->get_data_ptr()) : (real_t*)NULL,
					gnx, gny, gnz,
					shift, fix, flip);

				if (is_root) {
					//... interpolate to finer grid(s)
					meshvar_bnd delta_longrange(*delta.get_grid(levelmin + i));
					coarse->copy_unpad(delta_longrange);

					LOGUSER("Injecting long range component");
					mg_cubic().prolong_add(delta_longrange, *delta.get_grid(levelmin + i + 1));

					//... 3) the coarse-grid correction
					LOGUSER("Computing coarse grid correction");
					*coarse = *coarse_save;
					coarse->subtract_oct_mean();
				}
				convolution::perform_dist<real_t>(
					the_tf_kernel->fetch_kernel(levelmin + i, /*isolated=*/false, /*distributed=*/true),
					is_root ? reinterpret_cast<real_t*>(coarse->get_data_ptr()) : (real_t*)NULL,
					gnx, gny, gnz,
					shift, fix, flip);
				if (is_root) {
					coarse->subtract_mean();
					coarse->upload_bnd_add(*delta.get_grid(levelmin + i - 1));

					//... clean up
					delete coarse;
					delete coarse_save;
				}
				the_tf_kernel->deallocate();
			}

			if (is_root) coarse = fine;
		}

		//... and convolution for finest grid (outside loop)
		if (levelmax > levelmin)
		{
			/**********************************************************************************************************\
			 *	multi-grid: finest sub-grid .....
			\**********************************************************************************************************/
			if (is_root) {
				std::cout << " - Performing noise convolution on level " << std::setw(2) << levelmax << " ..." << std::endl;
			}
			LOGUSER("Performing noise convolution on level %3d", levelmax);

			//... 1) grid self-contribution
			LOGUSER("Computing density self-contribution");
			PaddedDensitySubGrid<real_t> *coarse_save = is_root ? new PaddedDensitySubGrid<real_t>(*coarse) : NULL;

			// All ranks compute finest-level FFT extents from refh.
			const size_t gnx = 2 * (size_t)refh.size(levelmax, 0);
			const size_t gny = 2 * (size_t)refh.size(levelmax, 1);
			const size_t gnz = 2 * (size_t)refh.size(levelmax, 2);

			if (is_root) {
				//... subtract oct mean on boundary but not in interior
				coarse->subtract_boundary_oct_mean();
			}

			//... perform convolution
			convolution::perform_dist<real_t>(
				the_tf_kernel->fetch_kernel(levelmax, /*isolated=*/false, /*distributed=*/true),
				is_root ? reinterpret_cast<real_t*>(coarse->get_data_ptr()) : (real_t*)NULL,
				gnx, gny, gnz,
				shift, fix, flip);

			if (is_root) {
				//... copy to grid hierarchy
				coarse->copy_add_unpad(*delta.get_grid(levelmax));

				//... 2) boundary correction to top grid
				LOGUSER("Computing coarse grid correction");
				*coarse = *coarse_save;

				//... subtract oct mean
				coarse->subtract_oct_mean();
			}

			//... perform convolution
			convolution::perform_dist<real_t>(
				the_tf_kernel->fetch_kernel(levelmax, /*isolated=*/false, /*distributed=*/true),
				is_root ? reinterpret_cast<real_t*>(coarse->get_data_ptr()) : (real_t*)NULL,
				gnx, gny, gnz,
				shift, fix, flip);

			the_tf_kernel->deallocate();

			if (is_root) {
				coarse->subtract_mean();

				//... upload data to coarser grid
				coarse->upload_bnd_add(*delta.get_grid(levelmax - 1));

				delete coarse;
				delete coarse_save;
			}
		}
	}

	delete the_tf_kernel;

	// Phase D.2b/E.1b: allocate per-box MeshvarBnd sub-meshes alongside the
	// union meshes. populate is SPMD (every rank allocs only the boxes it
	// owns per m_pbox_owner_). The union mesh lives on rank 0, so initial
	// sync runs there: rank 0 alloc_root_tenants -> sync_per_box_from_union
	// fills both owned slots and tenants -> collective scatter ships tenants
	// to their owners and frees the rank-0 tenants. Under owner=0 the alloc
	// and scatter helpers are no-ops (every box is already on rank 0).
	double __t_tail = omp_get_wtime();
	delta.populate_per_box_meshes(refh.get_level_boxes());
	if( MUSIC::mpi::is_root() ){
		delta.alloc_root_tenants();
		delta.sync_per_box_from_union();
		delta.log_per_box_stats("density");
	}
	delta.scatter_per_box_from_root();

	// Phase G.1 (opt-in): allocate per-cluster z-slab buffers and seed them
	// from the now-populated per-box meshes. Default off → bit-identical with
	// pre-G.1 behaviour (G.1 only adds memory; no compute path consumes it
	// yet — G.2+ flips MG operators to read/write the slabs).
	if( cf.getValueSafe<bool>("setup", "zoom_slab", false) ){
		const int halo_w = cf.getValueSafe<int>("setup", "zoom_slab_halo", (int)delta.m_nbnd);
		delta.populate_per_box_zslabs(halo_w);
		delta.sync_zslab_from_per_box();
		if( cf.getValueSafe<bool>("setup", "zoom_slab_verify", false) ){
			if( !delta.verify_zslab_roundtrip() )
				throw std::runtime_error("G.1 zoom_slab roundtrip verification failed");
		}
	}

	// B.1: per-cluster sub_comm registry for G.2b smoother bridge.
	// Default policy "world" preserves A' behaviour at np=1 (subcomm_for_box
	// returns MPI_COMM_WORLD); "self" picks MPI_COMM_SELF per box;
	// "round_robin" splits ranks into nb groups (requires nb<=np).
	// Registry covers all refinement levels [levelmin+1, levelmax].
	if( cf.getValueSafe<bool>("setup", "zoom_slab_smoother", false) ){
		std::string policy = cf.getValueSafe<std::string>("setup", "zoom_slab_subcomm", std::string("world"));
		MUSIC::zoom_slab::set_subcomm_policy(policy);
		std::vector<std::size_t> nb_per_level(refh.levelmax()+1, 0);
		for( unsigned L=refh.levelmin()+1; L<=refh.levelmax(); ++L )
			nb_per_level[L] = refh.num_boxes(L);
		MUSIC::zoom_slab::build_subcomm_registry(nb_per_level);
		LOGINFO("B.1: zoom_slab subcomm policy=\"%s\"; registry built for levels [%u..%u]",
		        MUSIC::zoom_slab::subcomm_policy().c_str(),
		        refh.levelmin()+1, refh.levelmax());
	}
	prof_tail += omp_get_wtime() - __t_tail;

#ifndef SINGLETHREAD_FFTW
	tend = omp_get_wtime();
	if (true) //verbosity > 1 )
		std::cout << " - Density calculation took " << tend - tstart << "s with " << omp_get_max_threads() << " threads." << std::endl;
#else
	tend = (double)clock() / CLOCKS_PER_SEC;
	if (true) //verbosity > 1 )
		std::cout << " - Density calculation took " << tend - tstart << "s." << std::endl;
#endif

	LOGUSER("Finished computing the density field in %fs", tend - tstart);

	// Load-balance profile dump (rank 0 only). pre/post are rank-0 serial
	// blocks (workers idle in worker_pump); conv/interp are SPMD.
	if (MUSIC::mpi::is_root() && kspaceTF) {
		double s_pre = 0.0, s_conv = 0.0, s_interp = 0.0, s_post = 0.0;
		double s_pre_alloc = 0.0, s_pre_load = 0.0;
		double s_post_addpatch = 0.0, s_post_copy = 0.0;
		LOGINFO("density-profile per-level (s): L     pre       conv      interp    post");
		for (int slot = 0; slot < prof_nlev; ++slot) {
			LOGINFO("density-profile  L=%3u %9.4f %9.4f %9.4f %9.4f",
			        (unsigned)(levelmin + slot),
			        prof_pre[slot], prof_conv[slot], prof_interp[slot], prof_post[slot]);
			s_pre += prof_pre[slot];
			s_conv += prof_conv[slot];
			s_interp += prof_interp[slot];
			s_post += prof_post[slot];
			s_pre_alloc += prof_pre_alloc[slot];
			s_pre_load += prof_pre_load[slot];
			s_post_addpatch += prof_post_addpatch[slot];
			s_post_copy += prof_post_copy[slot];
		}
		const double serial = s_pre + s_post + prof_tail;
		const double spmd = s_conv + s_interp;
		const double accounted = serial + spmd;
		const double total = tend - tstart;
		LOGINFO("density-profile  SUM   %9.4f %9.4f %9.4f %9.4f tail=%.4f",
		        s_pre, s_conv, s_interp, s_post, prof_tail);
		LOGINFO("density-profile  total=%.3fs  rank-0 serial=%.3fs (%.1f%%)  SPMD=%.3fs (%.1f%%)  unaccounted=%.3fs",
		        total, serial, 100.0*serial/total, spmd, 100.0*spmd/total, total - accounted);
		LOGINFO("density-profile sub-phase (s): L  pre_alloc  pre_load  post_addp post_copy");
		for (int slot = 0; slot < prof_nlev; ++slot) {
			LOGINFO("density-profile  L=%3u %9.4f %9.4f %9.4f %9.4f",
			        (unsigned)(levelmin + slot),
			        prof_pre_alloc[slot], prof_pre_load[slot],
			        prof_post_addpatch[slot], prof_post_copy[slot]);
		}
		LOGINFO("density-profile  SUM   %9.4f %9.4f %9.4f %9.4f",
		        s_pre_alloc, s_pre_load, s_post_addpatch, s_post_copy);
	}
}

/*******************************************************************************************/
/*******************************************************************************************/
/*******************************************************************************************/

// H.3c: callable collectively on every rank (worker_pump not required).
//
// Three paths:
//   1) MPI size==1: fall through to the original serial code path.
//   2) MPI size>1 AND grid[levelmin] is a slab on rank 0 (H.4 production wire
//      engaged, so every rank holds its own slab): SPMD path — each rank sums
//      its slab, Allreduce to get the global sum, subtract on its local slab.
//      Refinement levels (>= levelmin+1) stay rank-0-only since workers do
//      not own them.
//   3) MPI size>1 AND grid[levelmin] is full on rank 0 (workers may carry an
//      empty hierarchy from rank-0-only GenerateDensityHierarchy): rank-0-only
//      serial path (matches the pre-H.3c behavior bit-by-bit). Workers no-op.
//
// Bit-identicality contract: default behavior (slab_levelmin_density=no) is
// path (1) at np=1 and path (3) at np>1, both bit-identical to pre-H.3c.
void normalize_density(grid_hierarchy &delta)
{
	const bool is_root = MUSIC::mpi::is_root();
#ifdef USE_MPI
	const int sz = MUSIC::mpi::size();
	if (sz > 1) {
		// Workers may carry an empty GH; consult rank 0 for slab status.
		int slab_mode = 0;
		if (is_root) {
			const unsigned lm = delta.levelmin();
			if (lm < delta.levelmax() + 1 && delta.is_level_slab(lm)) slab_mode = 1;
		}
		MPI_Bcast(&slab_mode, 1, MPI_INT, 0, MUSIC::mpi::world());

		if (slab_mode) {
			// SPMD slab path: every rank holds its own slab at levelmin.
			const unsigned levelmin = delta.levelmin();
			const unsigned levelmax = delta.levelmax();
			MeshvarBnd<real_t> *g = delta.get_grid(levelmin);
			const size_t local_nx = g->size(0);
			const size_t ny       = g->size(1);
			const size_t nz       = g->size(2);
			const size_t gnx      = delta.size(levelmin, 0); // global

			long double local_sum = 0.0;
#pragma omp parallel for reduction(+ : local_sum)
			for (int ix = 0; ix < (int)local_nx; ++ix)
				for (size_t iy = 0; iy < ny; ++iy)
					for (size_t iz = 0; iz < nz; ++iz)
						local_sum += (*g)(ix, iy, iz);

			double sum_d = (double)local_sum;
			double global_sum_d = 0.0;
			MPI_Allreduce(&sum_d, &global_sum_d, 1, MPI_DOUBLE, MPI_SUM,
			              MUSIC::mpi::world());
			const long double sum_total = (long double)global_sum_d;
			const long double npts      = (long double)gnx * (long double)ny * (long double)nz;
			const long double sum_mean  = sum_total / npts;

			if (is_root) {
				std::cout << " - Top grid mean density is off by " << sum_mean
				          << ", correcting..." << std::endl;
				LOGUSER("Grid mean density is %g. Correcting...", (double)sum_mean);
			}

			// Subtract on this rank's slab.
#pragma omp parallel for
			for (int ix = 0; ix < (int)local_nx; ++ix)
				for (size_t iy = 0; iy < ny; ++iy)
					for (size_t iz = 0; iz < nz; ++iz)
						(*g)(ix, iy, iz) -= sum_mean;

			// Refinement levels live on rank 0 only (workers have empty GH).
			if (is_root) {
				for (unsigned i = levelmin + 1; i <= levelmax; ++i) {
					MeshvarBnd<real_t> *gi = delta.get_grid(i);
					const size_t nx_i = gi->size(0);
					const size_t ny_i = gi->size(1);
					const size_t nz_i = gi->size(2);
#pragma omp parallel for
					for (int ix = 0; ix < (int)nx_i; ++ix)
						for (size_t iy = 0; iy < ny_i; ++iy)
							for (size_t iz = 0; iz < nz_i; ++iz)
								(*gi)(ix, iy, iz) -= sum_mean;
				}
			}
			return;
		}
		// slab_mode == 0: fall through; rank 0 does serial work, workers no-op.
		if (!is_root) return;
	}
#endif

	// Serial path: np==1, or np>1 with rank-0-only full hierarchy.
	long double sum = 0.0;
	unsigned levelmin = delta.levelmin(), levelmax = delta.levelmax();

	{
		size_t nx, ny, nz;

		nx = delta.get_grid(levelmin)->size(0);
		ny = delta.get_grid(levelmin)->size(1);
		nz = delta.get_grid(levelmin)->size(2);

#pragma omp parallel for reduction(+ \
								   : sum)
		for (int ix = 0; ix < (int)nx; ++ix)
			for (size_t iy = 0; iy < ny; ++iy)
				for (size_t iz = 0; iz < nz; ++iz)
					sum += (*delta.get_grid(levelmin))(ix, iy, iz);

		sum /= (double)(nx * ny * nz);
	}

	std::cout << " - Top grid mean density is off by " << sum << ", correcting..." << std::endl;
	LOGUSER("Grid mean density is %g. Correcting...", sum);

	for (unsigned i = levelmin; i <= levelmax; ++i)
	{
		size_t nx, ny, nz;
		nx = delta.get_grid(i)->size(0);
		ny = delta.get_grid(i)->size(1);
		nz = delta.get_grid(i)->size(2);

#pragma omp parallel for
		for (int ix = 0; ix < (int)nx; ++ix)
			for (size_t iy = 0; iy < ny; ++iy)
				for (size_t iz = 0; iz < nz; ++iz)
					(*delta.get_grid(i))(ix, iy, iz) -= sum;
	}
}

// H.3b: callable collectively on every rank (worker_pump not required). The
// fft branch routes through fft_coarsen_dist so workers participate in the
// slab FFT/reduce; the mg_straight() restrict branch and the cut_patch +
// post-restrict loop stay rank-0-only (no SPMD twin for mg_straight yet).
//
// Bit-identicality contract:
//   - When levelmin_TF == rh.levelmin() the coarsen loop runs 0 iters, no
//     SPMD operation fires, and the function is observably a worker no-op.
//   - When levelmin_TF >  rh.levelmin() AND kspace == true, the SPMD
//     fft_coarsen_dist (H.3a smoke verified max=0 vs serial) replaces serial
//     fft_coarsen on rank 0; output bit-identical at the same np.
//   - When levelmin_TF >  rh.levelmin() AND kspace == false, the rank-0-only
//     mg_straight().restrict branch is unchanged.
//
// Pre-H.3b call sites gated `if (is_root)` MUST drop that gate to engage the
// SPMD coarsen path; lambda bodies inside with_pbox_distributed run rank-0
// only by wpd contract and stay using the rank-0 branches inside this
// function (workers parked in worker_pump never enter).
void coarsen_density(const refinement_hierarchy &rh, GridHierarchy<real_t> &u, bool kspace)
{
	const bool is_root = MUSIC::mpi::is_root();

#ifdef USE_MPI
	// H.4 production wire: when this rank's levelmin grid is a per-rank slab,
	// fft_coarsen_dist / mg_straight().restrict would read past the slab end
	// (they assume full extent on rank 0). H.4 is currently gated to unigrid
	// (levelmin == levelmax), where the cascade output u.get_grid(0..levelmin-1)
	// is NOT consumed downstream (normalize_density and slab_solve_unigrid only
	// touch the levelmin grid). Check locally — no MPI_Bcast — so this guard
	// is safe whether called collectively (all ranks see slab in H.4 mode)
	// OR called rank-0-only (e.g. inside `if (is_root)` blocks with data_forIO,
	// where data_forIO has already been gathered back to full by slab_gradient_*).
	if (MUSIC::mpi::size() > 1
	    && u.levelmin() < u.levelmax() + 1
	    && u.is_level_slab(u.levelmin())) {
		if (u.levelmin() != rh.levelmin()) {
			LOGERR("coarsen_density: slab levelmin grid but multilevel cascade requested "
			       "(u.levelmin=%u rh.levelmin=%u). Not supported by H.4.",
			       u.levelmin(), rh.levelmin());
			throw std::runtime_error("coarsen_density: slab + multilevel cascade unsupported");
		}
		return;
	}
#endif

	if (kspace)
	{
#ifdef USE_MPI
		if (MUSIC::mpi::size() > 1) {
			// Workers may not hold the GH skeleton (callers that wrap
			// GenerateDensityHierarchy rank-0-only leave workers with empty
			// m_pgrids). Bcast loop bounds from rank 0 so workers participate
			// in fft_coarsen_dist with NULL src/dst.
			int bounds[2] = {0, 1};
			if (is_root) {
				bounds[0] = (int)u.levelmin();
				bounds[1] = (int)rh.levelmin();
			}
			MPI_Bcast(bounds, 2, MPI_INT, 0, MUSIC::mpi::world());
			for (int i = bounds[0]; i >= bounds[1]; --i) {
				MeshvarBnd<real_t> *src = is_root ? u.get_grid(i)     : (MeshvarBnd<real_t> *)NULL;
				MeshvarBnd<real_t> *dst = is_root ? u.get_grid(i - 1) : (MeshvarBnd<real_t> *)NULL;
				fft_coarsen_dist(src, dst);
			}
		} else {
			const unsigned levelmin_TF = u.levelmin();
			for (int i = (int)levelmin_TF; i >= (int)rh.levelmin(); --i)
				fft_coarsen(*(u.get_grid(i)), *(u.get_grid(i - 1)));
		}
#else
		const unsigned levelmin_TF = u.levelmin();
		for (int i = (int)levelmin_TF; i >= (int)rh.levelmin(); --i)
			fft_coarsen(*(u.get_grid(i)), *(u.get_grid(i - 1)));
#endif
	}
	else
	{
		if (is_root) {
			const unsigned levelmin_TF = u.levelmin();
			for (int i = (int)levelmin_TF; i >= (int)rh.levelmin(); --i)
				mg_straight().restrict(*(u.get_grid(i)), *(u.get_grid(i - 1)));
		}
	}

	if (is_root) {
		for (unsigned i = 1; i <= rh.levelmax(); ++i)
		{
			if (rh.offset(i, 0) != u.get_grid(i)->offset(0) || rh.offset(i, 1) != u.get_grid(i)->offset(1) || rh.offset(i, 2) != u.get_grid(i)->offset(2) || rh.size(i, 0) != u.get_grid(i)->size(0) || rh.size(i, 1) != u.get_grid(i)->size(1) || rh.size(i, 2) != u.get_grid(i)->size(2))
			{
				u.cut_patch(i, rh.offset_abs(i, 0), rh.offset_abs(i, 1), rh.offset_abs(i, 2),
							rh.size(i, 0), rh.size(i, 1), rh.size(i, 2));
			}
		}

		for (int i = rh.levelmax(); i > 0; --i)
			mg_straight().restrict(*(u.get_grid(i)), *(u.get_grid(i - 1)));
	}
}
