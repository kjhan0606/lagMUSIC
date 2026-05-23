/*
 
 main.cc - This file is part of MUSIC -
 a code to generate multi-scale initial conditions 
 for cosmological simulations 
 
 Copyright (C) 2010  Oliver Hahn
 
*/


#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <math.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_integration.h>

#if defined(CMAKE_BUILD)
	extern "C"
	{
			extern const char* GIT_TAG;
			extern const char* GIT_REV;
			extern const char* GIT_BRANCH;
	}
#endif

#include "general.hh"
#include "defaults.hh"
#include "output.hh"

#include "config_file.hh"

#include "poisson.hh"
#include "mg_solver.hh"
#include "fd_schemes.hh"
#include "random.hh"
#include "densities.hh"

#include "convolution_kernel.hh"
#include "cosmology.hh"
#include "transfer_function.hh"
#include "mpi_helper.hh"
#include "mesh_distributed.hh"
#include "mpi_fft.hh"
#include "mpi_poisson.hh"

#include <vector>
#include <cstdlib>
#include <string>

#ifdef USE_MPI
#include <fftw3-mpi.h>
#endif

#define THE_CODE_NAME "music!"
#define THE_CODE_VERSION "1.53"


namespace music
{

	struct framework
	{
		transfer_function *the_transfer_function;
		//poisson_solver *the_poisson_solver;
		config_file *the_config_file;
		refinement_hierarchy *the_refinement_hierarchy;
	};
	
}
 
//... declare static class members here
transfer_function *TransferFunction_real::ptf_ = NULL;
transfer_function *TransferFunction_k::ptf_ = NULL;
tf_type TransferFunction_k::type_;
tf_type TransferFunction_real::type_;
real_t TransferFunction_real::nspec_ = -1.0;
real_t TransferFunction_k::nspec_ = -1.0;


//... prototypes for routines used in main driver routine
void splash(void);
void modify_grid_for_TF( const refinement_hierarchy& rh_full, refinement_hierarchy& rh_TF, config_file& cf );
void print_hierarchy_stats( config_file& cf, const refinement_hierarchy& rh );
void store_grid_structure( config_file& cf, const refinement_hierarchy& rh );
double compute_finest_mean( grid_hierarchy& u );
double compute_finest_sigma( grid_hierarchy& u );


void splash(void)
{
	
	std::cout 
		<< "\n    __    __     __  __     ______     __     ______      \n"
		<< "   /\\ \"-./  \\   /\\ \\/\\ \\   /\\  ___\\   /\\ \\   /\\  ___\\  \n"   
		<< "   \\ \\ \\-./\\ \\  \\ \\ \\_\\ \\  \\ \\___  \\  \\ \\ \\  \\ \\ \\____ \n"
		<< "    \\ \\_\\ \\ \\_\\  \\ \\_____\\  \\/\\_____\\  \\ \\_\\  \\ \\_____\\ \n"
		<< "     \\/_/  \\/_/   \\/_____/   \\/_____/   \\/_/   \\/_____/ \n\n"
		<< "                            this is " << THE_CODE_NAME << " version " << THE_CODE_VERSION << "\n\n";

	#if defined(CMAKE_BUILD)
		LOGINFO("Version built from git rev.: %s, tag: %s, branch: %s", GIT_REV, GIT_TAG, GIT_BRANCH);
	#endif
	#if defined(SINGLE_PRECISION)
		LOGINFO("Version was compiled for single precision.");
	#else
		LOGINFO("Version was compiled for double precision.");
	#endif
	std::cout << "\n\n";
}

void modify_grid_for_TF( const refinement_hierarchy& rh_full, refinement_hierarchy& rh_TF, config_file& cf )
{
	unsigned lbase, lbaseTF, lmax, overlap;
	
	lbase				= cf.getValue<unsigned>( "setup", "levelmin" );
	lmax				= cf.getValue<unsigned>( "setup", "levelmax" );
	lbaseTF				= cf.getValueSafe<unsigned>( "setup", "levelmin_TF", lbase );
	overlap				= cf.getValueSafe<unsigned>( "setup", "overlap", 4 );
	rh_TF = rh_full;
	
	unsigned pad = overlap;
	
	for( unsigned i=lbase+1; i<=lmax; ++i )
	{
		int x0[3], lx[3], lxmax = 0;
		
		for( int j=0; j<3; ++j )
		{
			lx[j] = rh_TF.size(i,j)+2*pad;
			x0[j] = rh_TF.offset_abs(i,j)-pad;
			
			if( lx[j] > lxmax )
				lxmax = lx[j];
		}
		
		//... make sure that grids are divisible by 4 for convolution.
		lxmax += lxmax%4;
		
		
		for( int j=0; j<3; ++j )
		{
			double dl = 0.5*((double)(lxmax-lx[j]));
			int add_left = (int)ceil(dl);
			
			lx[j] = lxmax;
			x0[j] -= add_left;
			x0[j] += x0[j]%2;
		}
		
		rh_TF.adjust_level(i, lx[0], lx[1], lx[2], x0[0], x0[1], x0[2] );
	}
	
	if( lbaseTF > lbase )
	{
		std::cout << " - Will use levelmin = " << lbaseTF << " to compute density field...\n";
	
		for( unsigned i=lbase; i<=lbaseTF; ++i )
		{
			unsigned nfull = (unsigned)pow(2,i);
			rh_TF.adjust_level(i, nfull, nfull, nfull, 0, 0, 0);
		}
	}
	
	
		
}

void print_hierarchy_stats( config_file& cf, const refinement_hierarchy& rh )
{
	double omegam = cf.getValue<double>("cosmology","Omega_m");
	double omegab = cf.getValue<double>("cosmology","Omega_b");
	bool bbaryons = cf.getValue<bool>("setup","baryons");
	double boxlength = cf.getValue<double>("setup","boxlength");
	
	unsigned levelmin = rh.levelmin();
	double dx = boxlength/(double)(1<<levelmin), dx3=dx*dx*dx;
	double rhom = 2.77519737e11; // h-1 M_o / (h-1 Mpc)**3
	double cmass, bmass(0.0), mtotgrid;
	if( bbaryons )
	{
		cmass = (omegam-omegab)*rhom*dx3;
		bmass = omegab*rhom*dx3;
	}else
		cmass = omegam*rhom*dx3;
	
	std::cout << "-------------------------------------------------------------\n";
	
	if( rh.get_shift(0)!=0||rh.get_shift(1)!=0||rh.get_shift(2)!=0 )
		std::cout << " - Domain will be shifted by (" << rh.get_shift(0) << ", " << rh.get_shift(1) << ", " << rh.get_shift(2) << ")\n" << std::endl;
	
	std::cout << " - Grid structure:\n";
	
	 
	
	for( unsigned ilevel=rh.levelmin(); ilevel<=rh.levelmax(); ++ilevel )
	{
		double rfac = 1.0/(1<<(ilevel-rh.levelmin())), rfac3=rfac*rfac*rfac;
		
		mtotgrid = omegam*rhom*dx3*rfac3*rh.size(ilevel, 0)*rh.size(ilevel, 1)*rh.size(ilevel, 2);
		std::cout 
		<< "     Level " << std::setw(3) << ilevel << " :   offset = (" << std::setw(5) << rh.offset(ilevel,0) << ", " << std::setw(5) << rh.offset(ilevel,1) << ", " << std::setw(5) << rh.offset(ilevel,2) << ")\n"
		<< "                     size = (" << std::setw(5) << rh.size(ilevel,0) << ", " << std::setw(5) << rh.size(ilevel,1) << ", " << std::setw(5) << rh.size(ilevel,2) << ")\n";
		
		if( ilevel == rh.levelmax() )
		{
			std::cout << "-------------------------------------------------------------\n";
			std::cout << " - Finest level :\n";

			if( dx*rfac > 0.1 )	
			  std::cout << "                   extent =  " << dx*rfac*rh.size(ilevel,0) << " x " << dx*rfac*rh.size(ilevel,1) << " x " << dx*rfac * rh.size(ilevel,2) << " h-3 Mpc**3\n";
			else if( dx*rfac > 1e-4 )
			  std::cout << "                   extent =  " << dx*rfac*1000.0*rh.size(ilevel,0) << " x " << dx*rfac*1000.0*rh.size(ilevel,1) << " x " << dx*rfac*1000.0*rh.size(ilevel,2) << " h-3 kpc**3\n";
			else
			  std::cout << "                   extent =  " << dx*rfac*1.e6*rh.size(ilevel,0) << " x " << dx*rfac*1.e6*rh.size(ilevel,1) << " x " << dx*rfac*1.e6 * rh.size(ilevel,2) << " h-3 pc**3\n";

			std::cout << "                 mtotgrid =  " << mtotgrid << " h-1 M_o\n";
			std::cout << "            particle mass =  " << cmass*rfac3 << " h-1 M_o\n";
			if( bbaryons )
				std::cout << "         baryon mass/cell =  " << bmass*rfac3 << " h-1 M_o\n";
			if( dx*rfac > 0.1 )
				std::cout << "                       dx =  " << dx*rfac << " h-1 Mpc\n";
			else if( dx*rfac > 1e-4 )
				std::cout << "                       dx =  " << dx*rfac*1000.0 << " h-1 kpc\n";
			else
				std::cout << "                       dx =  " << dx*rfac*1.e6 << " h-1 pc\n";
		}
		
	}
	std::cout << "-------------------------------------------------------------\n";
}


void store_grid_structure( config_file& cf, const refinement_hierarchy& rh )
{
	char str1[128], str2[128];
	for( unsigned i=rh.levelmin(); i<=rh.levelmax(); ++i )
	{
		for( int j=0; j<3; ++j )
		{
			sprintf(str1,"offset(%d,%d)",i,j);	
			sprintf(str2,"%d",rh.offset(i,j));
			cf.insertValue("setup",str1,str2);

			sprintf(str1,"size(%d,%d)",i,j);	
			sprintf(str2,"%ld",rh.size(i,j));
			cf.insertValue("setup",str1,str2);
			
		}		
	}
}

double compute_finest_mean( grid_hierarchy& u )
{
	// D.3.3/D.4: at multi-box finest level iterate per-box (excludes gap
	// cells); at single-box use union with is_refined gate (existing path).
	unsigned L = u.levelmax();
	double sum = 0.0;
	size_t count = 0;
	const size_t nb = u.num_boxes(L);

	if( nb > 1 ){
		for( size_t b=0; b<nb; ++b ){
			meshvar_bnd * pm = u.get_grid(L, b);
			int nx=pm->size(0), ny=pm->size(1), nz=pm->size(2);
			for( int ix=0; ix<nx; ++ix )
				for( int iy=0; iy<ny; ++iy )
					for( int iz=0; iz<nz; ++iz ){
						sum += (*pm)(ix,iy,iz);
						++count;
					}
		}
	} else {
		meshvar_bnd * pm = u.get_grid(L);
		for( int ix = 0; ix < (int)pm->size(0); ++ix )
			for( int iy = 0; iy < (int)pm->size(1); ++iy )
				for( int iz = 0; iz < (int)pm->size(2); ++iz )
					if( ! u.is_refined(L,ix,iy,iz) ){
						sum += (*pm)(ix,iy,iz);
						++count;
					}
	}
	if( count ) sum /= count;
	return sum;
}

double compute_finest_sigma( grid_hierarchy& u )
{
	// D.3.3/D.4: per-box at multi-box finest level.
	unsigned L = u.levelmax();
	double sum = 0.0, sum2 = 0.0;
	size_t N = 0;
	const size_t nb = u.num_boxes(L);

	if( nb > 1 ){
		for( size_t b=0; b<nb; ++b ){
			meshvar_bnd * pm = u.get_grid(L, b);
			int nx=pm->size(0), ny=pm->size(1), nz=pm->size(2);
			for( int ix=0; ix<nx; ++ix )
				for( int iy=0; iy<ny; ++iy )
					for( int iz=0; iz<nz; ++iz ){
						double v = (*pm)(ix,iy,iz);
						sum  += v;
						sum2 += v*v;
						++N;
					}
		}
	} else {
		meshvar_bnd * pm = u.get_grid(L);
		int nx=pm->size(0), ny=pm->size(1), nz=pm->size(2);
		for( int ix=0; ix<nx; ++ix )
			for( int iy=0; iy<ny; ++iy )
				for( int iz=0; iz<nz; ++iz ){
					double v = (*pm)(ix,iy,iz);
					sum  += v;
					sum2 += v*v;
				}
		N = (size_t)nx*(size_t)ny*(size_t)nz;
	}
	if( N==0 ) return 0.0;
	sum /= N; sum2 /= N;
	return sqrt(sum2 - sum*sum);
}

double compute_finest_max( grid_hierarchy& u )
{
	// D.3.3/D.4: per-box at multi-box finest level.
	unsigned L = u.levelmax();
	double valmax = 0.0;
	const size_t nb = u.num_boxes(L);

	if( nb > 1 ){
		for( size_t b=0; b<nb; ++b ){
			meshvar_bnd * pm = u.get_grid(L, b);
			int nx=pm->size(0), ny=pm->size(1), nz=pm->size(2);
			for( int ix=0; ix<nx; ++ix )
				for( int iy=0; iy<ny; ++iy )
					for( int iz=0; iz<nz; ++iz ){
						double v = (*pm)(ix,iy,iz);
						if( fabs(v) > fabs(valmax) ) valmax = v;
					}
		}
	} else {
		meshvar_bnd * pm = u.get_grid(L);
		for( int ix = 0; ix < (int)pm->size(0); ++ix )
			for( int iy = 0; iy < (int)pm->size(1); ++iy )
				for( int iz = 0; iz < (int)pm->size(2); ++iz ){
					double v = (*pm)(ix,iy,iz);
					if( fabs(v) > fabs(valmax) ) valmax = v;
				}
	}
	return valmax;
}



/*****************************************************************************************************/
/*****************************************************************************************************/
/*****************************************************************************************************/

region_generator_plugin *the_region_generator;
RNG_plugin *the_random_number_generator;

//------------------------------------------------------------------------------
// Collective MPI-FFT smoke test: forward + inverse on an N^3 slab-distributed
// grid, reports max reconstruction error. Used to validate the slab Meshvar +
// FFTW3-MPI plan wiring independently of the science pipeline.
//------------------------------------------------------------------------------
static int run_mpi_fft_smoke( size_t N )
{
	using MUSIC::dist::make_slab_meshvar;
	using MUSIC::dist::slab_layout;
	using MUSIC::dist::compute_slab_layout;

	if( N == 0 ){
		if( MUSIC::mpi::is_root() )
			std::cerr << "[mpi-fft-smoke] N must be > 0\n";
		return 1;
	}

	Meshvar<fftw_real>* m = make_slab_meshvar<fftw_real>(N, N, N, /*fftw_inplace_pad=*/true);
	const size_t local_nx     = m->local_nx();
	const size_t local_x_off  = m->local_x_start();
	const size_t ny           = N;
	const size_t nz_logical   = N;
	const size_t nz_padded    = 2*(N/2+1);
	const double TWO_PI       = 2.0 * M_PI;

	// Fill with a deterministic separable mode so the FFT keeps a small support
	// and reconstruction error is dominated by fp roundoff rather than aliasing.
	for( size_t ix=0; ix<local_nx; ++ix ){
		size_t gix = local_x_off + ix;
		double sx = std::sin(TWO_PI*(double)gix/(double)N);
		for( size_t iy=0; iy<ny; ++iy ){
			double sy = std::sin(TWO_PI*(double)iy/(double)N);
			for( size_t iz=0; iz<nz_logical; ++iz ){
				double sz = std::sin(TWO_PI*(double)iz/(double)N);
				m->m_pdata[(ix*ny + iy)*nz_padded + iz] = (fftw_real)(sx*sy*sz);
			}
		}
	}

	// Snapshot the unpadded payload for later comparison.
	std::vector<fftw_real> orig(local_nx*ny*nz_logical);
	for( size_t ix=0; ix<local_nx; ++ix )
		for( size_t iy=0; iy<ny; ++iy )
			for( size_t iz=0; iz<nz_logical; ++iz )
				orig[(ix*ny + iy)*nz_logical + iz] =
					m->m_pdata[(ix*ny + iy)*nz_padded + iz];

#ifdef USE_MPI
	MUSIC::fft::fft_plan_t pf = MUSIC::fft::plan_r2c_3d_mpi(
		(ptrdiff_t)N, (ptrdiff_t)N, (ptrdiff_t)N, m->m_pdata);
	MUSIC::fft::fft_plan_t pi = MUSIC::fft::plan_c2r_3d_mpi(
		(ptrdiff_t)N, (ptrdiff_t)N, (ptrdiff_t)N, m->m_pdata);
#else
	MUSIC::fft::fft_plan_t pf = MUSIC::fft::plan_r2c_3d_serial(
		(int)N, (int)N, (int)N, m->m_pdata);
	MUSIC::fft::fft_plan_t pi = MUSIC::fft::plan_c2r_3d_serial(
		(int)N, (int)N, (int)N, m->m_pdata);
#endif
	MUSIC::fft::execute(pf);
	MUSIC::fft::execute(pi);

	const double norm = 1.0 / ((double)N*(double)N*(double)N);
	double local_max_err = 0.0, local_max_val = 0.0;
	for( size_t ix=0; ix<local_nx; ++ix )
		for( size_t iy=0; iy<ny; ++iy )
			for( size_t iz=0; iz<nz_logical; ++iz ){
				double v = (double)m->m_pdata[(ix*ny + iy)*nz_padded + iz] * norm;
				double o = (double)orig[(ix*ny + iy)*nz_logical + iz];
				double e = std::fabs(v - o);
				if( e > local_max_err ) local_max_err = e;
				if( std::fabs(o) > local_max_val ) local_max_val = std::fabs(o);
			}

	MUSIC::fft::destroy(pf);
	MUSIC::fft::destroy(pi);
	delete m;

	double global_max_err = local_max_err;
	double global_max_val = local_max_val;
#ifdef USE_MPI
	MPI_Allreduce(&local_max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, MUSIC::mpi::world());
	MPI_Allreduce(&local_max_val, &global_max_val, 1, MPI_DOUBLE, MPI_MAX, MUSIC::mpi::world());
#endif

	if( MUSIC::mpi::is_root() ){
		double rel = global_max_err / std::max(global_max_val, 1e-300);
		std::cerr << "[mpi-fft-smoke] N=" << N
		          << " ranks=" << MUSIC::mpi::size()
		          << " threads=" << omp_get_max_threads()
		          << " precision=" <<
#ifdef SINGLE_PRECISION
		             "single"
#else
		             "double"
#endif
		          << " max|orig|="          << global_max_val
		          << " max|recon-orig|="    << global_max_err
		          << " rel="                << rel
		          << "\n";
	}
	return 0;
}

//------------------------------------------------------------------------------
// Convolution-pipeline smoke test: replays the perform_mpi() forward-multiply-
// inverse sequence with Tk == 1 on a slab grid filled with a separable sine
// (DC == 0, single Fourier mode). The output should equal the input scaled by
// (N * fftnorm) on every cell — we check that ratio uniformly across the
// distributed buffer, and report a global checksum for cross-rank-count
// determinism.
//------------------------------------------------------------------------------
static int run_mpi_conv_smoke( size_t N )
{
	using MUSIC::dist::make_slab_meshvar;

	if( N == 0 ){
		if( MUSIC::mpi::is_root() )
			std::cerr << "[mpi-conv-smoke] N must be > 0\n";
		return 1;
	}

	Meshvar<fftw_real>* m = make_slab_meshvar<fftw_real>(N, N, N, /*fftw_inplace_pad=*/true);
	const size_t local_nx     = m->local_nx();
	const size_t local_x_off  = m->local_x_start();
	const size_t ny           = N;
	const size_t nz_logical   = N;
	const size_t nz_complex   = N/2 + 1;
	const size_t nz_padded    = 2*nz_complex;
	const double TWO_PI       = 2.0 * M_PI;
	const double lx = 1.0, ly = 1.0, lz = 1.0;

	// sin*sin*sin: pure mode at k=(1,1,1) + its symmetric partners, DC == 0.
	for( size_t ix=0; ix<local_nx; ++ix ){
		size_t gix = local_x_off + ix;
		double sx = std::sin(TWO_PI*(double)gix/(double)N);
		for( size_t iy=0; iy<ny; ++iy ){
			double sy = std::sin(TWO_PI*(double)iy/(double)N);
			for( size_t iz=0; iz<nz_logical; ++iz ){
				double sz = std::sin(TWO_PI*(double)iz/(double)N);
				m->m_pdata[(ix*ny + iy)*nz_padded + iz] = (fftw_real)(sx*sy*sz);
			}
		}
	}

	// Snapshot the input for ratio comparison post-roundtrip.
	std::vector<fftw_real> orig(local_nx*ny*nz_logical);
	for( size_t ix=0; ix<local_nx; ++ix )
		for( size_t iy=0; iy<ny; ++iy )
			for( size_t iz=0; iz<nz_logical; ++iz )
				orig[(ix*ny + iy)*nz_logical + iz] = m->m_pdata[(ix*ny + iy)*nz_padded + iz];

	const double fftnormp = 1.0 / std::sqrt((double)N*(double)N*(double)N);
	const double fftnorm  = std::pow(TWO_PI, 1.5) / std::sqrt(lx*ly*lz) * fftnormp;

	fftw_real    *data  = m->m_pdata;
	fftw_complex *cdata = reinterpret_cast<fftw_complex*>(data);

#ifdef USE_MPI
	MUSIC::fft::fft_plan_t pf = MUSIC::fft::plan_r2c_3d_mpi(
		(ptrdiff_t)N, (ptrdiff_t)N, (ptrdiff_t)N, data);
	MUSIC::fft::fft_plan_t pi = MUSIC::fft::plan_c2r_3d_mpi(
		(ptrdiff_t)N, (ptrdiff_t)N, (ptrdiff_t)N, data);
#else
	MUSIC::fft::fft_plan_t pf = MUSIC::fft::plan_r2c_3d_serial(
		(int)N, (int)N, (int)N, data);
	MUSIC::fft::fft_plan_t pi = MUSIC::fft::plan_c2r_3d_serial(
		(int)N, (int)N, (int)N, data);
#endif
	MUSIC::fft::execute(pf);

	// K-space Tk == 1 multiply (matches perform_mpi math, dstag == 0).
	for( size_t ix=0; ix<local_nx; ++ix ){
		for( size_t iy=0; iy<ny; ++iy ){
			for( size_t k=0; k<nz_complex; ++k ){
				size_t ii = (ix*ny + iy)*nz_complex + k;
				double re = (double)RE(cdata[ii]) * fftnorm;
				double im = (double)IM(cdata[ii]) * fftnorm;
				RE(cdata[ii]) = (fftw_real)re;
				IM(cdata[ii]) = (fftw_real)im;
			}
		}
	}
	// Zero DC on the owning rank (input already had DC == 0; this just mirrors perform_mpi).
	if( local_x_off == 0 && local_nx > 0 ){
		RE(cdata[0]) = 0.0;
		IM(cdata[0]) = 0.0;
	}

	MUSIC::fft::execute(pi);
	MUSIC::fft::destroy(pf);
	MUSIC::fft::destroy(pi);

	// Expected per-cell scale: N^3 (unnormalized FFTW forward+inverse) * fftnorm.
	const double expected_scale = (double)N*(double)N*(double)N * fftnorm;

	double local_max_dev = 0.0;
	double local_sumsq   = 0.0;
	double local_max_in  = 0.0;
	for( size_t ix=0; ix<local_nx; ++ix )
		for( size_t iy=0; iy<ny; ++iy )
			for( size_t iz=0; iz<nz_logical; ++iz ){
				double out = (double)m->m_pdata[(ix*ny + iy)*nz_padded + iz];
				double in  = (double)orig[(ix*ny + iy)*nz_logical + iz];
				double dev = std::fabs(out - in*expected_scale);
				if( dev > local_max_dev ) local_max_dev = dev;
				if( std::fabs(in) > local_max_in ) local_max_in = std::fabs(in);
				local_sumsq += out*out;
			}

	double global_max_dev = local_max_dev;
	double global_sumsq   = local_sumsq;
	double global_max_in  = local_max_in;
#ifdef USE_MPI
	MPI_Allreduce(&local_max_dev, &global_max_dev, 1, MPI_DOUBLE, MPI_MAX, MUSIC::mpi::world());
	MPI_Allreduce(&local_sumsq,   &global_sumsq,   1, MPI_DOUBLE, MPI_SUM, MUSIC::mpi::world());
	MPI_Allreduce(&local_max_in,  &global_max_in,  1, MPI_DOUBLE, MPI_MAX, MUSIC::mpi::world());
#endif

	delete m;

	if( MUSIC::mpi::is_root() ){
		double rel = global_max_dev / std::max(global_max_in * expected_scale, 1e-300);
		std::cerr.setf(std::ios::scientific);
		std::cerr.precision(12);
		std::cerr << "[mpi-conv-smoke] N=" << N
		          << " ranks=" << MUSIC::mpi::size()
		          << " threads=" << omp_get_max_threads()
		          << " precision=" <<
#ifdef SINGLE_PRECISION
		             "single"
#else
		             "double"
#endif
		          << " expected_scale=" << expected_scale
		          << " max|out - in*scale|=" << global_max_dev
		          << " rel="                 << rel
		          << " sum(out^2)="          << global_sumsq
		          << "\n";
	}
	return 0;
}

//------------------------------------------------------------------------------
// perform_dist smoke test: drives the full scatter / perform_mpi / gather
// pipeline through an actual ksampled kernel built from a real config file.
// All ranks collectively call convolution::perform_dist; root owns a
// DensityGrid<real_t> sized to (N,N,N) with the in-place r2c padding and
// pre-filled with a deterministic separable mode (so RNG state isn't a
// confound). After the call, root prints a reproducible checksum that must
// match bit-for-bit across rank counts.
//
// The config file just needs the keys tf_kernel_k reads (setup/boxlength,
// cosmology/nspec + pnorm, etc.) — ics_example.conf works as-is once we
// inject pnorm.
//------------------------------------------------------------------------------
static int run_mpi_perform_dist_smoke( const char* cfgfile, size_t N )
{
	if( cfgfile == NULL || N == 0 ){
		if( MUSIC::mpi::is_root() )
			std::cerr << "[mpi-perform-dist-smoke] usage: --mpi-perform-dist-smoke <conf> <N>\n";
		return 1;
	}

	config_file cf(cfgfile);
	// pnorm is normally injected by main() after the cosmology calc. For the
	// smoke we just pick a fixed value; any positive number suffices for a
	// determinism test.
	cf.insertValue("cosmology","pnorm","1.0");
	// Some kernel paths consult periodic_TF / deconvolve / fft_fine; defaults
	// are safe but make them explicit so the test config is self-contained.
	if( !cf.containsKey("setup","periodic_TF") ) cf.insertValue("setup","periodic_TF","yes");
	if( !cf.containsKey("setup","deconvolve") )  cf.insertValue("setup","deconvolve","yes");
	if( !cf.containsKey("poisson","fft_fine") )  cf.insertValue("poisson","fft_fine","yes");
	// The region generator and some other helpers consult output/format; the
	// smoke does not actually emit any output but the key must be present.
	if( !cf.containsKey("output","format") )    cf.insertValue("output","format","generic");
	if( !cf.containsKey("output","filename") )  cf.insertValue("output","filename","/dev/null");

	transfer_function* ptf = select_transfer_function_plugin(cf);
	// refinement_hierarchy ctor dereferences the global region generator; set
	// it up the same way main() does before constructing rh_Poisson.
	the_region_generator = select_region_generator_plugin(cf);
	refinement_hierarchy refh(cf);

	convolution::kernel_creator* kc =
#ifdef SINGLE_PRECISION
		convolution::get_kernel_map()["tf_kernel_k_float"];
#else
		convolution::get_kernel_map()["tf_kernel_k_double"];
#endif
	if( kc == NULL ){
		if( MUSIC::mpi::is_root() )
			std::cerr << "[mpi-perform-dist-smoke] kernel creator not found\n";
		return 1;
	}
	convolution::kernel* the_kernel = kc->create(cf, ptf, refh, total);

	// fetch_kernel populates cparam_ with the (nx,ny,nz) that this level will
	// convolve over — this comes from refh, not from our N argument. We honour
	// that to keep the test config-consistent; the user-supplied N is only
	// used to pick which level to convolve (smallest level whose extent <= N
	// fits in memory for the test).
	unsigned lmin = cf.getValue<unsigned>("setup","levelmin");
	unsigned lminTF = cf.getValueSafe<unsigned>("setup","levelmin_TF", lmin);
	(void)N;  // accepted for CLI consistency; the actual grid is set by refh

	convolution::kernel* fetched = the_kernel->fetch_kernel((int)lminTF, false);
	const size_t gnx = (size_t)fetched->cparam_.nx;
	const size_t gny = (size_t)fetched->cparam_.ny;
	const size_t gnz = (size_t)fetched->cparam_.nz;

	const size_t nz_padded = 2*(gnz/2+1);
	const size_t total_padded = gnx * gny * nz_padded;

	// Root allocates the full padded buffer; workers pass NULL.
	std::vector<real_t> root_buf;
	real_t* root_ptr = NULL;
	if( MUSIC::mpi::is_root() ){
		root_buf.assign(total_padded, (real_t)0);
		const double TWO_PI = 2.0 * M_PI;
		for( size_t ix=0; ix<gnx; ++ix ){
			double sx = std::sin(TWO_PI*(double)ix/(double)gnx);
			for( size_t iy=0; iy<gny; ++iy ){
				double sy = std::sin(TWO_PI*(double)iy/(double)gny);
				for( size_t iz=0; iz<gnz; ++iz ){
					double sz = std::sin(TWO_PI*(double)iz/(double)gnz);
					root_buf[(ix*gny + iy)*nz_padded + iz] = (real_t)(sx*sy*sz);
				}
			}
		}
		root_ptr = &root_buf[0];
	}

	convolution::perform_dist<real_t>( fetched, root_ptr, gnx, gny, gnz,
	                                   /*shift=*/false, /*fix=*/false, /*flip=*/false );

	// Reproducible checksum (sum and sum-of-squares of the logical, non-padded
	// cells) — printed only on root.
	double sum = 0.0, sumsq = 0.0;
	double absmax = 0.0;
	if( MUSIC::mpi::is_root() ){
		for( size_t ix=0; ix<gnx; ++ix )
			for( size_t iy=0; iy<gny; ++iy )
				for( size_t iz=0; iz<gnz; ++iz ){
					double v = (double)root_buf[(ix*gny + iy)*nz_padded + iz];
					sum   += v;
					sumsq += v*v;
					if( std::fabs(v) > absmax ) absmax = std::fabs(v);
				}
		std::cerr.setf(std::ios::scientific);
		std::cerr.precision(12);
		std::cerr << "[mpi-perform-dist-smoke] gnx=" << gnx
		          << " gny=" << gny << " gnz=" << gnz
		          << " level=" << lminTF
		          << " ranks=" << MUSIC::mpi::size()
		          << " threads=" << omp_get_max_threads()
		          << " precision=" <<
#ifdef SINGLE_PRECISION
		             "single"
#else
		             "double"
#endif
		          << " sum="    << sum
		          << " sumsq="  << sumsq
		          << " max|v|=" << absmax
		          << "\n";
	}

	the_kernel->deallocate();
	delete the_kernel;
	delete ptf;
	return 0;
}

int main (int argc, const char * argv[])
{
#ifdef USE_MPI
	int mpi_thread_provided = 0;
	MPI_Init_thread(&argc, const_cast<char***>(&argv), MPI_THREAD_FUNNELED, &mpi_thread_provided);
	// silence stdout/clog on non-root ranks to keep console output clean.
	// stderr stays open so genuine errors (Error/FatalError) from any rank are visible.
	if( !MUSIC::mpi::is_root() ){
		std::cout.rdbuf(NULL);
		std::clog.rdbuf(NULL);
	}
#endif

	const unsigned nbnd = 4;

	unsigned lbase, lmax, lbaseTF;
	double   err = 1.0;

	//------------------------------------------------------------------------------
	//... parse command line options
	//------------------------------------------------------------------------------

	// Diagnostic modes: `MUSIC --mpi-fft-smoke <N>` runs a collective FFT
	// round-trip; `MUSIC --mpi-conv-smoke <N>` replays the convolution-style
	// forward-multiply-inverse pipeline with Tk == 1. All ranks must
	// participate, so these are dispatched after FFTW MPI init but before
	// the non-root early-return.
	enum { SMOKE_NONE, SMOKE_FFT, SMOKE_CONV, SMOKE_PERFORM_DIST } smoke_kind = SMOKE_NONE;
	bool   smoke_mode = false;
	size_t smoke_N    = 0;
	const char* smoke_cfg = NULL;
	if( argc >= 2 ){
		std::string a1 = argv[1];
		if( a1 == "--mpi-fft-smoke" )           smoke_kind = SMOKE_FFT;
		else if( a1 == "--mpi-conv-smoke" )     smoke_kind = SMOKE_CONV;
		else if( a1 == "--mpi-perform-dist-smoke" ) smoke_kind = SMOKE_PERFORM_DIST;
		smoke_mode = (smoke_kind != SMOKE_NONE);
	}
	if( smoke_mode ){
		if( smoke_kind == SMOKE_PERFORM_DIST ){
			if( argc < 4 ){
				if( MUSIC::mpi::is_root() )
					std::cerr << "Usage: " << argv[0] << " --mpi-perform-dist-smoke <conf> <N>\n";
#ifdef USE_MPI
				MPI_Finalize();
#endif
				return 1;
			}
			smoke_cfg = argv[2];
			smoke_N   = (size_t)std::atol(argv[3]);
		} else {
			if( argc < 3 ){
				if( MUSIC::mpi::is_root() )
					std::cerr << "Usage: " << argv[0] << " " << argv[1] << " <N>\n";
#ifdef USE_MPI
				MPI_Finalize();
#endif
				return 1;
			}
			smoke_N = (size_t)std::atol(argv[2]);
		}
	}

	if( !smoke_mode && MUSIC::mpi::is_root() ) splash();
	if( !smoke_mode && argc != 2 ){
		if( MUSIC::mpi::is_root() ){
			std::cout << " This version is compiled with the following plug-ins:\n";

			print_region_generator_plugins();
			print_transfer_function_plugins();
			print_RNG_plugins();
			print_output_plugins();

			std::cerr << "\n In order to run, you need to specify a parameter file!\n\n";
		}
#ifdef USE_MPI
		MPI_Finalize();
#endif
		exit(0);
	}

	//------------------------------------------------------------------------------
	//... open log file (rank 0 only for now)
	//------------------------------------------------------------------------------

	char logfname[128];
	time_t ltime=time(NULL);
	if( !smoke_mode ){
		sprintf(logfname,"%s_log.txt",argv[1]);
		if( MUSIC::mpi::is_root() ) MUSIC::log::setOutput(logfname);
		LOGINFO("Opening log file \'%s\'.",logfname);
		LOGUSER("Running %s, version %s",THE_CODE_NAME,THE_CODE_VERSION);
		LOGUSER("Log is for run started %s",asctime( localtime(&ltime) ));
	}
	
#ifdef FFTW3
	LOGUSER("Code was compiled using FFTW version 3.x");
#else
	LOGUSER("Code was compiled using FFTW version 2.x");
#endif
	
#ifdef SINGLETHREAD_FFTW
	LOGUSER("Code was compiled for single-threaded FFTW");
#else
	LOGUSER("Code was compiled for multi-threaded FFTW");
	LOGUSER("Running with a maximum of %d OpenMP threads", omp_get_max_threads() );
#endif
	
#ifdef SINGLE_PRECISION
	LOGUSER("Code was compiled for single precision.");
#else
	LOGUSER("Code was compiled for double precision.");
#endif

	//------------------------------------------------------------------------------
	//... smoke-mode short-circuit: init threaded + MPI FFTW, run the collective
	//... round-trip test, tear down FFTW, finalize MPI, exit. All ranks
	//... participate; no config file is read.
	//------------------------------------------------------------------------------
	if( smoke_mode ){
#if not defined(SINGLETHREAD_FFTW) && defined(FFTW3)
	#ifdef SINGLE_PRECISION
		fftwf_init_threads();
		fftwf_plan_with_nthreads(omp_get_max_threads());
	#else
		fftw_init_threads();
		fftw_plan_with_nthreads(omp_get_max_threads());
	#endif
#endif
#ifdef USE_MPI
	#ifdef SINGLE_PRECISION
		fftwf_mpi_init();
	#else
		fftw_mpi_init();
	#endif
#endif
		int rc;
		if( smoke_kind == SMOKE_PERFORM_DIST )
			rc = run_mpi_perform_dist_smoke(smoke_cfg, smoke_N);
		else if( smoke_kind == SMOKE_CONV )
			rc = run_mpi_conv_smoke(smoke_N);
		else
			rc = run_mpi_fft_smoke(smoke_N);
#ifdef USE_MPI
	#ifdef SINGLE_PRECISION
		fftwf_mpi_cleanup();
	#else
		fftw_mpi_cleanup();
	#endif
#endif
#if not defined(SINGLETHREAD_FFTW) && defined(FFTW3)
	#ifdef SINGLE_PRECISION
		fftwf_cleanup_threads();
	#else
		fftw_cleanup_threads();
	#endif
#endif
#ifdef USE_MPI
		MPI_Finalize();
#endif
		return rc;
	}

	//------------------------------------------------------------------------------
	//... read and interpret config file
	//------------------------------------------------------------------------------
	config_file cf(argv[1]);
	std::string tfname,randfname,temp;
	bool force_shift(false);
	double boxlength;
	
	//------------------------------------------------------------------------------
	//... initialize some parameters about grid set-up
	//------------------------------------------------------------------------------
	
	boxlength           = cf.getValue<double>( "setup", "boxlength" );
	lbase				= cf.getValue<unsigned>( "setup", "levelmin" );
	lmax				= cf.getValue<unsigned>( "setup", "levelmax" );
	lbaseTF				= cf.getValueSafe<unsigned>( "setup", "levelmin_TF", lbase );
	
	if( lbase == lmax && !force_shift )
		cf.insertValue("setup","no_shift","yes");
	
	if( lbaseTF < lbase )
	{
		std::cout << " - WARNING: levelminTF < levelmin. This is not good!\n"
				  << "            I will set levelminTF = levelmin.\n";
		
		LOGUSER("levelminTF < levelmin. set levelminTF = levelmin.");
		
		lbaseTF = lbase;
		cf.insertValue("setup","levelmin_TF",cf.getValue<std::string>("setup","levelmin"));
	}
	
    // .. determine if spectral sampling should be used
    if( !cf.containsKey( "setup", "kspace_TF" ))
        cf.insertValue( "setup", "kspace_TF", "yes");
    
    bool bspectral_sampling = cf.getValue<bool>( "setup", "kspace_TF" );
    
	if( bspectral_sampling )
	  LOGINFO("Using k-space sampled transfer functions...");
	else
	  LOGINFO("Using real space sampled transfer functions...");
		
	//------------------------------------------------------------------------------
	//... initialize multithread FFTW
	//------------------------------------------------------------------------------
	
#if not defined(SINGLETHREAD_FFTW)
#ifdef FFTW3
	#ifdef SINGLE_PRECISION
	fftwf_init_threads();
	fftwf_plan_with_nthreads(omp_get_max_threads());
	#else
	fftw_init_threads();
	fftw_plan_with_nthreads(omp_get_max_threads());
	#endif
#else
	fftw_threads_init();
#endif
#endif

#ifdef USE_MPI
	#ifdef SINGLE_PRECISION
	fftwf_mpi_init();
	#else
	fftw_mpi_init();
	#endif

	// SPMD-light: all ranks now participate in main flow. They reach every
	// collective convolution::perform_dist() call inside GenerateDensityHierarchy.
	// Heavy serial work (Poisson solve, output writes) is gated behind
	// MUSIC::mpi::is_root() — workers no-op those sections.
#endif
	
	//------------------------------------------------------------------------------
	//... initialize cosmology
	//------------------------------------------------------------------------------
	bool 
	  do_baryons	= cf.getValue<bool>("setup","baryons"),
	  do_2LPT	= cf.getValueSafe<bool>("setup","use_2LPT",false),
	  do_LLA       	= cf.getValueSafe<bool>("setup","use_LLA",false);
	
	transfer_function_plugin *the_transfer_function_plugin
		= select_transfer_function_plugin( cf );
    
	cosmology cosmo( cf );
	
	std::cout << " - starting at a=" << cosmo.astart << std::endl;
	
	CosmoCalc ccalc(cosmo,the_transfer_function_plugin);
	cosmo.pnorm	= ccalc.ComputePNorm( 2.0*M_PI/boxlength );
	cosmo.dplus	= ccalc.CalcGrowthFactor( cosmo.astart )/ccalc.CalcGrowthFactor( 1.0 );
	cosmo.vfact = ccalc.CalcVFact( cosmo.astart );
	
	if( !the_transfer_function_plugin->tf_has_total0() )
	        cosmo.pnorm *= cosmo.dplus*cosmo.dplus;
	
	//... directly use the normalisation via a parameter rather than the calculated one
	cosmo.pnorm = cf.getValueSafe<double>("setup","force_pnorm",cosmo.pnorm);
	

	double vfac2lpt = 1.0;
	
	if( the_transfer_function_plugin->tf_velocity_units() && do_baryons )
	{
		vfac2lpt = cosmo.vfact; // if the velocities are in velocity units, we need to divide by vfact for the 2lPT term
		cosmo.vfact = 1.0;
	}

	
	//
	{
		char tmpstr[128];
		sprintf(tmpstr,"%.12g",cosmo.pnorm);
		cf.insertValue("cosmology","pnorm",tmpstr);
		sprintf(tmpstr,"%.12g",cosmo.dplus);
		cf.insertValue("cosmology","dplus",tmpstr);
		sprintf(tmpstr,"%.12g",cosmo.vfact);
		cf.insertValue("cosmology","vfact",tmpstr);
		
	}
	
    the_region_generator = select_region_generator_plugin( cf );
  
    the_random_number_generator = select_RNG_plugin( cf );
	//------------------------------------------------------------------------------
	//... determine run parameters
	//------------------------------------------------------------------------------
	
	if( !the_transfer_function_plugin->tf_is_distinct() && do_baryons )
		std::cout	<< " - WARNING: The selected transfer function does not support\n"
		<< "            distinct amplitudes for baryon and DM fields!\n"
		<< "            Perturbation amplitudes will be identical!" << std::endl;
	

	
	
	//------------------------------------------------------------------------------
	//... determine the refinement hierarchy
	//------------------------------------------------------------------------------
	
	refinement_hierarchy rh_Poisson( cf );
	store_grid_structure(cf, rh_Poisson);
	//rh_Poisson.output();
	print_hierarchy_stats( cf, rh_Poisson );
	
	refinement_hierarchy rh_TF( rh_Poisson );
	modify_grid_for_TF( rh_Poisson, rh_TF, cf );
	//rh_TF.output();

	LOGUSER("Grid structure for Poisson solver:");
	rh_Poisson.output_log();
	LOGUSER("Grid structure for density convolution:");
	rh_TF.output_log();
	
	//------------------------------------------------------------------------------
	//... initialize the output plug-in
	//------------------------------------------------------------------------------
	std::string outformat, outfname;
	outformat			= cf.getValue<std::string>( "output", "format" );
	outfname			= cf.getValue<std::string>( "output", "filename" );
	// Plugins that override finalize_collective() (gadget2 family) instantiate
	// on ALL ranks so workers can participate in the parallel multi-file write.
	// Everyone else stays rank-0-only — workers leave the_output_plugin NULL
	// and skip all the_output_plugin->... calls via is_root() gates.
	const bool plugin_all_ranks = (outformat == "gadget2" || outformat == "gadget2_double");
	output_plugin *the_output_plugin = (plugin_all_ranks || MUSIC::mpi::is_root())
	                                   ? select_output_plugin( cf ) : NULL;
	
	//------------------------------------------------------------------------------
	//... initialize the random numbers
	//------------------------------------------------------------------------------
	std::cout << "=============================================================\n";
	std::cout << "   GENERATING WHITE NOISE\n";
	std::cout << "-------------------------------------------------------------\n";
	LOGUSER("Computing white noise...");
	rand_gen rand( cf, rh_TF, the_transfer_function_plugin );
	
	//------------------------------------------------------------------------------
	//... initialize the Poisson solver
	//------------------------------------------------------------------------------
	bool bdefd	= cf.getValueSafe<bool> ( "poisson" , "fft_fine", true );
	bool bglass = cf.getValueSafe<bool>("output","glass", false);
	bool bsph	= cf.getValueSafe<bool>("setup","do_SPH",false) && do_baryons;
	bool bbshift= bsph && !bglass;
	
	bool kspace	= cf.getValueSafe<bool>( "poisson", "kspace", false );
    bool kspace2LPT = kspace;

	bool decic_DM = cf.getValueSafe<bool>( "output", "glass_cicdeconvolve", false );
	bool decic_baryons = cf.getValueSafe<bool>( "output", "glass_cicdeconvolve", false ) & bsph;

	//... if in unigrid mode, use k-space instead of hybrid
	if(bdefd && (lbase==lmax))
	{
		kspace=true;
		bdefd=false;
		kspace2LPT=false;
	}
	
	std::string poisson_solver_name;
	if( kspace )
		poisson_solver_name = std::string("fft_poisson");
	else
		poisson_solver_name = std::string("mg_poisson");
	
	unsigned grad_order = cf.getValueSafe<unsigned> ( "poisson" , "grad_order", 4 );
	
	
	
	
	//... switch off if using kspace anyway
	//bdefd &= !kspace;
	
	poisson_plugin_creator *the_poisson_plugin_creator = get_poisson_plugin_map()[ poisson_solver_name ];
	// Only root runs the (still-serial) Poisson solver; workers leave it NULL.
	poisson_plugin *the_poisson_solver = MUSIC::mpi::is_root() ? the_poisson_plugin_creator->create( cf ) : NULL;
	
	//---------------------------------------------------------------------------------
	//... THIS IS THE MAIN DRIVER BRANCHING TREE RUNNING THE VARIOUS PARTS OF THE CODE
	//---------------------------------------------------------------------------------
	bool bfatal = false;
	try{
		if( ! do_2LPT )
		{
			LOGUSER("Entering 1LPT branch");
			
			//------------------------------------------------------------------------------
			//... cdm density and displacements
			//------------------------------------------------------------------------------
			std::cout << "=============================================================\n";
			std::cout << "   COMPUTING DARK MATTER DISPLACEMENTS\n";
			std::cout << "-------------------------------------------------------------\n";
			LOGUSER("Computing dark matter displacements...");
			
			grid_hierarchy f( nbnd );//, u(nbnd);
			tf_type my_tf_type = cdm;
			if( !do_baryons || !the_transfer_function_plugin->tf_is_distinct() )
				my_tf_type = total;
			
			
			GenerateDensityHierarchy(	cf, the_transfer_function_plugin, my_tf_type , rh_TF, rand, f, false, false );
			if( MUSIC::mpi::is_root() ){ coarsen_density(rh_Poisson, f, bspectral_sampling); f.add_refinement_mask( rh_Poisson.get_coord_shift() ); normalize_density(f); }

			LOGUSER("Writing CDM data");
			if( MUSIC::mpi::is_root() ) the_output_plugin->write_dm_mass(f);
			if( MUSIC::mpi::is_root() ) the_output_plugin->write_dm_density(f);

			grid_hierarchy u( f );	u.zero();
			{
				const bool use_slab_solve = kspace && (lbase==lmax)
					&& cf.getValueSafe<bool>("setup", "slab_solve_unigrid", false)
					&& MUSIC::mpi::size() > 1;
				if( use_slab_solve ){
					const size_t ng = (size_t)1 << lmax;
					MUSIC::poisson::slab_solve_unigrid( f, u, lmax, ng, ng, ng );
					err = 0.0;
				} else {
					MUSIC::poisson::with_pbox_distributed([&]{
						err = the_poisson_solver->solve(f, u);
					}, f, u);
				}
			}

			if(!bdefd)
				f.deallocate();

			LOGUSER("Writing CDM potential");
			if( MUSIC::mpi::is_root() ) the_output_plugin->write_dm_potential(u);


			//------------------------------------------------------------------------------
			//... DM displacements
			//------------------------------------------------------------------------------
			{
				grid_hierarchy data_forIO(u);
				MUSIC::poisson::with_pbox_distributed([&]{
					for( int icoord = 0; icoord < 3; ++icoord )
					{
						if( bdefd )
						{
							data_forIO.zero();
							*data_forIO.get_grid(data_forIO.levelmax()) = *f.get_grid(f.levelmax());
							poisson_hybrid(*data_forIO.get_grid(data_forIO.levelmax()), icoord, grad_order,
								       data_forIO.levelmin()==data_forIO.levelmax(), decic_DM );
							*data_forIO.get_grid(data_forIO.levelmax()) /= 1<<f.levelmax();
							the_poisson_solver->gradient_add(icoord, u, data_forIO );

						}
						else
							//... displacement
							the_poisson_solver->gradient(icoord, u, data_forIO );
						double dispmax = compute_finest_max( data_forIO );
						LOGINFO("max. %c-displacement of HR particles is %f [mean dx]",'x'+icoord, dispmax*(double)(1ll<<data_forIO.levelmax()));
						coarsen_density( rh_Poisson, data_forIO, false );
						LOGUSER("Writing CDM displacements");
						the_output_plugin->write_dm_position(icoord, data_forIO );
					}
				}, u, data_forIO);
				// data_forIO falls out of SPMD scope below
			}
			if( do_baryons )
				u.deallocate();
			
			
			//------------------------------------------------------------------------------
			//... gas density
			//------------------------------------------------------------------------------
			if( do_baryons )
			{
				std::cout << "=============================================================\n";
				std::cout << "   COMPUTING BARYON DENSITY\n";
				std::cout << "-------------------------------------------------------------\n";
				LOGUSER("Computing baryon density...");
				GenerateDensityHierarchy(	cf, the_transfer_function_plugin, baryon , rh_TF, rand, f, false, bbshift );
				MUSIC::poisson::with_pbox_distributed([&]{
				    coarsen_density(rh_Poisson, f, bspectral_sampling);
				    f.add_refinement_mask( rh_Poisson.get_coord_shift() );
				    normalize_density(f);

				    if( !do_LLA )
				    {
				        LOGUSER("Writing baryon density");
				        the_output_plugin->write_gas_density(f);
				    }
				}, f);

				if( bsph )
				{
				    MUSIC::poisson::with_pbox_distributed([&]{
				        u = f;	u.zero();
				        err = the_poisson_solver->solve(f, u);
				    }, f, u);
				    if(!bdefd)
				        f.deallocate();  // SPMD
				    {
				        grid_hierarchy data_forIO(u);
				        MUSIC::poisson::with_pbox_distributed([&]{
				            for( int icoord = 0; icoord < 3; ++icoord )
				            {
				                if( bdefd )
				                {
				                    data_forIO.zero();
				                    *data_forIO.get_grid(data_forIO.levelmax()) = *f.get_grid(f.levelmax());
				                    poisson_hybrid(*data_forIO.get_grid(data_forIO.levelmax()), icoord, grad_order,
				                               data_forIO.levelmin()==data_forIO.levelmax(), decic_baryons);
				                    *data_forIO.get_grid(data_forIO.levelmax()) /= 1<<f.levelmax();
				                    the_poisson_solver->gradient_add(icoord, u, data_forIO );
				                }
				                else
				                    //... displacement
				                    the_poisson_solver->gradient(icoord, u, data_forIO );

				                coarsen_density( rh_Poisson, data_forIO, false );
				                LOGUSER("Writing baryon displacements");
				                the_output_plugin->write_gas_position(icoord, data_forIO );
				            }
				        }, u, data_forIO);
				    }
				    u.deallocate();  // SPMD
				    if( bdefd )
				        f.deallocate();  // SPMD (idempotent)
				}
				else if( do_LLA )
				{
				    MUSIC::poisson::with_pbox_distributed([&]{
				        u = f;	u.zero();
				        err = the_poisson_solver->solve(f, u);
				        compute_LLA_density( u, f,grad_order );
				    }, f, u);
				    u.deallocate();  // SPMD
				    if( MUSIC::mpi::is_root() ) {
				        normalize_density(f);
				        LOGUSER("Writing baryon density");
				        the_output_plugin->write_gas_density(f);
				    }
				}

				f.deallocate();  // SPMD
			}
			
			
			
			//------------------------------------------------------------------------------
			//... velocities
			//------------------------------------------------------------------------------
			if( (!the_transfer_function_plugin->tf_has_velocities() || !do_baryons) && !bsph )
			{	
				std::cout << "=============================================================\n";
				std::cout << "   COMPUTING VELOCITIES\n";
				std::cout << "-------------------------------------------------------------\n";
				LOGUSER("Computing velocitites...");
				
				if( do_baryons || the_transfer_function_plugin->tf_has_velocities() )
				{
				  LOGUSER("Generating velocity perturbations...");
				  GenerateDensityHierarchy( cf, the_transfer_function_plugin, vtotal , rh_TF, rand, f, false, false );
				  MUSIC::poisson::with_pbox_distributed([&]{
				    coarsen_density(rh_Poisson, f, bspectral_sampling);
				    f.add_refinement_mask( rh_Poisson.get_coord_shift() );
				    normalize_density(f);
				    u = f;
				    u.zero();
				    err = the_poisson_solver->solve(f, u);
				  }, f, u);
				  if(!bdefd)
				    f.deallocate();  // SPMD
				}
				{
				    grid_hierarchy data_forIO(u);
				    MUSIC::poisson::with_pbox_distributed([&]{
				        for( int icoord = 0; icoord < 3; ++icoord )
				        {
				            //... displacement
				            if(bdefd)
				            {
				                data_forIO.zero();
				                *data_forIO.get_grid(data_forIO.levelmax()) = *f.get_grid(f.levelmax());
				                poisson_hybrid(*data_forIO.get_grid(data_forIO.levelmax()), icoord, grad_order,
				                           data_forIO.levelmin()==data_forIO.levelmax(), decic_baryons );
				                *data_forIO.get_grid(data_forIO.levelmax()) /= 1<<f.levelmax();
				                the_poisson_solver->gradient_add(icoord, u, data_forIO );
				            }
				            else
				                the_poisson_solver->gradient(icoord, u, data_forIO );

				            //... multiply to get velocity
				            data_forIO *= cosmo.vfact;

				            double sigv = compute_finest_sigma( data_forIO );
				            LOGINFO("sigma of %c-velocity of high-res particles is %f",'x'+icoord, sigv);

				            coarsen_density( rh_Poisson, data_forIO, false );
				            LOGUSER("Writing CDM velocities");
				            the_output_plugin->write_dm_velocity(icoord, data_forIO);

				            if( do_baryons )
				            {
				                LOGUSER("Writing baryon velocities");
				                the_output_plugin->write_gas_velocity(icoord, data_forIO);
				            }
				        }
				    }, u, data_forIO);
				}
				u.deallocate();  // SPMD

			}
			else
			{
				LOGINFO("Computing separate velocities for CDM and baryons:");
				std::cout << "=============================================================\n";
				std::cout << "   COMPUTING DARK MATTER VELOCITIES\n";
				std::cout << "-------------------------------------------------------------\n";
				LOGUSER("Computing dark matter velocitites...");
				
				//... we do baryons and have velocity transfer functions, or we do SPH and not to shift
				//... do DM first
				GenerateDensityHierarchy(	cf, the_transfer_function_plugin, vcdm , rh_TF, rand, f, false, false );
				MUSIC::poisson::with_pbox_distributed([&]{
				    coarsen_density(rh_Poisson, f, bspectral_sampling);
				    f.add_refinement_mask( rh_Poisson.get_coord_shift() );
				    normalize_density(f);
				    u = f;	u.zero();
				    err = the_poisson_solver->solve(f, u);
				}, f, u);
				if(!bdefd)
				    f.deallocate();  // SPMD
				{
				    grid_hierarchy data_forIO(u);
				    MUSIC::poisson::with_pbox_distributed([&]{
				        for( int icoord = 0; icoord < 3; ++icoord )
				        {
				            //... displacement
				            if(bdefd)
				            {
				                data_forIO.zero();
				                *data_forIO.get_grid(data_forIO.levelmax()) = *f.get_grid(f.levelmax());
				                poisson_hybrid(*data_forIO.get_grid(data_forIO.levelmax()), icoord, grad_order,
				                           data_forIO.levelmin()==data_forIO.levelmax(), decic_DM );
				                *data_forIO.get_grid(data_forIO.levelmax()) /= 1<<f.levelmax();
				                the_poisson_solver->gradient_add(icoord, u, data_forIO );
				            }
				            else
				                the_poisson_solver->gradient(icoord, u, data_forIO );

				            //... multiply to get velocity
				            data_forIO *= cosmo.vfact;

				            double sigv = compute_finest_sigma( data_forIO );
				            LOGINFO("sigma of %c-velocity of high-res DM is %f",'x'+icoord, sigv);

				            coarsen_density( rh_Poisson, data_forIO, false );
				            LOGUSER("Writing CDM velocities");
				            the_output_plugin->write_dm_velocity(icoord, data_forIO);
				        }
				    }, u, data_forIO);
				}
				u.deallocate();  // SPMD
				f.deallocate();  // SPMD (idempotent w.r.t. earlier conditional)


				std::cout << "=============================================================\n";
				std::cout << "   COMPUTING BARYON VELOCITIES\n";
				std::cout << "-------------------------------------------------------------\n";
				LOGUSER("Computing baryon velocitites...");
				//... do baryons
				GenerateDensityHierarchy(	cf, the_transfer_function_plugin, vbaryon , rh_TF, rand, f, false, bbshift );
				MUSIC::poisson::with_pbox_distributed([&]{
				    coarsen_density(rh_Poisson, f, bspectral_sampling);
				    f.add_refinement_mask( rh_Poisson.get_coord_shift() );
				    normalize_density(f);
				    u = f;	u.zero();
				    err = the_poisson_solver->solve(f, u);
				}, f, u);
				if(!bdefd)
				    f.deallocate();  // SPMD
				{
				    grid_hierarchy data_forIO(u);
				    MUSIC::poisson::with_pbox_distributed([&]{
				        for( int icoord = 0; icoord < 3; ++icoord )
				        {
				            //... displacement
				            if(bdefd)
				            {
				                data_forIO.zero();
				                *data_forIO.get_grid(data_forIO.levelmax()) = *f.get_grid(f.levelmax());
				                poisson_hybrid(*data_forIO.get_grid(data_forIO.levelmax()), icoord, grad_order,
				                           data_forIO.levelmin()==data_forIO.levelmax(), decic_baryons );
				                *data_forIO.get_grid(data_forIO.levelmax()) /= 1<<f.levelmax();
				                the_poisson_solver->gradient_add(icoord, u, data_forIO );
				            }
				            else
				                the_poisson_solver->gradient(icoord, u, data_forIO );

				            //... multiply to get velocity
				            data_forIO *= cosmo.vfact;

				            double sigv = compute_finest_sigma( data_forIO );
				            LOGINFO("sigma of %c-velocity of high-res baryons is %f",'x'+icoord, sigv);

				            coarsen_density( rh_Poisson, data_forIO, false );
				            LOGUSER("Writing baryon velocities");
				            the_output_plugin->write_gas_velocity(icoord, data_forIO);
				        }
				    }, u, data_forIO);
				}
				u.deallocate();  // SPMD
				f.deallocate();  // SPMD
			}
		/*********************************************************************************************/
		/*********************************************************************************************/
		/*** 2LPT ************************************************************************************/
		/*********************************************************************************************/
		}else {
			//.. use 2LPT ...
			LOGUSER("Entering 2LPT branch");
			
			grid_hierarchy f( nbnd ), u1(nbnd), u2LPT(nbnd), f2LPT( nbnd );
			
			
			
			tf_type my_tf_type = vcdm;
			bool dm_only = !do_baryons;
			if( !do_baryons || !the_transfer_function_plugin->tf_has_velocities() )
				my_tf_type = total;
			
			std::cout << "=============================================================\n";
			if( my_tf_type == total )
			{
				std::cout << "   COMPUTING VELOCITIES\n";
				LOGUSER("Computing velocities...");				
			}else{
				std::cout << "   COMPUTING DARK MATTER VELOCITIES\n";
				LOGUSER("Computing dark matter velocities...");	
			}
			std::cout << "-------------------------------------------------------------\n";	

			
			GenerateDensityHierarchy(	cf, the_transfer_function_plugin, my_tf_type , rh_TF, rand, f, false, false );
			{ MUSIC::poisson::phase_scope _ps;
			if( MUSIC::mpi::is_root() )
			{
			coarsen_density(rh_Poisson, f, bspectral_sampling);
			f.add_refinement_mask( rh_Poisson.get_coord_shift() );
            normalize_density(f);

			if( dm_only )
			{
				the_output_plugin->write_dm_density(f);
				the_output_plugin->write_dm_mass(f);
			}

			u1 = f;	u1.zero();

			//... compute 1LPT term
			err = the_poisson_solver->solve(f, u1);


			//... compute 2LPT term
			if(bdefd)
				f2LPT=f;
			else
				f.deallocate();

			LOGINFO("Computing 2LPT term....");
			if( !kspace2LPT )
				compute_2LPT_source(u1, f2LPT, grad_order );
			else{
				LOGUSER("computing term using FFT");
				compute_2LPT_source_FFT(cf, u1, f2LPT);
			}

            LOGINFO("Solving 2LPT Poisson equation");
			u2LPT = u1; u2LPT.zero();
			err = the_poisson_solver->solve(f2LPT, u2LPT);


			//... if doing the hybrid step, we need a combined source term
			if( bdefd )
			{
				f2LPT*=6.0/7.0/vfac2lpt;
				f+=f2LPT;

				if( !dm_only )
					f2LPT.deallocate();
			}

			//... add the 2LPT contribution
			u2LPT *= 6.0/7.0/vfac2lpt;
			u1 += u2LPT;


			grid_hierarchy data_forIO(u1);
			for( int icoord = 0; icoord < 3; ++icoord )
			{
				if(bdefd)
				{
					data_forIO.zero();
					*data_forIO.get_grid(data_forIO.levelmax()) = *f.get_grid(f.levelmax());
					poisson_hybrid(*data_forIO.get_grid(data_forIO.levelmax()), icoord, grad_order,
						       data_forIO.levelmin()==data_forIO.levelmax(), decic_DM );
					*data_forIO.get_grid(data_forIO.levelmax()) /= (1<<f.levelmax());
					the_poisson_solver->gradient_add(icoord, u1, data_forIO );
				}
				else
					the_poisson_solver->gradient(icoord, u1, data_forIO );

				data_forIO *= cosmo.vfact;

				double sigv = compute_finest_sigma( data_forIO );
				std::cerr << " - velocity component " << icoord << " : sigma = " << sigv << std::endl;

				coarsen_density( rh_Poisson, data_forIO, false );
				LOGUSER("Writing CDM velocities");
				the_output_plugin->write_dm_velocity(icoord, data_forIO);

				if( do_baryons && !the_transfer_function_plugin->tf_has_velocities() && !bsph)
				{
					LOGUSER("Writing baryon velocities");
					the_output_plugin->write_gas_velocity(icoord, data_forIO);
				}
			}
			data_forIO.deallocate();
			if( !dm_only )
				u1.deallocate();
			} // end is_root() — 2LPT vcdm/total post-density
			}
			
			
			if( do_baryons && (the_transfer_function_plugin->tf_has_velocities() || bsph) )
			{
				std::cout << "=============================================================\n";
				std::cout << "   COMPUTING BARYON VELOCITIES\n";
				std::cout << "-------------------------------------------------------------\n";
				LOGUSER("Computing baryon displacements...");
				
				GenerateDensityHierarchy(	cf, the_transfer_function_plugin, vbaryon , rh_TF, rand, f, false, bbshift );
				{ MUSIC::poisson::phase_scope _ps;
				if( MUSIC::mpi::is_root() )
				{
				coarsen_density(rh_Poisson, f, bspectral_sampling);
				f.add_refinement_mask( rh_Poisson.get_coord_shift() );
                normalize_density(f);

				u1 = f;	u1.zero();

				if(bdefd)
					f2LPT=f;

				//... compute 1LPT term
				err = the_poisson_solver->solve(f, u1);

				LOGINFO("Writing baryon potential");
				the_output_plugin->write_gas_potential(u1);

				//... compute 2LPT term
				u2LPT = f; u2LPT.zero();

				if( !kspace2LPT )
					compute_2LPT_source(u1, f2LPT, grad_order );
				else
					compute_2LPT_source_FFT(cf, u1, f2LPT);


				err = the_poisson_solver->solve(f2LPT, u2LPT);

				//... if doing the hybrid step, we need a combined source term
				if( bdefd )
				{
					f2LPT*=6.0/7.0/vfac2lpt;
					f+=f2LPT;

					f2LPT.deallocate();
				}

				//... add the 2LPT contribution
				u2LPT *= 6.0/7.0/vfac2lpt;
				u1 += u2LPT;
				u2LPT.deallocate();

				grid_hierarchy data_forIO(u1);
				for( int icoord = 0; icoord < 3; ++icoord )
				{
					if(bdefd)
					{
						data_forIO.zero();
						*data_forIO.get_grid(data_forIO.levelmax()) = *f.get_grid(f.levelmax());
						poisson_hybrid(*data_forIO.get_grid(data_forIO.levelmax()), icoord, grad_order,
							       data_forIO.levelmin()==data_forIO.levelmax(), decic_baryons );
						*data_forIO.get_grid(data_forIO.levelmax()) /= (1<<f.levelmax());
						the_poisson_solver->gradient_add(icoord, u1, data_forIO );
					}
					else
						the_poisson_solver->gradient(icoord, u1, data_forIO );

					data_forIO *= cosmo.vfact;

					double sigv = compute_finest_sigma( data_forIO );
					std::cerr << " - velocity component " << icoord << " : sigma = " << sigv << std::endl;

					coarsen_density( rh_Poisson, data_forIO, false );
					LOGUSER("Writing baryon velocities");
					the_output_plugin->write_gas_velocity(icoord, data_forIO);
				}
				data_forIO.deallocate();
				u1.deallocate();
				} // end is_root() — 2LPT vbaryon post-density
				}
			}
			
			
			std::cout << "=============================================================\n";
			std::cout << "   COMPUTING DARK MATTER DISPLACEMENTS\n";
			std::cout << "-------------------------------------------------------------\n";
			LOGUSER("Computing dark matter displacements...");
			
			//... if baryons are enabled, the displacements have to be recomputed
			//... otherwise we can compute them directly from the velocities
			if( !dm_only )
			{
				// my_tf_type is cdm if do_baryons==true, total otherwise
				my_tf_type = cdm;
				if( !do_baryons || !the_transfer_function_plugin->tf_is_distinct() )
					my_tf_type = total;

				GenerateDensityHierarchy(	cf, the_transfer_function_plugin, my_tf_type , rh_TF, rand, f, false, false );
				{ MUSIC::poisson::phase_scope _ps;
				if( MUSIC::mpi::is_root() )
				{
				coarsen_density(rh_Poisson, f, bspectral_sampling);
				f.add_refinement_mask( rh_Poisson.get_coord_shift() );
                normalize_density(f);

				LOGUSER("Writing CDM data");
				the_output_plugin->write_dm_density(f);
				the_output_plugin->write_dm_mass(f);
				u1 = f;	u1.zero();

				if(bdefd)
					f2LPT=f;

				//... compute 1LPT term
				err = the_poisson_solver->solve(f, u1);

				//... compute 2LPT term
				u2LPT = f; u2LPT.zero();

				if( !kspace2LPT )
					compute_2LPT_source(u1, f2LPT, grad_order );
				else
					compute_2LPT_source_FFT(cf, u1, f2LPT);

				err = the_poisson_solver->solve(f2LPT, u2LPT);

				if( bdefd )
				{
					f2LPT*=3.0/7.0;
					f+=f2LPT;
					f2LPT.deallocate();
				}

				u2LPT *= 3.0/7.0;
				u1 += u2LPT;
				u2LPT.deallocate();
				} // end is_root() — 2LPT !dm_only DM displacements
				}
			}else if( MUSIC::mpi::is_root() ){
				//... reuse prior data
				/*f-=f2LPT;
				the_output_plugin->write_dm_density(f);
				the_output_plugin->write_dm_mass(f);
				f+=f2LPT;*/

				u2LPT *= 0.5;
				u1 -= u2LPT;
				u2LPT.deallocate();

				if(bdefd)
				{
					f2LPT *= 0.5;
					f-=f2LPT;
					f2LPT.deallocate();
				}
			}

			{ MUSIC::poisson::phase_scope _ps;
			if( MUSIC::mpi::is_root() )
			{
			grid_hierarchy data_forIO(u1);

			for( int icoord = 0; icoord < 3; ++icoord )
			{
				//... displacement
				if(bdefd)
				{
					data_forIO.zero();
					*data_forIO.get_grid(data_forIO.levelmax()) = *f.get_grid(f.levelmax());
					poisson_hybrid(*data_forIO.get_grid(data_forIO.levelmax()), icoord, grad_order,
						       data_forIO.levelmin()==data_forIO.levelmax(), decic_DM );
					*data_forIO.get_grid(data_forIO.levelmax()) /= 1<<f.levelmax();
					the_poisson_solver->gradient_add(icoord, u1, data_forIO );
				}
				else
					the_poisson_solver->gradient(icoord, u1, data_forIO );

				double dispmax = compute_finest_max( data_forIO );
				LOGINFO("max. %c-displacement of HR particles is %f [mean dx]",'x'+icoord, dispmax*(double)(1ll<<data_forIO.levelmax()));

				coarsen_density( rh_Poisson, data_forIO, false );
				LOGUSER("Writing CDM displacements");
				the_output_plugin->write_dm_position(icoord, data_forIO );
			}

			data_forIO.deallocate();
			u1.deallocate();
			} // end is_root() — 2LPT DM displacements post-density
			}
			

			if( do_baryons && !bsph )
			{	
				std::cout << "=============================================================\n";
				std::cout << "   COMPUTING BARYON DENSITY\n";
				std::cout << "-------------------------------------------------------------\n";
				LOGUSER("Computing baryon density...");
				
				GenerateDensityHierarchy(	cf, the_transfer_function_plugin, baryon , rh_TF, rand, f, true, false );
				{ MUSIC::poisson::phase_scope _ps;
				if( MUSIC::mpi::is_root() )
				{
				coarsen_density(rh_Poisson, f, bspectral_sampling);
				f.add_refinement_mask( rh_Poisson.get_coord_shift() );
                normalize_density(f);

				if( !do_LLA )
					the_output_plugin->write_gas_density(f);
				else
				{
					u1 = f;	u1.zero();

					//... compute 1LPT term
					err = the_poisson_solver->solve(f, u1);

					//... compute 2LPT term
					u2LPT = f; u2LPT.zero();

					if( !kspace2LPT )
						compute_2LPT_source(u1, f2LPT, grad_order );
					else
						compute_2LPT_source_FFT(cf, u1, f2LPT);

					err = the_poisson_solver->solve(f2LPT, u2LPT);
					u2LPT *= 3.0/7.0;
					u1 += u2LPT;
					u2LPT.deallocate();

					compute_LLA_density( u1, f, grad_order );
                    normalize_density(f);

					LOGUSER("Writing baryon density");
					the_output_plugin->write_gas_density(f);
				}
				} // end is_root() — 2LPT gas density (!bsph)
				}
			}
			else if( do_baryons && bsph )
			{
				std::cout << "=============================================================\n";
				std::cout << "   COMPUTING BARYON DISPLACEMENTS\n";
				std::cout << "-------------------------------------------------------------\n";
				LOGUSER("Computing baryon displacements...");
				
				GenerateDensityHierarchy(	cf, the_transfer_function_plugin, baryon , rh_TF, rand, f, false, bbshift );
				{ MUSIC::poisson::phase_scope _ps;
				if( MUSIC::mpi::is_root() )
				{
				coarsen_density(rh_Poisson, f, bspectral_sampling);
				f.add_refinement_mask( rh_Poisson.get_coord_shift() );
                normalize_density(f);

				LOGUSER("Writing baryon density");
				the_output_plugin->write_gas_density(f);
				u1 = f;	u1.zero();

				if(bdefd)
					f2LPT=f;

				//... compute 1LPT term
				err = the_poisson_solver->solve(f, u1);

				//... compute 2LPT term
				u2LPT = f; u2LPT.zero();

				if( !kspace2LPT )
					compute_2LPT_source(u1, f2LPT, grad_order );
				else
					compute_2LPT_source_FFT(cf, u1, f2LPT);

				err = the_poisson_solver->solve(f2LPT, u2LPT);

				if( bdefd )
				{
					f2LPT*=3.0/7.0;
					f+=f2LPT;
					f2LPT.deallocate();
				}

				u2LPT *= 3.0/7.0;
				u1 += u2LPT;
				u2LPT.deallocate();

				grid_hierarchy data_forIO(u1);

				for( int icoord = 0; icoord < 3; ++icoord )
				{
					//... displacement
					if(bdefd)
					{
						data_forIO.zero();
						*data_forIO.get_grid(data_forIO.levelmax()) = *f.get_grid(f.levelmax());
						poisson_hybrid(*data_forIO.get_grid(data_forIO.levelmax()), icoord, grad_order,
							       data_forIO.levelmin()==data_forIO.levelmax(), decic_baryons );
						*data_forIO.get_grid(data_forIO.levelmax()) /= 1<<f.levelmax();
						the_poisson_solver->gradient_add(icoord, u1, data_forIO );
					}
					else
						the_poisson_solver->gradient(icoord, u1, data_forIO );

					coarsen_density( rh_Poisson, data_forIO, false );
					LOGUSER("Writing baryon displacements");
					the_output_plugin->write_gas_position(icoord, data_forIO );
				}
				} // end is_root() — 2LPT gas displacements (bsph)
				}
			}

		}

		//------------------------------------------------------------------------------
		//... finish output
		//------------------------------------------------------------------------------

		// Collective: workers participate in finalize_collective() for plugins
		// that override it (e.g. gadget2 multi-file). Default impl is rank-0-only.
		if( the_output_plugin ){
			the_output_plugin->finalize_collective();
			delete the_output_plugin;
			the_output_plugin = NULL;
		}
		
	}catch(std::runtime_error& excp){
		LOGERR("Fatal error occured. Code will exit:");
		LOGERR("Exception: %s",excp.what());
		std::cerr << " - " << excp.what() << std::endl;
		std::cerr << " - A fatal error occured. We need to exit...\n";
		bfatal = true;
	}

	std::cout << "=============================================================\n";
	
	

	if( !bfatal )
	{	
		std::cout << " - Wrote output file \'" << outfname << "\'\n     using plugin \'" << outformat << "\'...\n";
		LOGUSER("Wrote output file \'%s\'.",outfname.c_str());
	}
	
	//------------------------------------------------------------------------------
	//... clean up
	//------------------------------------------------------------------------------
	delete the_transfer_function_plugin;
	delete the_poisson_solver;

#if defined(FFTW3) and not defined(SINGLETHREAD_FFTW)
	#ifdef SINGLE_PRECISION
	fftwf_cleanup_threads();
	#else
	fftw_cleanup_threads();
	#endif
#endif

#ifdef USE_MPI
	#ifdef SINGLE_PRECISION
	fftwf_mpi_cleanup();
	#else
	fftw_mpi_cleanup();
	#endif
#endif
	
	
	//------------------------------------------------------------------------------
	//... we are done !
	//------------------------------------------------------------------------------
	if( MUSIC::mpi::is_root() ) std::cout << " - Done!" << std::endl << std::endl;

	ltime=time(NULL);

	LOGUSER("Run finished succesfully on %s",asctime( localtime(&ltime) ));

	if( MUSIC::mpi::is_root() ) cf.log_dump();

#ifdef USE_MPI
	MPI_Finalize();
#endif

	return 0;
}
