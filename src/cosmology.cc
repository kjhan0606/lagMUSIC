/*
 
 cosmology.cc - This file is part of MUSIC -
 a code to generate multi-scale initial conditions 
 for cosmological simulations 
 
 Copyright (C) 2010  Oliver Hahn
 
 */

#include "cosmology.hh"
#include "mesh.hh"
#include "mg_operators.hh"
#include "general.hh"
#include "mpi_helper.hh"
#include "mpi_poisson.hh"

#define ACC(i,j,k) ((*u.get_grid((ilevel)))((i),(j),(k)))
#define SQR(x)	((x)*(x))

#if defined(FFTW3) && defined(SINGLE_PRECISION)
#define fftw_complex fftwf_complex
#endif


void compute_LLA_density( const grid_hierarchy& u, grid_hierarchy& fnew, unsigned order )
{
	fnew = u;
	
	for( unsigned ilevel=u.levelmin(); ilevel<=u.levelmax(); ++ilevel )
	{
		double h = pow(2.0,ilevel), h2 = h*h, h2_4 = 0.25*h2;
		meshvar_bnd *pvar = fnew.get_grid(ilevel);
		
		
		if( order == 2 )
		{
			#pragma omp parallel for //reduction(+:sum_corr,sum,sum2)
			for( int ix = 0; ix < (int)(*u.get_grid(ilevel)).size(0); ++ix )
				for( int iy = 0; iy < (int)(*u.get_grid(ilevel)).size(1); ++iy )
					for( int iz = 0; iz < (int)(*u.get_grid(ilevel)).size(2); ++iz )
					{
						double D[3][3];
						
						D[0][0] = (ACC(ix-1,iy,iz)-2.0*ACC(ix,iy,iz)+ACC(ix+1,iy,iz)) * h2;
						D[1][1] = (ACC(ix,iy-1,iz)-2.0*ACC(ix,iy,iz)+ACC(ix,iy+1,iz)) * h2;
						D[2][2] = (ACC(ix,iy,iz-1)-2.0*ACC(ix,iy,iz)+ACC(ix,iy,iz+1)) * h2;
											
						D[0][1] = D[1][0] = (ACC(ix-1,iy-1,iz)-ACC(ix-1,iy+1,iz)-ACC(ix+1,iy-1,iz)+ACC(ix+1,iy+1,iz))*h2_4;
						D[0][2] = D[2][0] = (ACC(ix-1,iy,iz-1)-ACC(ix-1,iy,iz+1)-ACC(ix+1,iy,iz-1)+ACC(ix+1,iy,iz+1))*h2_4;
						D[1][2] = D[2][1] = (ACC(ix,iy-1,iz-1)-ACC(ix,iy-1,iz+1)-ACC(ix,iy+1,iz-1)+ACC(ix,iy+1,iz+1))*h2_4;
						
						D[0][0] += 1.0;
						D[1][1] += 1.0;
						D[2][2] += 1.0;
						
						double det = D[0][0]*D[1][1]*D[2][2]
						-	D[0][0]*D[1][2]*D[2][1]
						-   D[1][0]*D[0][1]*D[2][2]
						+	D[1][0]*D[0][2]*D[1][2]
						+	D[2][0]*D[0][1]*D[1][2]
						-	D[2][0]*D[0][2]*D[1][1];
						
						(*pvar)(ix,iy,iz) = 1.0/det-1.0;
						
					}
		}
		else if ( order == 4 )
		{
			#pragma omp parallel for 
			for( int ix = 0; ix < (int)(*u.get_grid(ilevel)).size(0); ++ix )
				for( int iy = 0; iy < (int)(*u.get_grid(ilevel)).size(1); ++iy )
					for( int iz = 0; iz < (int)(*u.get_grid(ilevel)).size(2); ++iz )
					{
						double D[3][3];
						
						D[0][0] = (-ACC(ix-2,iy,iz)+16.*ACC(ix-1,iy,iz)-30.0*ACC(ix,iy,iz)+16.*ACC(ix+1,iy,iz)-ACC(ix+2,iy,iz)) * h2/12.0;
						D[1][1] = (-ACC(ix,iy-2,iz)+16.*ACC(ix,iy-1,iz)-30.0*ACC(ix,iy,iz)+16.*ACC(ix,iy+1,iz)-ACC(ix,iy+2,iz)) * h2/12.0;
						D[2][2] = (-ACC(ix,iy,iz-2)+16.*ACC(ix,iy,iz-1)-30.0*ACC(ix,iy,iz)+16.*ACC(ix,iy,iz+1)-ACC(ix,iy,iz+2)) * h2/12.0;
						
						D[0][1] = D[1][0] = (ACC(ix-1,iy-1,iz)-ACC(ix-1,iy+1,iz)-ACC(ix+1,iy-1,iz)+ACC(ix+1,iy+1,iz))*h2_4;
						D[0][2] = D[2][0] = (ACC(ix-1,iy,iz-1)-ACC(ix-1,iy,iz+1)-ACC(ix+1,iy,iz-1)+ACC(ix+1,iy,iz+1))*h2_4;
						D[1][2] = D[2][1] = (ACC(ix,iy-1,iz-1)-ACC(ix,iy-1,iz+1)-ACC(ix,iy+1,iz-1)+ACC(ix,iy+1,iz+1))*h2_4;
						
						
						D[0][0] += 1.0;
						D[1][1] += 1.0;
						D[2][2] += 1.0;
						
						double det = D[0][0]*D[1][1]*D[2][2]
						-	D[0][0]*D[1][2]*D[2][1]
						-   D[1][0]*D[0][1]*D[2][2]
						+	D[1][0]*D[0][2]*D[1][2]
						+	D[2][0]*D[0][1]*D[1][2]
						-	D[2][0]*D[0][2]*D[1][1];
						
						(*pvar)(ix,iy,iz) = 1.0/det-1.0;
						
					}
		}
		else if ( order == 6 )
		{
			h2_4/=36.;
			h2/=180.;
			#pragma omp parallel for 
			for( int ix = 0; ix < (int)(*u.get_grid(ilevel)).size(0); ++ix )
				for( int iy = 0; iy < (int)(*u.get_grid(ilevel)).size(1); ++iy )
					for( int iz = 0; iz < (int)(*u.get_grid(ilevel)).size(2); ++iz )
					{
						double D[3][3];
						
						D[0][0] = (2.*ACC(ix-3,iy,iz)-27.*ACC(ix-2,iy,iz)+270.*ACC(ix-1,iy,iz)-490.0*ACC(ix,iy,iz)+270.*ACC(ix+1,iy,iz)-27.*ACC(ix+2,iy,iz)+2.*ACC(ix+3,iy,iz)) * h2;
						D[1][1] = (2.*ACC(ix,iy-3,iz)-27.*ACC(ix,iy-2,iz)+270.*ACC(ix,iy-1,iz)-490.0*ACC(ix,iy,iz)+270.*ACC(ix,iy+1,iz)-27.*ACC(ix,iy+2,iz)+2.*ACC(ix,iy+3,iz)) * h2;
						D[2][2] = (2.*ACC(ix,iy,iz-3)-27.*ACC(ix,iy,iz-2)+270.*ACC(ix,iy,iz-1)-490.0*ACC(ix,iy,iz)+270.*ACC(ix,iy,iz+1)-27.*ACC(ix,iy,iz+2)+2.*ACC(ix,iy,iz+3)) * h2;
						
						//.. this is actually 8th order accurate
						D[0][1] = D[1][0] = (64.*(ACC(ix-1,iy-1,iz)-ACC(ix-1,iy+1,iz)-ACC(ix+1,iy-1,iz)+ACC(ix+1,iy+1,iz))
											 -8.*(ACC(ix-2,iy-1,iz)-ACC(ix+2,iy-1,iz)-ACC(ix-2,iy+1,iz)+ACC(ix+2,iy+1,iz)
												+ ACC(ix-1,iy-2,iz)-ACC(ix-1,iy+2,iz)-ACC(ix+1,iy-2,iz)+ACC(ix+1,iy+2,iz))
											 +1.*(ACC(ix-2,iy-2,iz)-ACC(ix-2,iy+2,iz)-ACC(ix+2,iy-2,iz)+ACC(ix+2,iy+2,iz)))*h2_4;
						D[0][2] = D[2][0] = (64.*(ACC(ix-1,iy,iz-1)-ACC(ix-1,iy,iz+1)-ACC(ix+1,iy,iz-1)+ACC(ix+1,iy,iz+1))
											 -8.*(ACC(ix-2,iy,iz-1)-ACC(ix+2,iy,iz-1)-ACC(ix-2,iy,iz+1)+ACC(ix+2,iy,iz+1)
												+ ACC(ix-1,iy,iz-2)-ACC(ix-1,iy,iz+2)-ACC(ix+1,iy,iz-2)+ACC(ix+1,iy,iz+2))
											 +1.*(ACC(ix-2,iy,iz-2)-ACC(ix-2,iy,iz+2)-ACC(ix+2,iy,iz-2)+ACC(ix+2,iy,iz+2)))*h2_4;
						D[1][2] = D[2][1] = (64.*(ACC(ix,iy-1,iz-1)-ACC(ix,iy-1,iz+1)-ACC(ix,iy+1,iz-1)+ACC(ix,iy+1,iz+1))
											 -8.*(ACC(ix,iy-2,iz-1)-ACC(ix,iy+2,iz-1)-ACC(ix,iy-2,iz+1)+ACC(ix,iy+2,iz+1)
												+ ACC(ix,iy-1,iz-2)-ACC(ix,iy-1,iz+2)-ACC(ix,iy+1,iz-2)+ACC(ix,iy+1,iz+2))
											 +1.*(ACC(ix,iy-2,iz-2)-ACC(ix,iy-2,iz+2)-ACC(ix,iy+2,iz-2)+ACC(ix,iy+2,iz+2)))*h2_4;
						
						D[0][0] += 1.0;
						D[1][1] += 1.0;
						D[2][2] += 1.0;
						
						double det = D[0][0]*D[1][1]*D[2][2]
						-	D[0][0]*D[1][2]*D[2][1]
						-   D[1][0]*D[0][1]*D[2][2]
						+	D[1][0]*D[0][2]*D[1][2]
						+	D[2][0]*D[0][1]*D[1][2]
						-	D[2][0]*D[0][2]*D[1][1];
						
						(*pvar)(ix,iy,iz) = 1.0/det-1.0;
						
					}
			
		}else
			throw std::runtime_error("compute_LLA_density : invalid operator order specified");

	}
	
}


void compute_Lu_density( const grid_hierarchy& u, grid_hierarchy& fnew, unsigned order )
{
	fnew = u;
	
	for( unsigned ilevel=u.levelmin(); ilevel<=u.levelmax(); ++ilevel )
	{
		double h = pow(2.0,ilevel), h2 = h*h;
		meshvar_bnd *pvar = fnew.get_grid(ilevel);
		
		#pragma omp parallel for
		for( int ix = 0; ix < (int)(*u.get_grid(ilevel)).size(0); ++ix )
			for( int iy = 0; iy < (int)(*u.get_grid(ilevel)).size(1); ++iy )
				for( int iz = 0; iz < (int)(*u.get_grid(ilevel)).size(2); ++iz )
				{
					double D[3][3];
					
					D[0][0] = 1.0 + (ACC(ix-1,iy,iz)-2.0*ACC(ix,iy,iz)+ACC(ix+1,iy,iz)) * h2;
					D[1][1] = 1.0 + (ACC(ix,iy-1,iz)-2.0*ACC(ix,iy,iz)+ACC(ix,iy+1,iz)) * h2;
					D[2][2] = 1.0 + (ACC(ix,iy,iz-1)-2.0*ACC(ix,iy,iz)+ACC(ix,iy,iz+1)) * h2;
					
					(*pvar)(ix,iy,iz) = -(D[0][0]+D[1][1]+D[2][2] - 3.0);
					
				}
	}
	
}


void compute_2LPT_source_FFT( config_file& cf_, const grid_hierarchy& u, grid_hierarchy& fnew )
{
	if( u.levelmin() != u.levelmax() )
		throw std::runtime_error("FFT 2LPT can only be run in Unigrid mode!");

	// Determine SPMD slab path *before* the rank-0 boost decision below. The
	// boost (B2-2lpt pattern) assumes workers are idle in their barrier; that's
	// true only for the rank-0-serial path. On the slab path workers are active
	// in FFTW3-MPI plans, so boosting rank 0 makes the per-rank plan thread
	// counts asymmetric and over-subscribes rank 0's cores when multiple ranks
	// share a node.
#ifdef USE_MPI
	const bool use_slab_2lpt = (MUSIC::mpi::size() > 1)
		&& cf_.getValueSafe<bool>("setup", "slab_2lpt_unigrid", true);
#else
	const bool use_slab_2lpt = false;
#endif

	// B2-2lpt: rank-0-serial path runs on rank-0 only (all 5 call sites in
	// main.cc are inside `if(MUSIC::mpi::is_root())` blocks; workers idle in
	// MPI_Barrier after the surrounding GenerateDensityHierarchy returns). 1
	// r2c + 6 c2r plus the OMP-parallel kernel and copy loops all benefit from
	// boosting rank-0's thread pool to claim the workers' cores. Mirrors the
	// random.cc B2/B2-fix pattern (omp_set_num_threads + fftw_plan_with_nthreads
	// paired). Skipped for slab path (workers active).
	const int omp_saved = omp_get_max_threads();
	int omp_boost = omp_saved;
	if (!use_slab_2lpt) {
		const std::string boost_mode =
			cf_.getValueSafe<std::string>("setup", "lpt2_boost_threads",
			                              std::string("auto"));
		if (boost_mode != std::string("no")) {
			if (boost_mode == std::string("auto")) {
				const int avail = omp_get_num_procs();
				const int cap   = omp_saved * std::max(1, MUSIC::mpi::size());
				omp_boost = std::min(avail, cap);
			} else {
				try { omp_boost = std::max(1, std::stoi(boost_mode)); }
				catch(...) { omp_boost = omp_saved; }
			}
		}
		if (omp_boost > omp_saved) {
			LOGINFO("B2-2lpt: rank-0 2LPT-FFT OMP boost %d -> %d (workers idle in barrier)",
			        omp_saved, omp_boost);
			omp_set_num_threads(omp_boost);
#if defined(FFTW3) && !defined(SINGLETHREAD_FFTW)
#ifdef SINGLE_PRECISION
			fftwf_plan_with_nthreads(omp_boost);
#else
			fftw_plan_with_nthreads(omp_boost);
#endif
#endif
		}
	}
	const double t_lpt2_start = omp_get_wtime();

	fnew = u;
	size_t nx,ny,nz,nzp;
	nx = u.get_grid(u.levelmax())->size(0);
	ny = u.get_grid(u.levelmax())->size(1);
	nz = u.get_grid(u.levelmax())->size(2);
	nzp = 2*(nz/2+1);

	// Phase F2LPT.1: SPMD distributed path. Caller wraps this in a phase_scope
	// (workers parked in worker_pump), so rank 0 can broadcast OP_LPT2_FFT and
	// participate in the collective r2c/c2r FFTs. The composite (sum of 2x2
	// Hessian minors) is written to fnew's grid as in the serial path. Allocates
	// only 1 rank-0 padded buffer (the phi/result transfer staging) — the 6
	// Hessian-component scratch buffers are per-rank slabs inside rank0_dist_lpt2.
#ifdef USE_MPI
	if (use_slab_2lpt) {
		fftw_real *phi_buf = new fftw_real[nx*ny*nzp];
		fftw_real *dst_buf = new fftw_real[nx*ny*nzp];
		#pragma omp parallel for
		for( int i = 0; i < (int)nx; ++i )
			for( size_t j = 0; j < ny; ++j )
				for( size_t k = 0; k < nz; ++k ) {
					size_t idx = ((size_t)i*ny + j) * nzp + k;
					phi_buf[idx] = (*u.get_grid(u.levelmax()))(i, (int)j, (int)k);
				}
		MUSIC::poisson::rank0_dist_lpt2<fftw_real>(phi_buf, dst_buf, nx, ny, nz);
		#pragma omp parallel for
		for( int i = 0; i < (int)nx; ++i )
			for( size_t j = 0; j < ny; ++j )
				for( size_t k = 0; k < nz; ++k ) {
					size_t idx = ((size_t)i*ny + j) * nzp + k;
					(*fnew.get_grid(u.levelmax()))(i, (int)j, (int)k) = dst_buf[idx];
				}
		delete[] phi_buf;
		delete[] dst_buf;
		const double t_lpt2_total = omp_get_wtime() - t_lpt2_start;
		LOGINFO("lpt2-profile compute_2LPT_source_FFT(slab) nx=%zu  wall=%.3fs  np=%d  threads=%d",
		        nx, t_lpt2_total, MUSIC::mpi::size(), omp_boost);
		if (omp_boost > omp_saved) {
			omp_set_num_threads(omp_saved);
#if defined(FFTW3) && !defined(SINGLETHREAD_FFTW)
#ifdef SINGLE_PRECISION
			fftwf_plan_with_nthreads(omp_saved);
#else
			fftw_plan_with_nthreads(omp_saved);
#endif
#endif
		}
		return;
	}
#endif

	//... copy data ..................................................
	fftw_real *data = new fftw_real[nx*ny*nzp];
	fftw_complex *cdata = reinterpret_cast<fftw_complex*> (data);
	
	fftw_complex	*cdata_11, *cdata_12, *cdata_13, *cdata_22, *cdata_23, *cdata_33;
	fftw_real		*data_11, *data_12, *data_13, *data_22, *data_23, *data_33;
	
	data_11 = new fftw_real[nx*ny*nzp]; cdata_11 = reinterpret_cast<fftw_complex*> (data_11);
	data_12 = new fftw_real[nx*ny*nzp]; cdata_12 = reinterpret_cast<fftw_complex*> (data_12);
	data_13 = new fftw_real[nx*ny*nzp]; cdata_13 = reinterpret_cast<fftw_complex*> (data_13);
	data_22 = new fftw_real[nx*ny*nzp]; cdata_22 = reinterpret_cast<fftw_complex*> (data_22);
	data_23 = new fftw_real[nx*ny*nzp]; cdata_23 = reinterpret_cast<fftw_complex*> (data_23);
	data_33 = new fftw_real[nx*ny*nzp]; cdata_33 = reinterpret_cast<fftw_complex*> (data_33);
	
	#pragma omp parallel for
	for( int i=0; i<(int)nx; ++i )
		for( size_t j=0; j<ny; ++j )	
			for( size_t k=0; k<nz; ++k )
			{
				size_t idx = ((size_t)i*ny+j)*nzp+k;
				data[idx] = (*u.get_grid(u.levelmax()))(i,j,k);
			}
	
	//... perform FFT and Poisson solve................................
#ifdef FFTW3
	
	#ifdef SINGLE_PRECISION
	fftwf_plan
		plan  = fftwf_plan_dft_r2c_3d(nx,ny,nz, data, cdata, FFTW_ESTIMATE),
		iplan = fftwf_plan_dft_c2r_3d(nx,ny,nz, cdata, data, FFTW_ESTIMATE),
		ip11  = fftwf_plan_dft_c2r_3d(nx,ny,nz, cdata_11, data_11, FFTW_ESTIMATE),
		ip12  = fftwf_plan_dft_c2r_3d(nx,ny,nz, cdata_12, data_12, FFTW_ESTIMATE),
		ip13  = fftwf_plan_dft_c2r_3d(nx,ny,nz, cdata_13, data_13, FFTW_ESTIMATE),
		ip22  = fftwf_plan_dft_c2r_3d(nx,ny,nz, cdata_22, data_22, FFTW_ESTIMATE),
		ip23  = fftwf_plan_dft_c2r_3d(nx,ny,nz, cdata_23, data_23, FFTW_ESTIMATE),
		ip33  = fftwf_plan_dft_c2r_3d(nx,ny,nz, cdata_33, data_33, FFTW_ESTIMATE);
	
	fftwf_execute(plan);
	
	#else
	
	fftw_plan
		plan  = fftw_plan_dft_r2c_3d(nx,ny,nz, data, cdata, FFTW_ESTIMATE),
		iplan = fftw_plan_dft_c2r_3d(nx,ny,nz, cdata, data, FFTW_ESTIMATE),
		ip11  = fftw_plan_dft_c2r_3d(nx,ny,nz, cdata_11, data_11, FFTW_ESTIMATE),
		ip12  = fftw_plan_dft_c2r_3d(nx,ny,nz, cdata_12, data_12, FFTW_ESTIMATE),
		ip13  = fftw_plan_dft_c2r_3d(nx,ny,nz, cdata_13, data_13, FFTW_ESTIMATE),
		ip22  = fftw_plan_dft_c2r_3d(nx,ny,nz, cdata_22, data_22, FFTW_ESTIMATE),
		ip23  = fftw_plan_dft_c2r_3d(nx,ny,nz, cdata_23, data_23, FFTW_ESTIMATE),
		ip33  = fftw_plan_dft_c2r_3d(nx,ny,nz, cdata_33, data_33, FFTW_ESTIMATE);
	
	fftw_execute(plan);
	
	#endif
	
	double kfac = 2.0*M_PI;
	double norm = 1.0/((double)(nx*ny*nz));
	
	#pragma omp parallel for
	for( int i=0; i<(int)nx; ++i )
		for( size_t j=0; j<ny; ++j )	
			for( size_t l=0; l<nz/2+1; ++l )
			{
				int ii = i; if(ii>(int)nx/2) ii-=nx;
				int jj = (int)j; if(jj>(int)ny/2) jj-=ny;
				double ki = (double)ii;
				double kj = (double)jj;
				double kk = (double)l;
				
				double k[3];
				k[0] = (double)ki * kfac;
				k[1] = (double)kj * kfac;
				k[2] = (double)kk * kfac;
				
				size_t idx = ((size_t)i*ny+j)*nzp/2+l;
				//double re = cdata[idx][0];
				//double im = cdata[idx][1];
				
				cdata_11[idx][0] = -k[0]*k[0] * cdata[idx][0] * norm;
				cdata_11[idx][1] = -k[0]*k[0] * cdata[idx][1] * norm;
				
				cdata_12[idx][0] = -k[0]*k[1] * cdata[idx][0] * norm;
				cdata_12[idx][1] = -k[0]*k[1] * cdata[idx][1] * norm;
				
				cdata_13[idx][0] = -k[0]*k[2] * cdata[idx][0] * norm;
				cdata_13[idx][1] = -k[0]*k[2] * cdata[idx][1] * norm;
				
				cdata_22[idx][0] = -k[1]*k[1] * cdata[idx][0] * norm;
				cdata_22[idx][1] = -k[1]*k[1] * cdata[idx][1] * norm;
				
				cdata_23[idx][0] = -k[1]*k[2] * cdata[idx][0] * norm;
				cdata_23[idx][1] = -k[1]*k[2] * cdata[idx][1] * norm;
				
				cdata_33[idx][0] = -k[2]*k[2] * cdata[idx][0] * norm;
				cdata_33[idx][1] = -k[2]*k[2] * cdata[idx][1] * norm;
				
				
				if( i==(int)nx/2||j==ny/2||l==nz/2)
				{
					cdata_11[idx][0] = 0.0;
					cdata_11[idx][1] = 0.0;
					
					cdata_12[idx][0] = 0.0;
					cdata_12[idx][1] = 0.0;
					
					cdata_13[idx][0] = 0.0;
					cdata_13[idx][1] = 0.0;
					
					cdata_22[idx][0] = 0.0;
					cdata_22[idx][1] = 0.0;
					
					cdata_23[idx][0] = 0.0;
					cdata_23[idx][1] = 0.0;
					
					cdata_33[idx][0] = 0.0;
					cdata_33[idx][1] = 0.0;
				}
				
			}
	
	delete[] data;
	/*cdata_11[0][0]	= 0.0; cdata_11[0][1]	= 0.0;
	 cdata_12[0][0]	= 0.0; cdata_12[0][1]	= 0.0;
	 cdata_13[0][0]	= 0.0; cdata_13[0][1]	= 0.0;
	 cdata_22[0][0]	= 0.0; cdata_22[0][1]	= 0.0;
	 cdata_23[0][0]	= 0.0; cdata_23[0][1]	= 0.0;
	 cdata_33[0][0]	= 0.0; cdata_33[0][1]	= 0.0;*/
	
	
#ifdef SINGLE_PRECISION
	fftwf_execute(ip11);
	fftwf_execute(ip12);
	fftwf_execute(ip13);
	fftwf_execute(ip22);
	fftwf_execute(ip23);
	fftwf_execute(ip33);
	
	fftwf_destroy_plan(plan);
	fftwf_destroy_plan(iplan);
	fftwf_destroy_plan(ip11);
	fftwf_destroy_plan(ip12);
	fftwf_destroy_plan(ip13);
	fftwf_destroy_plan(ip22);
	fftwf_destroy_plan(ip23);
	fftwf_destroy_plan(ip33);
#else
	fftw_execute(ip11);
	fftw_execute(ip12);
	fftw_execute(ip13);
	fftw_execute(ip22);
	fftw_execute(ip23);
	fftw_execute(ip33);
	
	fftw_destroy_plan(plan);
	fftw_destroy_plan(iplan);
	fftw_destroy_plan(ip11);
	fftw_destroy_plan(ip12);
	fftw_destroy_plan(ip13);
	fftw_destroy_plan(ip22);
	fftw_destroy_plan(ip23);
	fftw_destroy_plan(ip33);

#endif
//#endif
	
	
#else
	rfftwnd_plan 
		plan = rfftw3d_create_plan( nx,ny,nz,
								   FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE|FFTW_IN_PLACE),
		iplan = rfftw3d_create_plan( nx,ny,nz,
									FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE|FFTW_IN_PLACE);
	
	
	#ifndef SINGLETHREAD_FFTW		
	rfftwnd_threads_one_real_to_complex( omp_get_max_threads(), plan, data, NULL );
	#else
	rfftwnd_one_real_to_complex( plan, data, NULL );
	#endif
//#endif
	//double fac = -1.0/(nx*ny*nz);
	double kfac = 2.0*M_PI;
	double norm = 1.0/((double)(nx*ny*nz));
	
	#pragma omp parallel for
	for( int i=0; i<(int)nx; ++i )
		for( size_t j=0; j<ny; ++j )	
			for( size_t l=0; l<nz/2+1; ++l )
			{
				int ii = (int)i; if(ii>(int)(nx/2)) ii-=(int)nx;
				int jj = (int)j; if(jj>(int)(ny/2)) jj-=(int)ny;
				double ki = (double)ii;
				double kj = (double)jj;
				double kk = (double)l;
				
				double k[3];
				k[0] = (double)ki * kfac;
				k[1] = (double)kj * kfac;
				k[2] = (double)kk * kfac;
				
				size_t idx = ((size_t)i*ny+j)*nzp/2+l;
				//double re = cdata[idx].re;
				//double im = cdata[idx].im;
				
				cdata_11[idx].re = -k[0]*k[0] * cdata[idx].re * norm;
				cdata_11[idx].im = -k[0]*k[0] * cdata[idx].im * norm;
				
				cdata_12[idx].re = -k[0]*k[1] * cdata[idx].re * norm;
				cdata_12[idx].im = -k[0]*k[1] * cdata[idx].im * norm;
				
				cdata_13[idx].re = -k[0]*k[2] * cdata[idx].re * norm;
				cdata_13[idx].im = -k[0]*k[2] * cdata[idx].im * norm;
				
				cdata_22[idx].re = -k[1]*k[1] * cdata[idx].re * norm;
				cdata_22[idx].im = -k[1]*k[1] * cdata[idx].im * norm;
				
				cdata_23[idx].re = -k[1]*k[2] * cdata[idx].re * norm;
				cdata_23[idx].im = -k[1]*k[2] * cdata[idx].im * norm;
				
				cdata_33[idx].re = -k[2]*k[2] * cdata[idx].re * norm;
				cdata_33[idx].im = -k[2]*k[2] * cdata[idx].im * norm;
				
				
				if( i==(int)(nx/2)||j==ny/2||l==nz/2)
				{
					cdata_11[idx].re = 0.0;
					cdata_11[idx].im = 0.0;
					
					cdata_12[idx].re = 0.0;
					cdata_12[idx].im = 0.0;
					
					cdata_13[idx].re = 0.0;
					cdata_13[idx].im = 0.0;
					
					cdata_22[idx].re = 0.0;
					cdata_22[idx].im = 0.0;
					
					cdata_23[idx].re = 0.0;
					cdata_23[idx].im = 0.0;
					
					cdata_33[idx].re = 0.0;
					cdata_33[idx].im = 0.0;
				}
				
			}
	
	delete[] data;
	/*cdata_11[0].re	= 0.0; cdata_11[0].im	= 0.0;
	cdata_12[0].re	= 0.0; cdata_12[0].im	= 0.0;
	cdata_13[0].re	= 0.0; cdata_13[0].im	= 0.0;
	cdata_22[0].re	= 0.0; cdata_22[0].im	= 0.0;
	cdata_23[0].re	= 0.0; cdata_23[0].im	= 0.0;
	cdata_33[0].re	= 0.0; cdata_33[0].im	= 0.0;*/
	
	
#ifndef SINGLETHREAD_FFTW		
	//rfftwnd_threads_one_complex_to_real( omp_get_max_threads(), iplan, cdata, NULL );
	rfftwnd_threads_one_complex_to_real( omp_get_max_threads(), iplan, cdata_11, NULL );
	rfftwnd_threads_one_complex_to_real( omp_get_max_threads(), iplan, cdata_12, NULL );
	rfftwnd_threads_one_complex_to_real( omp_get_max_threads(), iplan, cdata_13, NULL );
	rfftwnd_threads_one_complex_to_real( omp_get_max_threads(), iplan, cdata_22, NULL );
	rfftwnd_threads_one_complex_to_real( omp_get_max_threads(), iplan, cdata_23, NULL );
	rfftwnd_threads_one_complex_to_real( omp_get_max_threads(), iplan, cdata_33, NULL );
#else
	//rfftwnd_one_complex_to_real( iplan, cdata, NULL );
	rfftwnd_one_complex_to_real(iplan, cdata_11, NULL );
	rfftwnd_one_complex_to_real(iplan, cdata_12, NULL );
	rfftwnd_one_complex_to_real(iplan, cdata_13, NULL );
	rfftwnd_one_complex_to_real(iplan, cdata_22, NULL );
	rfftwnd_one_complex_to_real(iplan, cdata_23, NULL );
	rfftwnd_one_complex_to_real(iplan, cdata_33, NULL );
#endif
	
	
	
	rfftwnd_destroy_plan(plan);
	rfftwnd_destroy_plan(iplan);
#endif


	//... copy data ..........................................
	#pragma omp parallel for
	for( int i=0; i<(int)nx; ++i )
		for( size_t j=0; j<ny; ++j )	
			for( size_t k=0; k<nz; ++k )
			{
				size_t ii = ((size_t)i*ny+j)*nzp+k;
				(*fnew.get_grid(u.levelmax()))(i,j,k) = (( data_11[ii]*data_22[ii]-data_12[ii]*data_12[ii] ) +
														 ( data_11[ii]*data_33[ii]-data_13[ii]*data_13[ii] ) +
														 ( data_22[ii]*data_33[ii]-data_23[ii]*data_23[ii] ) );
				
				//(*fnew.get_grid(u.levelmax()))(i,j,k) = 
				
			}
	
	//delete[] data;
	delete[] data_11;
	delete[] data_12;
	delete[] data_13;
	delete[] data_23;
	delete[] data_22;
	delete[] data_33;

	const double t_lpt2_total = omp_get_wtime() - t_lpt2_start;
	LOGINFO("lpt2-profile compute_2LPT_source_FFT nx=%zu  wall=%.3fs  threads=%d",
	        nx, t_lpt2_total, omp_boost);
	if (omp_boost > omp_saved) {
		omp_set_num_threads(omp_saved);
#if defined(FFTW3) && !defined(SINGLETHREAD_FFTW)
#ifdef SINGLE_PRECISION
		fftwf_plan_with_nthreads(omp_saved);
#else
		fftw_plan_with_nthreads(omp_saved);
#endif
#endif
	}
}

void compute_2LPT_source( config_file& cf_, const grid_hierarchy& u, grid_hierarchy& fnew, unsigned order )
{
	// B2-2lpt: rank-0 only by call-site gating (same is_root() blocks as the FFT
	// variant). The level loop is OMP-parallel via the inner #pragma omp parallel
	// for over ix, so boosting rank-0's pool to claim idle worker cores cuts the
	// FD stencil cost. No FFTW plans here, so omp_set_num_threads alone suffices.
	const int omp_saved = omp_get_max_threads();
	int omp_boost = omp_saved;
	{
		const std::string boost_mode =
			cf_.getValueSafe<std::string>("setup", "lpt2_boost_threads",
			                              std::string("auto"));
		if (boost_mode != std::string("no")) {
			if (boost_mode == std::string("auto")) {
				const int avail = omp_get_num_procs();
				const int cap   = omp_saved * std::max(1, MUSIC::mpi::size());
				omp_boost = std::min(avail, cap);
			} else {
				try { omp_boost = std::max(1, std::stoi(boost_mode)); }
				catch(...) { omp_boost = omp_saved; }
			}
		}
	}
	if (omp_boost > omp_saved) {
		LOGINFO("B2-2lpt: rank-0 2LPT-FD OMP boost %d -> %d (workers idle in barrier)",
		        omp_saved, omp_boost);
		omp_set_num_threads(omp_boost);
	}
	const double t_lpt2_start = omp_get_wtime();

	fnew = u;
    fnew.zero();

	for( unsigned ilevel=u.levelmin(); ilevel<=u.levelmax(); ++ilevel )
	{
		double h = pow(2.0,ilevel), h2 = h*h, h2_4 = 0.25*h2;
		meshvar_bnd *pvar = fnew.get_grid(ilevel);
        
		if ( order == 2 )
		{
			
			#pragma omp parallel for
			for( int ix = 0; ix < (int)(*u.get_grid(ilevel)).size(0); ++ix )
			  for( int iy = 0; iy < (int)(*u.get_grid(ilevel)).size(1); ++iy )
			    for( int iz = 0; iz < (int)(*u.get_grid(ilevel)).size(2); ++iz )
			      {
				double D[3][3];
				
				D[0][0] = (ACC(ix-2,iy,iz)-2.0*ACC(ix,iy,iz)+ACC(ix+2,iy,iz)) * h2_4;
				D[1][1] = (ACC(ix,iy-2,iz)-2.0*ACC(ix,iy,iz)+ACC(ix,iy+2,iz)) * h2_4;
				D[2][2] = (ACC(ix,iy,iz-2)-2.0*ACC(ix,iy,iz)+ACC(ix,iy,iz+2)) * h2_4;
				
						
				D[0][1] = D[1][0] = (ACC(ix-1,iy-1,iz)-ACC(ix-1,iy+1,iz)-ACC(ix+1,iy-1,iz)+ACC(ix+1,iy+1,iz))*h2_4;
				D[0][2] = D[2][0] = (ACC(ix-1,iy,iz-1)-ACC(ix-1,iy,iz+1)-ACC(ix+1,iy,iz-1)+ACC(ix+1,iy,iz+1))*h2_4;
				D[1][2] = D[2][1] = (ACC(ix,iy-1,iz-1)-ACC(ix,iy-1,iz+1)-ACC(ix,iy+1,iz-1)+ACC(ix,iy+1,iz+1))*h2_4;
				
				(*pvar)(ix,iy,iz) =  ( D[0][0]*D[1][1] - D[0][1]*D[0][1]
						       + D[0][0]*D[2][2] - D[0][2]*D[0][2]
						       + D[1][1]*D[2][2] - D[1][2]*D[1][2] );
						
					}
		}
		else if ( order == 4 || order == 6 )
		{
			double h2_144 = h2 / 144.;
                        #pragma omp parallel for 
			for( int ix = 0; ix < (int)(*u.get_grid(ilevel)).size(0); ++ix )
			  for( int iy = 0; iy < (int)(*u.get_grid(ilevel)).size(1); ++iy )
			    for( int iz = 0; iz < (int)(*u.get_grid(ilevel)).size(2); ++iz )
			      {
				//.. this is actually 8th order accurate
				
				double D[3][3];

				D[0][0] = ((ACC(ix-4,iy,iz)+ACC(ix+4,iy,iz))
					   - 16. * (ACC(ix-3,iy,iz)+ACC(ix+3,iy,iz))
					   + 64. * (ACC(ix-2,iy,iz)+ACC(ix+2,iy,iz))
					   + 16. * (ACC(ix-1,iy,iz)+ACC(ix+1,iy,iz))
					   - 130.*  ACC(ix,iy,iz) ) * h2_144;
				
				D[1][1] = ((ACC(ix,iy-4,iz)+ACC(ix,iy+4,iz))
					   - 16. * (ACC(ix,iy-3,iz)+ACC(ix,iy+3,iz))
					   + 64. * (ACC(ix,iy-2,iz)+ACC(ix,iy+2,iz))
					   + 16. * (ACC(ix,iy-1,iz)+ACC(ix,iy+1,iz))
					   - 130.*  ACC(ix,iy,iz) ) * h2_144;
				
				D[2][2] = ((ACC(ix,iy,iz-4)+ACC(ix,iy,iz+4))
					   - 16. * (ACC(ix,iy,iz-3)+ACC(ix,iy,iz+3))
					   + 64. * (ACC(ix,iy,iz-2)+ACC(ix,iy,iz+2))
					   + 16. * (ACC(ix,iy,iz-1)+ACC(ix,iy,iz+1))
					   - 130.*  ACC(ix,iy,iz) ) * h2_144;
                        
                        
				D[0][1] = D[1][0] = (64.*(ACC(ix-1,iy-1,iz)-ACC(ix-1,iy+1,iz)-ACC(ix+1,iy-1,iz)+ACC(ix+1,iy+1,iz))
						     -8.*(ACC(ix-2,iy-1,iz)-ACC(ix+2,iy-1,iz)-ACC(ix-2,iy+1,iz)+ACC(ix+2,iy+1,iz)
							  + ACC(ix-1,iy-2,iz)-ACC(ix-1,iy+2,iz)-ACC(ix+1,iy-2,iz)+ACC(ix+1,iy+2,iz))
						     +1.*(ACC(ix-2,iy-2,iz)-ACC(ix-2,iy+2,iz)-ACC(ix+2,iy-2,iz)+ACC(ix+2,iy+2,iz)))*h2_144;
				D[0][2] = D[2][0] = (64.*(ACC(ix-1,iy,iz-1)-ACC(ix-1,iy,iz+1)-ACC(ix+1,iy,iz-1)+ACC(ix+1,iy,iz+1))
						     -8.*(ACC(ix-2,iy,iz-1)-ACC(ix+2,iy,iz-1)-ACC(ix-2,iy,iz+1)+ACC(ix+2,iy,iz+1)
							  + ACC(ix-1,iy,iz-2)-ACC(ix-1,iy,iz+2)-ACC(ix+1,iy,iz-2)+ACC(ix+1,iy,iz+2))
						     +1.*(ACC(ix-2,iy,iz-2)-ACC(ix-2,iy,iz+2)-ACC(ix+2,iy,iz-2)+ACC(ix+2,iy,iz+2)))*h2_144;
				D[1][2] = D[2][1] = (64.*(ACC(ix,iy-1,iz-1)-ACC(ix,iy-1,iz+1)-ACC(ix,iy+1,iz-1)+ACC(ix,iy+1,iz+1))
						     -8.*(ACC(ix,iy-2,iz-1)-ACC(ix,iy+2,iz-1)-ACC(ix,iy-2,iz+1)+ACC(ix,iy+2,iz+1)
							  + ACC(ix,iy-1,iz-2)-ACC(ix,iy-1,iz+2)-ACC(ix,iy+1,iz-2)+ACC(ix,iy+1,iz+2))
						     +1.*(ACC(ix,iy-2,iz-2)-ACC(ix,iy-2,iz+2)-ACC(ix,iy+2,iz-2)+ACC(ix,iy+2,iz+2)))*h2_144;
				
				(*pvar)(ix,iy,iz) =  ( D[0][0]*D[1][1] - SQR( D[0][1] )
						       + D[0][0]*D[2][2] - SQR( D[0][2] )
						       + D[1][1]*D[2][2] - SQR( D[1][2] ) );
						
			      }
			
			
		}
		else
			throw std::runtime_error("compute_2LPT_source : invalid operator order specified");


	}
	
    	//.. subtract global mean so the multi-grid poisson solver behaves well
	
	for( int i=fnew.levelmax(); i>(int)fnew.levelmin(); --i )
	  mg_straight().restrict( (*fnew.get_grid(i)), (*fnew.get_grid(i-1)) );
	
	long double sum = 0.0;
	int nx,ny,nz;
	
	nx = fnew.get_grid(fnew.levelmin())->size(0);
	ny = fnew.get_grid(fnew.levelmin())->size(1);
	nz = fnew.get_grid(fnew.levelmin())->size(2);
	
	for( int ix=0; ix<nx; ++ix )
	  for( int iy=0; iy<ny; ++iy )
	    for( int iz=0; iz<nz; ++iz )
	      sum += (*fnew.get_grid(fnew.levelmin()))(ix,iy,iz);
	
	sum /= (double)((size_t)nx*(size_t)ny*(size_t)nz);
	
	for( unsigned i=fnew.levelmin(); i<=fnew.levelmax(); ++i )
	{		
		nx = fnew.get_grid(i)->size(0);
		ny = fnew.get_grid(i)->size(1);
		nz = fnew.get_grid(i)->size(2);
		
		for( int ix=0; ix<nx; ++ix )
		  for( int iy=0; iy<ny; ++iy )
		    for( int iz=0; iz<nz; ++iz )
		      (*fnew.get_grid(i))(ix,iy,iz) -= sum;
	}

	const double t_lpt2_total = omp_get_wtime() - t_lpt2_start;
	LOGINFO("lpt2-profile compute_2LPT_source (FD) levels=%u..%u  wall=%.3fs  threads=%d",
	        u.levelmin(), u.levelmax(), t_lpt2_total, omp_boost);
	if (omp_boost > omp_saved) {
		omp_set_num_threads(omp_saved);
	}
}
#undef SQR
#undef ACC

// ---------------------------------------------------------------------------
// Task #155: memory-first distributed twin of compute_2LPT_source.
//
// Reproduces the three stages of compute_2LPT_source — (1) per-level FD source
// stencil, (2) cross-level fine->coarse restrict, (3) global-mean subtraction —
// but each stage runs collectively across ranks operating on the per-box
// MeshvarBnd hierarchy, so the rank-0 union mesh never has to hold the whole
// 2LPT source. All ranks MUST call this OUTSIDE phase_scope (the bridges and
// the parent-consolidation primitives are SPMD/collective).
//
// Bit-identical to compute_2LPT_source for the single-box-per-level case
// (zoom / unigrid), where each per-box mesh IS the union mesh including its
// ghost perimeter. At np<=1 it delegates straight to the serial function.
void compute_2LPT_source_distributed( config_file& cf_, const grid_hierarchy& u, grid_hierarchy& fnew, unsigned order )
{
#ifdef USE_MPI
	if( MUSIC::mpi::size() <= 1 ){
		compute_2LPT_source( cf_, u, fnew, order );
		return;
	}

	const double t_lpt2_start = omp_get_wtime();
	MPI_Comm world = MUSIC::mpi::world();
	#define L2T(...) do{}while(0)

	// Match the serial seed: fnew = u; fnew.zero(). The copy-assign deep-copies
	// the per-box structure (owner meshes only), so workers get the per-box
	// layout they need for the bridges below.
	fnew = u;
	fnew.zero();

	// Loop bounds MUST be byte-identical on every rank: the slab bridges and the
	// parent-consolidation primitives below are collective on `world`, so any
	// divergence in the level range deadlocks. Worker hierarchies may carry an
	// empty union m_pgrids (levelmax() == m_pgrids.size()-1 underflows to
	// UINT_MAX), so broadcast the authoritative structure from rank 0 (the
	// levelmin box-0 owner under b%nproc) and drive ALL loops from it.
	unsigned lmin_b = 0, lmax_b = 0;
	if( MUSIC::mpi::rank()==0 ){ lmin_b = u.levelmin(); lmax_b = u.levelmax(); }
	MPI_Bcast( &lmin_b, 1, MPI_UNSIGNED, 0, world );
	MPI_Bcast( &lmax_b, 1, MPI_UNSIGNED, 0, world );
	std::vector<unsigned long long> nbox( (size_t)lmax_b+1, 1ULL );
	if( MUSIC::mpi::rank()==0 )
		for( unsigned L=lmin_b; L<=lmax_b; ++L )
			nbox[L] = (unsigned long long)u.num_boxes(L);
	MPI_Bcast( nbox.data(), (int)((size_t)lmax_b+1), MPI_UNSIGNED_LONG_LONG, 0, world );

	// Storage convention (IDENTICAL to twoGrid_multibox_spmd / mg_solver):
	//   level L is "multi-box" iff num_boxes(L) > 1 -> operate on per-box meshes
	//   get_grid(L,b); otherwise operate on the single union mesh get_grid(L).
	// For the zoom / unigrid (single-box-per-level) case this reduces to pure
	// union, so stages 1-3 below are the SAME data flow as the serial function
	// with only the FD and restrict COMPUTE distributed across the slab bridges
	// -> trivially bit-identical.

	// === Stage 1: per-level FD source via the lpt2_fd_meshvarbnd z-slab bridge ===
	for( unsigned ilevel=lmin_b; ilevel<=lmax_b; ++ilevel )
	{
		const double h = pow(2.0,ilevel);
		const bool   multi = ( nbox[ilevel] > 1ULL );
		const size_t nb    = multi ? (size_t)nbox[ilevel] : 1;
		for( size_t b=0; b<nb; ++b )
		{
			const int  owner = multi ? u.owner_of_box(ilevel,b)
			                         : u.owner_of_box(ilevel,0);
			const bool mine  = ( MUSIC::mpi::rank()==owner );
			const MeshvarBnd<real_t>* ub = !mine ? (const MeshvarBnd<real_t>*)NULL
				: ( multi ? u.get_grid(ilevel,b) : u.get_grid(ilevel) );
			MeshvarBnd<real_t>* fb = !mine ? (MeshvarBnd<real_t>*)NULL
				: ( multi ? fnew.get_grid(ilevel,b) : fnew.get_grid(ilevel) );
			L2T("S1 enter ilevel=%u b=%zu owner=%d multi=%d", ilevel, b, owner, (int)multi);
			const bool did = MUSIC::zoom_slab::lpt2_fd_meshvarbnd(
				owner, ub, fb, h, order, world );
			L2T("S1 done  ilevel=%u b=%zu", ilevel, b);
			if( !did ){
				// sub_size==1 is impossible here (world size>1); a false return
				// signals a real geometry violation, not a benign fallback.
				LOGERR("compute_2LPT_source_distributed: lpt2_fd_meshvarbnd "
				       "returned false at ilevel=%u box=%zu (np=%d)",
				       ilevel, b, MUSIC::mpi::size());
				throw std::runtime_error(
					"compute_2LPT_source_distributed: FD bridge failed");
			}
		}
	}

	// === Stage 2: restrict fine source into parents (mirrors mg_straight().
	// restrict over the whole hierarchy). Dual parent-storage convention,
	// IDENTICAL to twoGrid_multibox_spmd:
	//   - multi-box parent : restrict into the per-box parent (local OR replica
	//     from broadcast_parents_to_child_owners), then ship disjoint footprints
	//     to the parent box owner with overwrite_children_to_parents (OVERWRITE).
	//   - single-box parent: restrict into the child OWNER's local copy of the
	//     UNION mesh (get_grid(L-1)); consolidate_child_writes_to_parent_owner
	//     (P2P, world-collective-neutral) gathers footprints onto parent_owner's
	//     union. The next (coarser) iteration's restrict reads get_grid(ci-1)
	//     ONLY on its owner, and for single-box levels that owner is always
	//     parent_owner==rank 0 (owner_of_box(L,0) = 0 under b%nproc), which
	//     consolidate already left current -> NO broadcast_union_mesh_from_owner.
	//     (That primitive early-returns on workers whose union m_pgrids is empty,
	//     so calling it here would issue 2 Bcasts on rank 0 against 0 on workers
	//     -> a world-collective desync that corrupts the next solve's op stream.)
	for( int i=(int)lmax_b; i>(int)lmin_b; --i )
	{
		const unsigned ci = (unsigned)i;
		const bool fine_multi   = ( nbox[ci]   > 1ULL );
		const bool parent_multi = ( nbox[ci-1] > 1ULL );

		L2T("S2 iter ci=%u fine_multi=%d parent_multi=%d", ci, (int)fine_multi, (int)parent_multi);
		if( parent_multi )
			fnew.broadcast_parents_to_child_owners( ci );
		else
			fnew.acquire_parent_union_scratch( ci );
		L2T("S2 post-acquire ci=%u", ci);

		const size_t nb = fine_multi ? (size_t)nbox[ci] : 1;
		for( size_t b=0; b<nb; ++b )
		{
			const int owner = fine_multi ? fnew.owner_of_box( ci, b )
			                             : fnew.owner_of_box( ci, 0 );
			const bool mine = ( MUSIC::mpi::rank()==owner );
			MeshvarBnd<real_t>* vf = !mine ? (MeshvarBnd<real_t>*)NULL
				: ( fine_multi ? fnew.get_grid( ci, b ) : fnew.get_grid( ci ) );
			MeshvarBnd<real_t>* Vc = NULL;
			if( mine )
				Vc = parent_multi ? fnew.parent_for_box( ci, b )
				                  : fnew.get_grid( ci-1 );   // UNION mesh
			// restrict_meshvarbnd is collective-symmetric (1 hdr Bcast then a
			// deterministic verdict on every rank). When it returns false (odd
			// dims or nzf not divisible by 2*np), the distributed path is
			// geometrically ineligible: the OWNER must perform the restrict
			// locally to honour the documented contract. mg_straight().restrict
			// is the SAME operator the serial compute_2LPT_source uses (line
			// 730), so the fallback is bit-identical.
			L2T("S2 restrict ci=%u b=%zu owner=%d mine=%d", ci, b, owner, (int)mine);
			const bool rdid = MUSIC::zoom_slab::restrict_meshvarbnd( owner, vf, Vc, world );
			if( !rdid && mine )
				mg_straight().restrict( *vf, *Vc );
			L2T("S2 restrict done ci=%u b=%zu rdid=%d", ci, b, (int)rdid);
		}

		L2T("S2 consolidate ci=%u", ci);
		if( parent_multi )
			fnew.overwrite_children_to_parents( ci );
		else {
			fnew.consolidate_child_writes_to_parent_owner( ci );
			fnew.release_parent_union_scratch( ci );
		}
		L2T("S2 iter-done ci=%u", ci);
	}

	// === Stage 3: deterministic global mean over levelmin, subtract everywhere ===
	const unsigned lmin = lmin_b;
	const bool lmin_multi = ( nbox[lmin] > 1ULL );
	long double local_sum = 0.0L;
	size_t      local_cnt = 0;
	{
		const size_t nb = lmin_multi ? (size_t)nbox[lmin] : 1;
		for( size_t b=0; b<nb; ++b )
		{
			const int owner = lmin_multi ? fnew.owner_of_box( lmin, b )
			                             : fnew.owner_of_box( lmin, 0 );
			if( MUSIC::mpi::rank() != owner ) continue;
			MeshvarBnd<real_t>* m = lmin_multi ? fnew.get_grid( lmin, b )
			                                   : fnew.get_grid( lmin );
			if( !m ) continue;
			const int nx=(int)m->size(0), ny=(int)m->size(1), nz=(int)m->size(2);
			for( int ix=0; ix<nx; ++ix )
				for( int iy=0; iy<ny; ++iy )
					for( int iz=0; iz<nz; ++iz )
						local_sum += (*m)(ix,iy,iz);
			local_cnt += (size_t)nx*(size_t)ny*(size_t)nz;
		}
	}
	long double global_sum = 0.0L;
	unsigned long long local_cnt_ull = (unsigned long long)local_cnt, global_cnt_ull = 0;
	MPI_Allreduce( &local_sum, &global_sum, 1, MPI_LONG_DOUBLE, MPI_SUM, world );
	MPI_Allreduce( &local_cnt_ull, &global_cnt_ull, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, world );

	long double mean = ( global_cnt_ull > 0 )
		? ( global_sum / (long double)((double)(size_t)global_cnt_ull) )
		: 0.0L;

	for( unsigned il=lmin_b; il<=lmax_b; ++il )
	{
		const bool   il_multi = ( nbox[il] > 1ULL );
		const size_t nb = il_multi ? (size_t)nbox[il] : 1;
		for( size_t b=0; b<nb; ++b )
		{
			const int owner = il_multi ? fnew.owner_of_box( il, b )
			                           : fnew.owner_of_box( il, 0 );
			if( MUSIC::mpi::rank() != owner ) continue;
			MeshvarBnd<real_t>* m = il_multi ? fnew.get_grid( il, b )
			                                 : fnew.get_grid( il );
			if( !m ) continue;
			const int nx=(int)m->size(0), ny=(int)m->size(1), nz=(int)m->size(2);
			#pragma omp parallel for
			for( int ix=0; ix<nx; ++ix )
				for( int iy=0; iy<ny; ++iy )
					for( int iz=0; iz<nz; ++iz )
						(*m)(ix,iy,iz) -= mean;
		}
	}

	const double t_lpt2_total = omp_get_wtime() - t_lpt2_start;
	LOGINFO("lpt2-profile compute_2LPT_source_distributed levels=%u..%u  "
	        "wall=%.3fs  np=%d  mean=%.6Le",
	        lmin_b, lmax_b, t_lpt2_total, MUSIC::mpi::size(),
	        mean);
	#undef L2T
#else
	compute_2LPT_source( cf_, u, fnew, order );
#endif
}

