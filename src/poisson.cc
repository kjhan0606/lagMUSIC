/*
 
 poisson.cc - This file is part of MUSIC -
 a code to generate multi-scale initial conditions 
 for cosmological simulations 
 
 Copyright (C) 2010  Oliver Hahn
 
 */

/****** ABSTRACT FACTORY PATTERN IMPLEMENTATION *******/

#include "poisson.hh"
#include "Numerics.hh"
#include "mpi_helper.hh"
#include "mpi_poisson.hh"
#include "mesh.hh"
#include "zoom_slab.hh"
#include "poisson_hybrid_kernel.hh"

// Rank-0 periodic wrap on a full unigrid MeshvarBnd. Lives in MUSIC::poisson
// so the SPMD slab path in mpi_poisson.hh can call it after gathering the
// slab back to rank 0. Existing fft_poisson_plugin::solve still calls it
// from the full-path branch (extracted from the original inline BC loops).
namespace MUSIC { namespace poisson {

template<typename real_t>
void apply_periodic_bc_unigrid_rank0( MeshvarBnd<real_t>& u, int nx, int ny, int nz )
{
    const int nb = u.m_nbnd;
    for( int iy=-nb; iy<ny+nb; ++iy )
        for( int iz=-nb; iz<nz+nb; ++iz )
        {
            const int iiy( (iy+ny)%ny ), iiz( (iz+nz)%nz );
            for( int i=-nb; i<0; ++i )
            {
                u(i,iy,iz)       = u(nx+i,iiy,iiz);
                u(nx-1-i,iy,iz)  = u(-1-i,iiy,iiz);
            }
        }
    for( int ix=-nb; ix<nx+nb; ++ix )
        for( int iz=-nb; iz<nz+nb; ++iz )
        {
            const int iix( (ix+nx)%nx ), iiz( (iz+nz)%nz );
            for( int i=-nb; i<0; ++i )
            {
                u(ix,i,iz)       = u(iix,ny+i,iiz);
                u(ix,ny-1-i,iz)  = u(iix,-1-i,iiz);
            }
        }
    for( int ix=-nb; ix<nx+nb; ++ix )
        for( int iy=-nb; iy<ny+nb; ++iy )
        {
            const int iix( (ix+nx)%nx ), iiy( (iy+ny)%ny );
            for( int i=-nb; i<0; ++i )
            {
                u(ix,iy,i)       = u(iix,iiy,nz+i);
                u(ix,iy,nz-1-i)  = u(iix,iiy,-1-i);
            }
        }
}

#ifdef SINGLE_PRECISION
template void apply_periodic_bc_unigrid_rank0<float >( MeshvarBnd<float >&, int, int, int );
#else
template void apply_periodic_bc_unigrid_rank0<double>( MeshvarBnd<double>&, int, int, int );
#endif

}} // namespace MUSIC::poisson

std::map< std::string, poisson_plugin_creator *>& 
get_poisson_plugin_map()
{
	static std::map< std::string, poisson_plugin_creator* > poisson_plugin_map;
	return poisson_plugin_map;
}

void print_poisson_plugins()
{
	std::map< std::string, poisson_plugin_creator *>& m = get_poisson_plugin_map();
	
	std::map< std::string, poisson_plugin_creator *>::iterator it;
	it = m.begin();
	std::cout << " - Available poisson solver plug-ins:\n";
	while( it!=m.end() )
	{
		if( (*it).second )
			std::cout << "\t\'" << (*it).first << "\'\n";
		
		LOGINFO("Poisson plug-in :: %s",std::string((*it).first).c_str());
		
		++it;
	}
}


/****** CALL IMPLEMENTATIONS OF POISSON SOLVER CLASSES ******/

#include "mg_solver.hh"
#include "fd_schemes.hh"

#ifdef SINGLE_PRECISION
typedef multigrid::solver< stencil_7P<float>, interp_O3_fluxcorr, mg_straight, float > poisson_solver_O2;
typedef multigrid::solver< stencil_13P<float>, interp_O5_fluxcorr, mg_straight, float > poisson_solver_O4;
typedef multigrid::solver< stencil_19P<float>, interp_O7_fluxcorr, mg_straight, float > poisson_solver_O6;
#else
typedef multigrid::solver< stencil_7P<double>, interp_O3_fluxcorr, mg_straight, double > poisson_solver_O2;
typedef multigrid::solver< stencil_13P<double>, interp_O5_fluxcorr, mg_straight, double > poisson_solver_O4;
typedef multigrid::solver< stencil_19P<double>, interp_O7_fluxcorr, mg_straight, double > poisson_solver_O6;
#endif


/**************************************************************************************/
/**************************************************************************************/


double multigrid_poisson_plugin::solve( grid_hierarchy& f, grid_hierarchy& u )
{
	LOGUSER("Initializing multi-grid Poisson solver...");
	
	unsigned verbosity = cf_.getValueSafe<unsigned>("setup","verbosity",2);
	
	if( verbosity > 0 )
	{	
		std::cout << "-------------------------------------------------------------\n";
		std::cout << " - Invoking multi-grid Poisson solver..." << std::endl;
	}
	
	double acc = 1e-5, err;
	std::string ps_smoother_name;
	unsigned ps_presmooth, ps_postsmooth, order;
	
	acc                 = cf_.getValueSafe<double>("poisson","accuracy",acc);
	ps_presmooth		= cf_.getValueSafe<unsigned>("poisson","pre_smooth",3);
	ps_postsmooth		= cf_.getValueSafe<unsigned>("poisson","post_smooth",3);
	ps_smoother_name	= cf_.getValueSafe<std::string>("poisson","smoother","gs");
	order				= cf_.getValueSafe<unsigned>( "poisson", "laplace_order", 4 );

	// Phase G.2b A': opt-in toggle for the z-slab bridge GS smoother in
	// multigrid::solver::twoGrid_multibox (per-box GS sites). Default false →
	// production GaussSeidel path unchanged. Bridge is only invoked when also
	// S::order==2 (stencil_7P) and the GridHierarchy has z-slabs populated.
	{
		bool zsm = cf_.getValueSafe<bool>("setup", "zoom_slab_smoother", false);
		MUSIC::zoom_slab::set_smoother_enabled(zsm);
		if( zsm )
			LOGINFO("G.2b: zoom_slab_smoother=yes order=%u (bridge fires at multibox levels when order==2 and (np==1 or zoom_slab_subcomm==self))",
			        order);
	}

	// Phase G.2b B.2b.2: opt-in toggle for the SPMD composite-MG path inside
	// twoGrid_multibox. Default false (current rank-0-only V-cycle preserved).
	{
		bool zsmg = cf_.getValueSafe<bool>("setup", "zoom_slab_spmd_multigrid", false);
		MUSIC::zoom_slab::set_spmd_multigrid_enabled(zsmg);
		if( zsmg )
			LOGINFO("G.2b B.2b.2: zoom_slab_spmd_multigrid=yes (twoGrid_multibox routes through SPMD path with B.2b.1 parent broadcast/accumulate)");
	}

	// Phase G.2b B.5.4.a: opt-in toggle for keep-in-slab pre/post smoothing.
	// Default false. When yes, each pre/post-smooth phase scatters uf+ff once
	// per box and runs N×(interp_cf + gs) on a shared (size+4)^3 slab buffer.
	{
		bool kss = cf_.getValueSafe<bool>("setup", "zoom_slab_keep_slab_smooth", false);
		MUSIC::zoom_slab::set_keep_slab_smooth_enabled(kss);
		if( kss )
			LOGINFO("G.2b B.5.4.a: zoom_slab_keep_slab_smooth=yes (pre/post smoothing keeps uf/ff in z-slab across N iters)");
	}

	// Phase G.2b B.5.4.b: opt-in toggle for keep-in-slab u-restrict (chains the
	// padded-cluster slab from B.5.4.a's pre-smooth into u-restrict via
	// restrict_meshvarbnd_from_padded_slab). Requires B.5.4.a to also be on.
	{
		bool ksu = cf_.getValueSafe<bool>("setup", "zoom_slab_keep_slab_urestrict", false);
		MUSIC::zoom_slab::set_keep_slab_urestrict_enabled(ksu);
		if( ksu )
			LOGINFO("G.2b B.5.4.b: zoom_slab_keep_slab_urestrict=yes (pre-smooth → u-restrict shares padded-cluster slab; saves 1 gather+scatter per box per V-cycle)");
	}

	// Phase G.2b B.5.4.c: opt-in toggle for fused keep-in-slab prolong_add
	// (fuses the post-coarse prolong_add with the post-smooth into one
	// scatter/gather via prolong_add_then_smooth_n_meshvarbnd). Requires
	// B.5.4.a to also be on.
	{
		bool ksp = cf_.getValueSafe<bool>("setup", "zoom_slab_keep_slab_prolong", false);
		MUSIC::zoom_slab::set_keep_slab_prolong_enabled(ksp);
		if( ksp )
			LOGINFO("G.2b B.5.4.c: zoom_slab_keep_slab_prolong=yes (post-coarse prolong_add → post-smooth fused; saves 1 scatter+gather per box per V-cycle)");
	}

	multigrid::opt::smtype ps_smtype = multigrid::opt::sm_gauss_seidel;
	
	if ( ps_smoother_name == std::string("gs") )
	{	
		ps_smtype = multigrid::opt::sm_gauss_seidel;
		LOGUSER("Selected Gauss-Seidel multigrid smoother");
	}
	else if ( ps_smoother_name == std::string("jacobi") )
	{	
		ps_smtype = multigrid::opt::sm_jacobi;
		LOGUSER("Selected Jacobi multigrid smoother");	
	}
	else if ( ps_smoother_name == std::string("sor") )
	{	
		ps_smtype = multigrid::opt::sm_sor;
		LOGUSER("Selected SOR multigrid smoother");
	}
	else
	{	
		LOGWARN("Unknown multigrid smoother \'%s\' specified. Reverting to Gauss-Seidel.",ps_smoother_name.c_str());
		std::cerr << " - Warning: unknown smoother \'" << ps_smoother_name << "\' for multigrid solver!\n"
			<< "            reverting to \'gs\' (Gauss-Seidel)" << std::endl;
	}
		
	
	
	double tstart, tend;
	
#ifndef SINGLETHREAD_FFTW
	tstart = omp_get_wtime();
#else
	tstart = (double)clock() / CLOCKS_PER_SEC;
#endif
	
	
	//----- run Poisson solver -----//
	if( order == 2 )
	{
		LOGUSER("Running multigrid solver with 2nd order Laplacian...");
		poisson_solver_O2 ps( f, ps_smtype, ps_presmooth, ps_postsmooth );
		err = ps.solve( u, acc, true );
	}
	else if( order == 4 )
	{
		LOGUSER("Running multigrid solver with 4th order Laplacian...");
		poisson_solver_O4 ps( f, ps_smtype, ps_presmooth, ps_postsmooth );
		err = ps.solve( u, acc, true );	
	}
	else if( order == 6 )
	{
		LOGUSER("Running multigrid solver with 6th order Laplacian..");
		poisson_solver_O6 ps( f, ps_smtype, ps_presmooth, ps_postsmooth );
		err = ps.solve( u, acc, true );	
	}	
	else
	{	
		LOGERR("Invalid order specified for Laplace operator");
		throw std::runtime_error("Invalid order specified for Laplace operator");
	}

	// D.3.3 (composite multigrid, dispatch enabled): per-box u is the
	// source of truth at multi-box levels. No sync to union here — the
	// gradient operator now reads per-box at multi-box levels and writes
	// per-box Du. (Union u at multi-box levels is left stale; nothing
	// reads it on the compute path.)

	//------------------------------//
	
#ifndef SINGLETHREAD_FFTW
	tend = omp_get_wtime();
	if( verbosity > 1 )
		std::cout << " - Poisson solver took " << tend-tstart << "s with " << omp_get_max_threads() << " threads." << std::endl;
#else
	tend = (double)clock() / CLOCKS_PER_SEC;
	if( verbosity > 1 )
		std::cout << " - Poisson solver took " << tend-tstart << "s." << std::endl;
	
#endif
	
	
	return err;
}

double multigrid_poisson_plugin::gradient( int dir, grid_hierarchy& u, grid_hierarchy& Du )
{
	Du = u;

	unsigned order = cf_.getValueSafe<unsigned>( "poisson", "grad_order", 4 );

	switch( order )
	{
		case 2:
			implementation().gradient_O2( dir, u, Du );
			break;
		case 4:
			implementation().gradient_O4( dir, u, Du );
			break;
		case 6:
			implementation().gradient_O6( dir, u, Du );
			break;
		default:
			LOGERR("Invalid order %d specified for gradient operator", order);
			throw std::runtime_error("Invalid order specified for gradient operator!");
	}

	// D.3.3 (transitional, pre-D.4): gradient writes per-box Du at multi-box
	// levels; mirror per-box cluster cells back to union so that output
	// plugins which still iterate union see correct values. Output gating
	// on is_in_mask skips gap cells.
	Du.sync_union_from_per_box();

	return 0.0;
}

double multigrid_poisson_plugin::gradient_add( int dir, grid_hierarchy& u, grid_hierarchy& Du )
{
	//Du = u;

	unsigned order = cf_.getValueSafe<unsigned>( "poisson", "grad_order", 4 );

	// D.3.3 proper: callers fill the hybrid Poisson correction into the
	// union finest level (see main.cc data_forIO pattern). Mirror that into
	// per-box first so the per-box gradient_add_OX kernels accumulate on
	// top of the same correction.
	Du.sync_per_box_from_union();

	switch( order )
	{
		case 2:
			implementation().gradient_add_O2( dir, u, Du );
			break;
		case 4:
			implementation().gradient_add_O4( dir, u, Du );
			break;
		case 6:
			implementation().gradient_add_O6( dir, u, Du );
			break;
		default:
			LOGERR("Invalid order %d specified for gradient operator!",order);
			throw std::runtime_error("Invalid order specified for gradient operator!");
	}

	// D.3.3 (transitional, pre-D.4): mirror per-box Du back to union.
	Du.sync_union_from_per_box();

	return 0.0;
}

void multigrid_poisson_plugin::implementation::gradient_O2( int dir, grid_hierarchy& u, grid_hierarchy& Du )
{
	LOGUSER("Computing a 2nd order finite difference gradient...");

	for( unsigned ilevel=u.levelmin(); ilevel<=u.levelmax(); ++ilevel )
	{
		double h = pow(2.0,ilevel);
		const size_t nb = u.num_boxes(ilevel);
		const bool multi = (nb > 1);

		for( size_t b=0; b < (multi ? nb : 1); ++b ){
			meshvar_bnd *pu   = multi ? u.get_grid(ilevel, b)  : u.get_grid(ilevel);
			meshvar_bnd *pvar = multi ? Du.get_grid(ilevel, b) : Du.get_grid(ilevel);

			if( dir == 0 )
#pragma omp parallel for
				for( int ix = 0; ix < (int)pu->size(0); ++ix )
					for( int iy = 0; iy < (int)pu->size(1); ++iy )
						for( int iz = 0; iz < (int)pu->size(2); ++iz )
							(*pvar)(ix,iy,iz) = 0.5*((*pu)(ix+1,iy,iz)-(*pu)(ix-1,iy,iz))*h;

			else if( dir == 1 )
#pragma omp parallel for
				for( int ix = 0; ix < (int)pu->size(0); ++ix )
					for( int iy = 0; iy < (int)pu->size(1); ++iy )
						for( int iz = 0; iz < (int)pu->size(2); ++iz )
							(*pvar)(ix,iy,iz) = 0.5*((*pu)(ix,iy+1,iz)-(*pu)(ix,iy-1,iz))*h;

			else if( dir == 2 )
#pragma omp parallel for
				for( int ix = 0; ix < (int)pu->size(0); ++ix )
					for( int iy = 0; iy < (int)pu->size(1); ++iy )
						for( int iz = 0; iz < (int)pu->size(2); ++iz )
							(*pvar)(ix,iy,iz) = 0.5*((*pu)(ix,iy,iz+1)-(*pu)(ix,iy,iz-1))*h;
		}
	}

	LOGUSER("Done computing a 2nd order finite difference gradient.");
}

void multigrid_poisson_plugin::implementation::gradient_add_O2( int dir, grid_hierarchy& u, grid_hierarchy& Du )
{
	LOGUSER("Computing a 2nd order finite difference gradient...");

	for( unsigned ilevel=u.levelmin(); ilevel<=u.levelmax(); ++ilevel )
	{
		double h = pow(2.0,ilevel);
		const size_t nb = u.num_boxes(ilevel);
		const bool multi = (nb > 1);

		for( size_t b=0; b < (multi ? nb : 1); ++b ){
			meshvar_bnd *pu   = multi ? u.get_grid(ilevel, b)  : u.get_grid(ilevel);
			meshvar_bnd *pvar = multi ? Du.get_grid(ilevel, b) : Du.get_grid(ilevel);

			if( dir == 0 )
				#pragma omp parallel for
				for( int ix = 0; ix < (int)pvar->size(0); ++ix )
					for( int iy = 0; iy < (int)pvar->size(1); ++iy )
						for( int iz = 0; iz < (int)pvar->size(2); ++iz )
							(*pvar)(ix,iy,iz) += 0.5*((*pu)(ix+1,iy,iz)-(*pu)(ix-1,iy,iz))*h;

			else if( dir == 1 )
				#pragma omp parallel for
				for( int ix = 0; ix < (int)pvar->size(0); ++ix )
					for( int iy = 0; iy < (int)pvar->size(1); ++iy )
						for( int iz = 0; iz < (int)pvar->size(2); ++iz )
							(*pvar)(ix,iy,iz) += 0.5*((*pu)(ix,iy+1,iz)-(*pu)(ix,iy-1,iz))*h;

			else if( dir == 2 )
				#pragma omp parallel for
				for( int ix = 0; ix < (int)pvar->size(0); ++ix )
					for( int iy = 0; iy < (int)pvar->size(1); ++iy )
						for( int iz = 0; iz < (int)pvar->size(2); ++iz )
							(*pvar)(ix,iy,iz) += 0.5*((*pu)(ix,iy,iz+1)-(*pu)(ix,iy,iz-1))*h;
		}
	}

	LOGUSER("Done computing a 2nd order finite difference gradient.");
}

void multigrid_poisson_plugin::implementation::gradient_O4( int dir, grid_hierarchy& u, grid_hierarchy& Du )
{
	LOGUSER("Computing a 4th order finite difference gradient...");

	for( unsigned ilevel=u.levelmin(); ilevel<=u.levelmax(); ++ilevel )
	{
		double h = pow(2.0,ilevel);
		h /= 12.0;
		const size_t nb = u.num_boxes(ilevel);
		const bool multi = (nb > 1);

		for( size_t b=0; b < (multi ? nb : 1); ++b ){
			meshvar_bnd *pu   = multi ? u.get_grid(ilevel, b)  : u.get_grid(ilevel);
			meshvar_bnd *pvar = multi ? Du.get_grid(ilevel, b) : Du.get_grid(ilevel);

			if( dir == 0 )
#pragma omp parallel for
				for( int ix = 0; ix < (int)pu->size(0); ++ix )
					for( int iy = 0; iy < (int)pu->size(1); ++iy )
						for( int iz = 0; iz < (int)pu->size(2); ++iz )
							(*pvar)(ix,iy,iz) = ((*pu)(ix-2,iy,iz)
												-8.0*(*pu)(ix-1,iy,iz)
												 +8.0*(*pu)(ix+1,iy,iz)
												 -(*pu)(ix+2,iy,iz))*h;

			else if( dir == 1 )
#pragma omp parallel for
				for( int ix = 0; ix < (int)pu->size(0); ++ix )
					for( int iy = 0; iy < (int)pu->size(1); ++iy )
						for( int iz = 0; iz < (int)pu->size(2); ++iz )
							(*pvar)(ix,iy,iz) = ((*pu)(ix,iy-2,iz)
												 -8.0*(*pu)(ix,iy-1,iz)
												 +8.0*(*pu)(ix,iy+1,iz)
												 -(*pu)(ix,iy+2,iz))*h;

			else if( dir == 2 )
#pragma omp parallel for
				for( int ix = 0; ix < (int)pu->size(0); ++ix )
					for( int iy = 0; iy < (int)pu->size(1); ++iy )
						for( int iz = 0; iz < (int)pu->size(2); ++iz )
							(*pvar)(ix,iy,iz) = ((*pu)(ix,iy,iz-2)
												 -8.0*(*pu)(ix,iy,iz-1)
												 +8.0*(*pu)(ix,iy,iz+1)
												 -(*pu)(ix,iy,iz+2))*h;
		}
	}

	LOGUSER("Done computing a 4th order finite difference gradient.");
}

void multigrid_poisson_plugin::implementation::gradient_add_O4( int dir, grid_hierarchy& u, grid_hierarchy& Du )
{
	LOGUSER("Computing a 4th order finite difference gradient...");

	for( unsigned ilevel=u.levelmin(); ilevel<=u.levelmax(); ++ilevel )
	{
		double h = pow(2.0,ilevel);
		h /= 12.0;
		const size_t nb = u.num_boxes(ilevel);
		const bool multi = (nb > 1);

		for( size_t b=0; b < (multi ? nb : 1); ++b ){
			meshvar_bnd *pu   = multi ? u.get_grid(ilevel, b)  : u.get_grid(ilevel);
			meshvar_bnd *pvar = multi ? Du.get_grid(ilevel, b) : Du.get_grid(ilevel);

			if( dir == 0 )
#pragma omp parallel for
				for( int ix = 0; ix < (int)pu->size(0); ++ix )
					for( int iy = 0; iy < (int)pu->size(1); ++iy )
						for( int iz = 0; iz < (int)pu->size(2); ++iz )
							(*pvar)(ix,iy,iz) += ((*pu)(ix-2,iy,iz)
												 -8.0*(*pu)(ix-1,iy,iz)
												 +8.0*(*pu)(ix+1,iy,iz)
												 -(*pu)(ix+2,iy,iz))*h;

			else if( dir == 1 )
#pragma omp parallel for
				for( int ix = 0; ix < (int)pu->size(0); ++ix )
					for( int iy = 0; iy < (int)pu->size(1); ++iy )
						for( int iz = 0; iz < (int)pu->size(2); ++iz )
							(*pvar)(ix,iy,iz) += ((*pu)(ix,iy-2,iz)
												 -8.0*(*pu)(ix,iy-1,iz)
												 +8.0*(*pu)(ix,iy+1,iz)
												 -(*pu)(ix,iy+2,iz))*h;

			else if( dir == 2 )
#pragma omp parallel for
				for( int ix = 0; ix < (int)pu->size(0); ++ix )
					for( int iy = 0; iy < (int)pu->size(1); ++iy )
						for( int iz = 0; iz < (int)pu->size(2); ++iz )
							(*pvar)(ix,iy,iz) += ((*pu)(ix,iy,iz-2)
												 -8.0*(*pu)(ix,iy,iz-1)
												 +8.0*(*pu)(ix,iy,iz+1)
												 -(*pu)(ix,iy,iz+2))*h;
		}
	}


	LOGUSER("Done computing a 4th order finite difference gradient.");
}


void multigrid_poisson_plugin::implementation::gradient_O6( int dir, grid_hierarchy& u, grid_hierarchy& Du )
{
	LOGUSER("Computing a 6th order finite difference gradient...");

	for( unsigned ilevel=u.levelmin(); ilevel<=u.levelmax(); ++ilevel )
	{
		double h = pow(2.0,ilevel);
		h /= 60.;
		const size_t nb = u.num_boxes(ilevel);
		const bool multi = (nb > 1);

		for( size_t b=0; b < (multi ? nb : 1); ++b ){
			meshvar_bnd *pu   = multi ? u.get_grid(ilevel, b)  : u.get_grid(ilevel);
			meshvar_bnd *pvar = multi ? Du.get_grid(ilevel, b) : Du.get_grid(ilevel);

			if( dir == 0 )
				#pragma omp parallel for
				for( int ix = 0; ix < (int)pu->size(0); ++ix )
					for( int iy = 0; iy < (int)pu->size(1); ++iy )
						for( int iz = 0; iz < (int)pu->size(2); ++iz )
							(*pvar)(ix,iy,iz) =
							(-(*pu)(ix-3,iy,iz)
							 +9.0*(*pu)(ix-2,iy,iz)
							 -45.0*(*pu)(ix-1,iy,iz)
							 +45.0*(*pu)(ix+1,iy,iz)
							 -9.0*(*pu)(ix+2,iy,iz)
							 +(*pu)(ix+3,iy,iz))*h;

			else if( dir == 1 )
				#pragma omp parallel for
				for( int ix = 0; ix < (int)pu->size(0); ++ix )
					for( int iy = 0; iy < (int)pu->size(1); ++iy )
						for( int iz = 0; iz < (int)pu->size(2); ++iz )
							(*pvar)(ix,iy,iz) =
							(-(*pu)(ix,iy-3,iz)
							 +9.0*(*pu)(ix,iy-2,iz)
							 -45.0*(*pu)(ix,iy-1,iz)
							 +45.0*(*pu)(ix,iy+1,iz)
							 -9.0*(*pu)(ix,iy+2,iz)
							 +(*pu)(ix,iy+3,iz))*h;

			else if( dir == 2 )
				#pragma omp parallel for
				for( int ix = 0; ix < (int)pu->size(0); ++ix )
					for( int iy = 0; iy < (int)pu->size(1); ++iy )
						for( int iz = 0; iz < (int)pu->size(2); ++iz )
							(*pvar)(ix,iy,iz) =
							(-(*pu)(ix,iy,iz-3)
							 +9.0*(*pu)(ix,iy,iz-2)
							 -45.0*(*pu)(ix,iy,iz-1)
							 +45.0*(*pu)(ix,iy,iz+1)
							 -9.0*(*pu)(ix,iy,iz+2)
							 +(*pu)(ix,iy,iz+3))*h;
		}
	}

	LOGUSER("Done computing a 6th order finite difference gradient.");
}
	

void multigrid_poisson_plugin::implementation::gradient_add_O6( int dir, grid_hierarchy& u, grid_hierarchy& Du )
{
	LOGUSER("Computing a 6th order finite difference gradient...");

	for( unsigned ilevel=u.levelmin(); ilevel<=u.levelmax(); ++ilevel )
	{
		double h = pow(2.0,ilevel);
		h /= 60.;
		const size_t nb = u.num_boxes(ilevel);
		const bool multi = (nb > 1);

		for( size_t b=0; b < (multi ? nb : 1); ++b ){
			meshvar_bnd *pu   = multi ? u.get_grid(ilevel, b)  : u.get_grid(ilevel);
			meshvar_bnd *pvar = multi ? Du.get_grid(ilevel, b) : Du.get_grid(ilevel);

			if( dir == 0 )
				#pragma omp parallel for
				for( int ix = 0; ix < (int)pu->size(0); ++ix )
					for( int iy = 0; iy < (int)pu->size(1); ++iy )
						for( int iz = 0; iz < (int)pu->size(2); ++iz )
							(*pvar)(ix,iy,iz) +=
							(-(*pu)(ix-3,iy,iz)
							 +9.0*(*pu)(ix-2,iy,iz)
							 -45.0*(*pu)(ix-1,iy,iz)
							 +45.0*(*pu)(ix+1,iy,iz)
							 -9.0*(*pu)(ix+2,iy,iz)
							 +(*pu)(ix+3,iy,iz))*h;

			else if( dir == 1 )
				#pragma omp parallel for
				for( int ix = 0; ix < (int)pu->size(0); ++ix )
					for( int iy = 0; iy < (int)pu->size(1); ++iy )
						for( int iz = 0; iz < (int)pu->size(2); ++iz )
							(*pvar)(ix,iy,iz) +=
							(-(*pu)(ix,iy-3,iz)
							 +9.0*(*pu)(ix,iy-2,iz)
							 -45.0*(*pu)(ix,iy-1,iz)
							 +45.0*(*pu)(ix,iy+1,iz)
							 -9.0*(*pu)(ix,iy+2,iz)
							 +(*pu)(ix,iy+3,iz))*h;

			else if( dir == 2 )
				#pragma omp parallel for
				for( int ix = 0; ix < (int)pu->size(0); ++ix )
					for( int iy = 0; iy < (int)pu->size(1); ++iy )
						for( int iz = 0; iz < (int)pu->size(2); ++iz )
							(*pvar)(ix,iy,iz) +=
							(-(*pu)(ix,iy,iz-3)
							 +9.0*(*pu)(ix,iy,iz-2)
							 -45.0*(*pu)(ix,iy,iz-1)
							 +45.0*(*pu)(ix,iy,iz+1)
							 -9.0*(*pu)(ix,iy,iz+2)
							 +(*pu)(ix,iy,iz+3))*h;
		}
	}
	
	LOGUSER("Done computing a 6th order finite difference gradient.");
}


/**************************************************************************************/
/**************************************************************************************/
#include "general.hh"

double fft_poisson_plugin::solve( grid_hierarchy& f, grid_hierarchy& u )
{
	LOGUSER("Entering k-space Poisson solver...");
	
	unsigned verbosity = cf_.getValueSafe<unsigned>("setup","verbosity",2);
	
	if( f.levelmin() != f.levelmax() )
	{	
		LOGERR("Attempt to run k-space Poisson solver on non unigrid mesh.");
		throw std::runtime_error("fft_poisson_plugin::solve : k-space method can only be used in unigrid mode (levelmin=levelmax)");
	}
	
	if( verbosity > 0 )
	{
		std::cout << "-------------------------------------------------------------\n";
		std::cout << " - Invoking unigrid FFT Poisson solver..." << std::endl;
	}
	int nx,ny,nz,nzp;
	nx = f.get_grid(f.levelmax())->size(0);
	ny = f.get_grid(f.levelmax())->size(1);
	nz = f.get_grid(f.levelmax())->size(2);
	nzp = 2*(nz/2+1);

	//... copy data ..................................................
	fftw_real *data = new fftw_real[(size_t)nx*(size_t)ny*(size_t)nzp];
	fftw_complex *cdata = reinterpret_cast<fftw_complex*> (data);

	#pragma omp parallel for
	for( int i=0; i<nx; ++i )
		for( int j=0; j<ny; ++j )
			for( int k=0; k<nz; ++k )
			{
				size_t idx = (size_t)(i*ny+j)*(size_t)nzp+(size_t)k;
				data[idx] = (*f.get_grid(f.levelmax()))(i,j,k);
			}

	//... perform FFT and Poisson solve................................
#ifdef USE_MPI
	if( MUSIC::mpi::size() > 1 ){
		LOGUSER("Performing distributed forward+backward FFT Poisson solve.");
		MUSIC::poisson::rank0_dist_solve<fftw_real>(data, (size_t)nx, (size_t)ny, (size_t)nz);
	} else
#endif
	{
	LOGUSER("Performing forward transform.");

#ifdef FFTW3
	#ifdef SINGLE_PRECISION
	fftwf_plan
		plan  = fftwf_plan_dft_r2c_3d( nx, ny, nz, data, cdata, FFTW_ESTIMATE ),
		iplan = fftwf_plan_dft_c2r_3d( nx, ny, nz, cdata, data, FFTW_ESTIMATE );

	fftwf_execute(plan);
	#else
	fftw_plan
	plan  = fftw_plan_dft_r2c_3d( nx, ny, nz, data, cdata, FFTW_ESTIMATE ),
	iplan = fftw_plan_dft_c2r_3d( nx, ny, nz, cdata, data, FFTW_ESTIMATE );

	fftw_execute(plan);
	#endif

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

#endif
	double kfac = 2.0*M_PI;
	double fac = -1.0/(double)((size_t)nx*(size_t)ny*(size_t)nz);

	#pragma omp parallel for
	for( int i=0; i<nx; ++i )
		for( int j=0; j<ny; ++j )
			for( int k=0; k<nz/2+1; ++k )
			{
				int ii = i; if(ii>nx/2) ii-=nx;
				int jj = j; if(jj>ny/2) jj-=ny;
				double ki = (double)ii;
				double kj = (double)jj;
				double kk = (double)k;

				double kk2 = kfac*kfac*(ki*ki+kj*kj+kk*kk);

				size_t idx = (size_t)(i*ny+j)*(size_t)(nzp/2)+(size_t)k;

				RE(cdata[idx]) *= -1.0/kk2*fac;
				IM(cdata[idx]) *= -1.0/kk2*fac;
			}

	RE(cdata[0]) = 0.0;
	IM(cdata[0]) = 0.0;

	LOGUSER("Performing backward transform.");

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
	rfftwnd_threads_one_complex_to_real( omp_get_max_threads(), iplan, cdata, NULL );
	#else
	rfftwnd_one_complex_to_real( iplan, cdata, NULL );
	#endif

	rfftwnd_destroy_plan(plan);
	rfftwnd_destroy_plan(iplan);
#endif
	} // serial FFT branch
	
	


	
	//... copy data ..........................................
	#pragma omp parallel for
	for( int i=0; i<nx; ++i )
		for( int j=0; j<ny; ++j )	
			for( int k=0; k<nz; ++k )
			{
				size_t idx = (size_t)(i*ny+j)*(size_t)nzp+(size_t)k;
				(*u.get_grid(u.levelmax()))(i,j,k) = data[idx];
			}
	
	delete[] data;
	
	//... set boundary values ................................
	MUSIC::poisson::apply_periodic_bc_unigrid_rank0<fftw_real>(
	    *u.get_grid(u.levelmax()), nx, ny, nz );

	LOGUSER("Done with k-space Poisson solver.");
	return 0.0;
}


double fft_poisson_plugin::gradient( int dir, grid_hierarchy& u, grid_hierarchy& Du )
{
	
	LOGUSER("Computing a gradient in k-space...\n");
	
	if( u.levelmin() != u.levelmax() )
		throw std::runtime_error("fft_poisson_plugin::gradient : k-space method can only be used in unigrid mode (levelmin=levelmax)");
	
	Du = u;
	int nx,ny,nz,nzp;
	nx = u.get_grid(u.levelmax())->size(0);
	ny = u.get_grid(u.levelmax())->size(1);
	nz = u.get_grid(u.levelmax())->size(2);
	nzp = 2*(nz/2+1);
	
	//... copy data ..................................................
	fftw_real *data = new fftw_real[(size_t)nx*(size_t)ny*(size_t)nzp];
	fftw_complex *cdata = reinterpret_cast<fftw_complex*> (data);
	
	#pragma omp parallel for
	for( int i=0; i<nx; ++i )
		for( int j=0; j<ny; ++j )	
			for( int k=0; k<nz; ++k )
			{
				size_t idx = (size_t)(i*ny+j)*(size_t)nzp+(size_t)k;
				data[idx] = (*u.get_grid(u.levelmax()))(i,j,k);
			}
	
	bool do_glass = cf_.getValueSafe<bool>("output","glass",false);
	bool deconvolve_cic = do_glass | cf_.getValueSafe<bool>("output","glass_cicdeconvolve",false);

	if( deconvolve_cic )
		LOGINFO("CIC deconvolution is enabled for kernel!");

	//... perform FFT and Poisson solve................................
#ifdef USE_MPI
	if( MUSIC::mpi::size() > 1 ){
		LOGUSER("Performing distributed k-space gradient.");
		MUSIC::poisson::rank0_dist_gradient<fftw_real>(dir, data, (size_t)nx, (size_t)ny, (size_t)nz, deconvolve_cic);
	} else
#endif
	{
#ifdef FFTW3
	#ifdef SINGLE_PRECISION
	fftwf_plan
		plan  = fftwf_plan_dft_r2c_3d(nx, ny, nz, data, cdata, FFTW_ESTIMATE),
		iplan = fftwf_plan_dft_c2r_3d(nx, ny, nz, cdata, data, FFTW_ESTIMATE);

	fftwf_execute(plan);
	#else
	fftw_plan
	plan  = fftw_plan_dft_r2c_3d(nx, ny, nz, data, cdata, FFTW_ESTIMATE),
	iplan = fftw_plan_dft_c2r_3d(nx, ny, nz, cdata, data, FFTW_ESTIMATE);

	fftw_execute(plan);
	#endif
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

#endif

	double fac = -1.0/(double)((size_t)nx*(size_t)ny*(size_t)nz);
	double kfac = 2.0*M_PI;
	
	#pragma omp parallel for
	for( int i=0; i<nx; ++i )
		for( int j=0; j<ny; ++j )	
			for( int k=0; k<nz/2+1; ++k )
			{
				size_t idx = (size_t)(i*ny+j)*(size_t)(nzp/2)+(size_t)k;
				int ii = i; if(ii>nx/2) ii-=nx;
				int jj = j; if(jj>ny/2) jj-=ny;
				const double ki = (double)ii;
				const double kj = (double)jj;
				const double kk = (double)k;
				
				const double kkdir[3] = {kfac*ki,kfac*kj,kfac*kk};
				const double kdir = kkdir[dir];
				
				double re = RE(cdata[idx]);
				double im = IM(cdata[idx]);
				
				RE(cdata[idx]) = fac*im*kdir;
				IM(cdata[idx]) = -fac*re*kdir;
				
				
#ifdef FFTW3
				if( deconvolve_cic )
				{
					double dfx, dfy, dfz;
					dfx = M_PI*ki/(double)nx; dfx = (i!=0)? sin(dfx)/dfx : 1.0;
					dfy = M_PI*kj/(double)ny; dfy = (j!=0)? sin(dfy)/dfy : 1.0;
					dfz = M_PI*kk/(double)nz; dfz = (k!=0)? sin(dfz)/dfz : 1.0;
					
					dfx = 1.0/(dfx*dfy*dfz); dfx = dfx*dfx;
					cdata[idx][0] *= dfx;
					cdata[idx][1] *= dfx;
					
				}
#else
				if( deconvolve_cic )
				{
					double dfx, dfy, dfz;
					dfx = M_PI*ki/(double)nx; dfx = (i!=0)? sin(dfx)/dfx : 1.0;
					dfy = M_PI*kj/(double)ny; dfy = (j!=0)? sin(dfy)/dfy : 1.0;
					dfz = M_PI*kk/(double)nz; dfz = (k!=0)? sin(dfz)/dfz : 1.0;
					
					dfx = 1.0/(dfx*dfy*dfz); dfx = dfx*dfx;

					cdata[idx].re *= dfx;
					cdata[idx].im *= dfx;
				}
#endif			
				
				/*double ktot = sqrt(ii*ii+jj*jj+k*k);
				if( ktot >= nx/2 )//dir == 0 && i==nx/2 || dir == 1 && j==ny/2 || dir == 2 && k==nz/2 )
				{
					cdata[idx].re = 0.0;
					cdata[idx].im = 0.0;
				}*/
			}
	
	RE(cdata[0]) = 0.0;
	IM(cdata[0]) = 0.0;
	
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
	rfftwnd_threads_one_complex_to_real( omp_get_max_threads(), iplan, cdata, NULL );
	#else
	rfftwnd_one_complex_to_real( iplan, cdata, NULL );
	#endif
	
	rfftwnd_destroy_plan(plan);
	rfftwnd_destroy_plan(iplan);
#endif
	} // serial FFT branch

	//... copy data ..........................................
	double dmax = 0.0;
	for( int i=0; i<nx; ++i )
		for( int j=0; j<ny; ++j )
			for( int k=0; k<nz; ++k )
			{
				size_t idx = ((size_t)i*ny+(size_t)j)*nzp+(size_t)k;
				(*Du.get_grid(u.levelmax()))(i,j,k) = data[idx];
				if(fabs(data[idx])>dmax)
					dmax = fabs(data[idx]);
			}

	delete[] data;

	LOGUSER("Done with k-space gradient.\n");

	return 0.0;
}

/**************************************************************************************/
/**************************************************************************************/


template<int order>
void do_poisson_hybrid( fftw_real* data, int idir, int nxp, int nyp, int nzp, bool periodic, bool deconvolve_cic )
{
	double fftnorm = 1.0/((double)nxp*(double)nyp*(double)nzp);
		
	fftw_complex	*cdata = reinterpret_cast<fftw_complex*>(data);
	
	if( deconvolve_cic )
	  LOGINFO("CIC deconvolution step is enabled.");

#ifdef FFTW3
	#ifdef SINGLE_PRECISION
	fftwf_plan iplan, plan;
	plan  = fftwf_plan_dft_r2c_3d(nxp, nyp, nzp, data, cdata, FFTW_ESTIMATE);
	iplan = fftwf_plan_dft_c2r_3d(nxp, nyp, nzp, cdata, data, FFTW_ESTIMATE);
	fftwf_execute(plan);
	#else
	fftw_plan iplan, plan;
	plan  = fftw_plan_dft_r2c_3d(nxp, nyp, nzp, data, cdata, FFTW_ESTIMATE);
	iplan = fftw_plan_dft_c2r_3d(nxp, nyp, nzp, cdata, data, FFTW_ESTIMATE);
	fftw_execute(plan);
	#endif
#else
	rfftwnd_plan	iplan, plan;
	
	plan  = rfftw3d_create_plan( nxp, nyp, nzp,
								FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE|FFTW_IN_PLACE);
	
	iplan = rfftw3d_create_plan( nxp, nyp, nzp, 
								FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE|FFTW_IN_PLACE);
	
	#ifndef SINGLETHREAD_FFTW		
	rfftwnd_threads_one_real_to_complex( omp_get_max_threads(), plan, data, NULL );
	#else
	rfftwnd_one_real_to_complex( plan, data, NULL );
	#endif
#endif
	
	long double ksum = 0.0;
	size_t kcount = 0;
	
	#pragma omp parallel for reduction(+:ksum,kcount)
	for( int i=0; i<nxp; ++i )
		for( int j=0; j<nyp; ++j )
			for( int k=0; k<nzp/2+1; ++k )
			{
				size_t ii = (size_t)(i*nyp + j) * (size_t)(nzp/2+1) + (size_t)k;
				
				if( k==0 || k==nzp/2 )
				{
					ksum  += RE(cdata[ii]);
					kcount++;
				}else{
					ksum  += 2.0*(RE(cdata[ii]));
					kcount+=2;
				}
			}
	
	ksum /= kcount;
	kcount = 0;
	
	
	#pragma omp parallel for
	for( int i=0; i<nxp; ++i )
		for( int j=0; j<nyp; ++j )
			for( int k=0; k<nzp/2+1; ++k )
			{
			
				size_t ii = (size_t)(i*nyp + j) * (size_t)(nzp/2+1) + (size_t)k;
	
				int ki(i), kj(j), kk(k);
				if( ki > nxp/2 ) ki-=nxp;
				if( kj > nyp/2 ) kj-=nyp;
				
				//... apply hybrid correction
				double dk = poisson_hybrid_kernel<order>(idir, ki, kj, k, nxp/2 );
				//RE(cdata[ii]) -= ksum;
				
				fftw_real re = RE(cdata[ii]), im = IM(cdata[ii]);
				
				RE(cdata[ii]) = -im*dk*fftnorm;
				IM(cdata[ii]) = re*dk*fftnorm;

				if( deconvolve_cic )
				{
					double dfx, dfy, dfz;
					dfx = M_PI*ki/(double)nxp; dfx = (i!=0)? sin(dfx)/dfx : 1.0;
					dfy = M_PI*kj/(double)nyp; dfy = (j!=0)? sin(dfy)/dfy : 1.0;
					dfz = M_PI*kk/(double)nzp; dfz = (k!=0)? sin(dfz)/dfz : 1.0;
					
					dfx = 1.0/(dfx*dfy*dfz); dfx = dfx*dfx;
					RE(cdata[ii]) *= dfx;
					IM(cdata[ii]) *= dfx;
					
				}

				

				//RE(cdata[ii]) += ksum*fftnorm;
				
				//if( i==nxp/2||j==nyp/2||k==nzp/2 )
				//{
				//	RE(cdata[ii]) = 0.0;
				//	IM(cdata[ii]) = 0.0;
				//}
			}
	
	RE(cdata[0]) = 0.0;
	IM(cdata[0]) = 0.0;
	
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
	rfftwnd_threads_one_complex_to_real( omp_get_max_threads(), iplan, cdata, NULL);
	#else		
	rfftwnd_one_complex_to_real(iplan, cdata, NULL);
	#endif
	
	rfftwnd_destroy_plan(plan);
	rfftwnd_destroy_plan(iplan);
#endif
	
}
   
template< typename T >
void poisson_hybrid( T& f, int idir, int order, bool periodic, bool deconvolve_cic )

{
	int nx=f.size(0), ny=f.size(1), nz=f.size(2), nxp, nyp, nzp;
	fftw_real		*data;
	int xo=0,yo=0,zo=0;
	int nmax = std::max(nx,std::max(ny,nz));
	
	LOGUSER("Entering hybrid Poisson solver...");
	
	if(!periodic)
	{
		nxp = 2*nmax;
		nyp = 2*nmax;
		nzp = 2*nmax;
		xo  = nmax/2;
		yo  = nmax/2;
		zo  = nmax/2;
	}
	else
	{
		nxp = nmax;
		nyp = nmax;
		nzp = nmax;
	}
	
	
	data		= new fftw_real[(size_t)nxp*(size_t)nyp*(size_t)(nzp+2)];
	
	if(idir==0)
		std::cout << "   - Performing hybrid Poisson step... (" << nxp <<  ", " << nyp << ", " << nzp << ")\n";
	
	//size_t N = (size_t)nxp*(size_t)nyp*2*((size_t)nzp/2+1);
	
	//#pragma omp parallel for
	//for( size_t i=0; i<N; ++i )
	//	data[i]=0.0;
	
	#pragma omp parallel for
	for( int i=0; i<nxp; ++i )
		for( int j=0; j<nyp; ++j )
			for( int k=0; k<=nzp; ++k )
			{
				size_t idx = ((size_t)i*(size_t)nxp+(size_t)j)*(size_t)(nzp+2)+(size_t)k;
				data[idx] = 0.0;
			}
	
	#pragma omp parallel for
	for( int i=0; i<nx; ++i )
		for( int j=0; j<ny; ++j )
			for( int k=0; k<nz; ++k )
			{
				size_t idx = (size_t)((i+xo)*nyp + j+yo) * (size_t)(nzp+2) + (size_t)(k+zo);
				data[idx] = f(i,j,k);
			}
	
	switch (order) {
		case 2:
		  do_poisson_hybrid<2>(data, idir, nxp, nyp, nzp, periodic, deconvolve_cic );
		  break;
		case 4:
		  do_poisson_hybrid<4>(data, idir, nxp, nyp, nzp, periodic, deconvolve_cic );
		  break;
		case 6:
		  do_poisson_hybrid<6>(data, idir, nxp, nyp, nzp, periodic, deconvolve_cic );
		  break;
		default:
			std::cerr << " - ERROR: invalid operator order specified in deconvolution.";
			LOGERR("Invalid operator order specified in deconvolution.");
			break;
	}
	
	LOGUSER("Copying hybrid correction factor...");
	
	#pragma omp parallel for
	for( int i=0; i<nx; ++i )
		for( int j=0; j<ny; ++j )
			for( int k=0; k<nz; ++k )
			{
				size_t idx = ((size_t)(i+xo)*nyp + (size_t)(j+yo)) * (size_t)(nzp+2) + (size_t)(k+zo);	
				f(i,j,k) = data[idx];
			}
	
	delete[] data;

	LOGUSER("Done with hybrid Poisson solve.");
}
	   
	   
/**************************************************************************************/
/**************************************************************************************/

template void poisson_hybrid< MeshvarBnd<double> >( MeshvarBnd<double>& f, int idir, int order, bool periodic, bool deconvolve_cic );
template void poisson_hybrid< MeshvarBnd<float> >( MeshvarBnd<float>& f, int idir, int order, bool periodic, bool deconvolve_cic );

namespace{
	poisson_plugin_creator_concrete<multigrid_poisson_plugin> multigrid_poisson_creator("mg_poisson");
	poisson_plugin_creator_concrete<fft_poisson_plugin> fft_poisson_creator("fft_poisson");
}
