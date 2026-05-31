/*
 
 mg_solver.hh - This file is part of MUSIC -
 a code to generate multi-scale initial conditions 
 for cosmological simulations 
 
 Copyright (C) 2010  Oliver Hahn
 
*/

#ifndef __MG_SOLVER_HH
#define __MG_SOLVER_HH

#include <cmath>
#include <iostream>

#include "mg_operators.hh"
#include "mg_interp.hh"

#include "mesh.hh"
#include "mg_dist.hh"
#include "mg_profile.hh"
#include "zoom_slab.hh"
#include "mpi_helper.hh"
#include "mpi_poisson.hh"

#define BEGIN_MULTIGRID_NAMESPACE namespace multigrid {
#define END_MULTIGRID_NAMESPACE }

BEGIN_MULTIGRID_NAMESPACE
	
//! options for multigrid smoothing operation
namespace opt {
	enum smtype { sm_jacobi, sm_gauss_seidel, sm_sor };
}


//! actual implementation of FAS adaptive multigrid solver
template< class S, class I, class O, typename T=double >
class solver
{
public:
	typedef S scheme;
	typedef O mgop;
	typedef I interp;

protected:
	scheme				m_scheme;				//!< finite difference scheme
	mgop				m_gridop;				//!< grid prolongation and restriction operator
	unsigned			m_npresmooth,			//!< number of pre sweeps
						m_npostsmooth;			//!< number of post sweeps
	opt::smtype			m_smoother;				//!< smoothing method to be applied
	unsigned			m_ilevelmin;			//!< index of the top grid level
	
	const static bool	m_bperiodic = true;		//!< flag whether top grid is periodic
	
	std::vector<double> m_residu_ini;			//!< vector of initial residuals for each level
	bool m_is_ini;								//!< bool that is true for first iteration

	GridHierarchy<T>	*m_pu,					//!< pointer to GridHierarchy for solution u
						*m_pf,					//!< pointer to GridHierarchy for right-hand-side
						*m_pfsave;				//!< pointer to saved state of right-hand-side (unused)
	
	const MeshvarBnd<T> *m_pubnd;
	
	//! compute residual for a level
  double compute_error( const MeshvarBnd<T>& u, const MeshvarBnd<T>& unew, int ilevel );
	
	//! compute residuals for entire grid hierarchy
	double compute_error( const GridHierarchy<T>& uh, const GridHierarchy<T>& uhnew, bool verbose );
	
	//! compute residuals for entire grid hierarchy
	double compute_RMS_resid( const GridHierarchy<T>& uh, const GridHierarchy<T>& fh, bool verbose );

protected:
	
	//! Jacobi smoothing 
	void Jacobi( T h, MeshvarBnd<T>* u, const MeshvarBnd<T>* f );
	
	//! Gauss-Seidel smoothing
	void GaussSeidel( T h, MeshvarBnd<T>* u, const MeshvarBnd<T>* f );
	
	//! Successive-Overrelaxation smoothing
	void SOR( T h, MeshvarBnd<T>* u, const MeshvarBnd<T>* f );
	
	//! main two-grid (V-cycle) for multi-grid iterations
	void twoGrid( unsigned ilevel );

	//! Phase D.3.2: multi-box variant of twoGrid. At fine level L with
	//! num_boxes(L)>1 each per-box mesh is pre/post-smoothed independently,
	//! restricts into the still-existing union parent uc at L-1 (which gives
	//! correct apply(uc) at gap cells), recurses through standard twoGrid
	//! on the single-box parent, and receives the correction back from
	//! the union uc into each per-box uf. Requires num_boxes(L-1)==1 in
	//! this phase (D.3.2). General multi-box parent is D.3.3.
	void twoGrid_multibox( unsigned ilevel );

	//! Phase G.2b B.2b.2.1: SPMD twin of twoGrid_multibox.
	//! Structurally a port of twoGrid_multibox with:
	//!  - parent_idx-keyed per-rank scratch (tLus / ucsave / cc)
	//!  - owner-gated per-box loops
	//!  - parent reads via m_pu->parent_for_box(ilevel, b) (local or replica)
	//!  - broadcast_parents_to_child_owners at entry + after recurse
	//!  - accumulate_children_to_parents after FAS source update
	//! At np==1 every primitive is a no-op and owner_of_box==my_rank everywhere
	//! → bit-identical to twoGrid_multibox. np>1 SPMD wiring (outer wpd_spmd
	//! swap, recursion gating) lands in B.2b.2.2.
	void twoGrid_multibox_spmd( unsigned ilevel );
	
	//! apply boundary conditions
	void setBC( unsigned ilevel );
	
	//! make top grid periodic boundary conditions
	void make_periodic( MeshvarBnd<T> *u );
	
	//void interp_coarse_fine_cubic( unsigned ilevel, MeshvarBnd<T>& coarse, MeshvarBnd<T>& fine );
		
public:
	
	//! constructor
	solver( GridHierarchy<T>& f, opt::smtype smoother, unsigned npresmooth, unsigned npostsmooth );
	
	//! destructor
	~solver()
	{  }
	
	//! solve Poisson's equation 
	double solve( GridHierarchy<T>& u, double accuracy, double h=-1.0, bool verbose=false );
	
	//! solve Poisson's equation 
	double solve( GridHierarchy<T>& u, double accuracy, bool verbose=false )
	{
		return this->solve ( u, accuracy, -1.0, verbose );
	}
	
	
	
};


template< class S, class I, class O, typename T >
solver<S,I,O,T>::solver( GridHierarchy<T>& f, opt::smtype smoother, unsigned npresmooth, unsigned npostsmooth )
:	m_scheme(), m_gridop(), m_npresmooth( npresmooth ), m_npostsmooth( npostsmooth ), 
m_smoother( smoother ), m_ilevelmin( f.levelmin() ), m_is_ini( true ), m_pf( &f )
{ 
	m_is_ini = true;
}


template< class S, class I, class O, typename T >
void solver<S,I,O,T>::Jacobi( T h, MeshvarBnd<T> *u, const MeshvarBnd<T>* f )
{
	int
		nx = u->size(0), 
		ny = u->size(1), 
		nz = u->size(2);
	
	double 
		c0 = -1.0/m_scheme.ccoeff(),
		h2 = h*h; 
	
	MeshvarBnd<T> uold(*u);
	
	double alpha = 0.95, ialpha = 1.0-alpha;
	
	#pragma omp parallel for
	for( int ix=0; ix<nx; ++ix )
		for( int iy=0; iy<ny; ++iy )
			for( int iz=0; iz<nz; ++iz )
				(*u)(ix,iy,iz) = ialpha * uold(ix,iy,iz) + alpha * (m_scheme.rhs( uold, ix, iy, iz ) + h2 * (*f)(ix,iy,iz))*c0;
	
}

template< class S, class I, class O, typename T >
void solver<S,I,O,T>::SOR( T h, MeshvarBnd<T> *u, const MeshvarBnd<T>* f )
{
	int
		nx = u->size(0), 
		ny = u->size(1), 
		nz = u->size(2);

	double 
		c0 = -1.0/m_scheme.ccoeff(),
		h2 = h*h; 
		
	MeshvarBnd<T> uold(*u);
	
	double 
		alpha = 1.2, 
	//alpha = 2 / (1 + 4 * atan(1.0) / double(u->size(0)))-1.0, //.. ideal alpha
		ialpha = 1.0-alpha;
	
	#pragma omp parallel for
	for( int ix=0; ix<nx; ++ix )
		for( int iy=0; iy<ny; ++iy )
			for( int iz=0; iz<nz; ++iz )
				if( (ix+iy+iz)%2==0 )
					(*u)(ix,iy,iz) = ialpha * uold(ix,iy,iz) + alpha * (m_scheme.rhs( uold, ix, iy, iz ) + h2 * (*f)(ix,iy,iz))*c0;
	
	
	#pragma omp parallel for
	for( int ix=0; ix<nx; ++ix )
		for( int iy=0; iy<ny; ++iy )
			for( int iz=0; iz<nz; ++iz )
				if( (ix+iy+iz)%2!=0 )
					(*u)(ix,iy,iz) = ialpha * uold(ix,iy,iz) + alpha * (m_scheme.rhs( *u, ix, iy, iz ) + h2 * (*f)(ix,iy,iz))*c0;
	
	
	
}

template< class S, class I, class O, typename T >
void solver<S,I,O,T>::GaussSeidel( T h, MeshvarBnd<T>* u, const MeshvarBnd<T>* f )
{
	int 
		nx = u->size(0), 
		ny = u->size(1), 
		nz = u->size(2);
	
	T
		c0 = -1.0/m_scheme.ccoeff(),
		h2 = h*h; 
	
	for( int color=0; color < 2; ++color )
		#pragma omp parallel for
		for( int ix=0; ix<nx; ++ix )
			for( int iy=0; iy<ny; ++iy )
				for( int iz=0; iz<nz; ++iz )
					if( (ix+iy+iz)%2 == color )
						(*u)(ix,iy,iz) = (m_scheme.rhs( *u, ix, iy, iz ) + h2 * (*f)(ix,iy,iz))*c0;
	
}


template< class S, class I, class O, typename T >
void solver<S,I,O,T>::twoGrid( unsigned ilevel )
{
	// Phase D.3.3 (proper, composite multigrid): at multi-box levels,
	// dispatch to per-cluster V-cycle. compute_error/compute_RMS_resid,
	// gradient operator, and final FAS-restrict loop are now all per-box
	// aware so the union mesh at multi-box levels is unused on the compute
	// path — per-box is the source of truth.
	if( m_pu->num_boxes(ilevel) > 1 ){
		twoGrid_multibox(ilevel);
		return;
	}

	MeshvarBnd<T> *uf, *uc, *ff, *fc;


	double
		h = 1.0/(1<<ilevel),
		c0 = -1.0/m_scheme.ccoeff(),
		h2 = h*h;

	uf = m_pu->get_grid(ilevel);
	ff = m_pf->get_grid(ilevel);
	
	uc = m_pu->get_grid(ilevel-1);
	fc = m_pf->get_grid(ilevel-1);	
	
	
	int
		nx = uf->size(0),
		ny = uf->size(1),
		nz = uf->size(2);

	if( m_bperiodic && ilevel <= m_ilevelmin) {
		double _tp = MUSIC::mg_profile::tic();
		make_periodic( uf );
		MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_PERIODIC, _tp);
	}
	else if(!m_bperiodic)
		setBC( ilevel );

	// MPI-distributed path at the finest level only (Phase 2).
	// Coarser levels stay serial on rank 0; the recursive descent is
	// irreducibly serial in this V-cycle.
	const bool mg_dist_active = MUSIC::mg::should_distribute(ilevel, m_pu->levelmax())
	                            && (m_smoother == opt::sm_gauss_seidel);
	const int  mg_dist_order  = S::order;

	if( mg_dist_active ){
		double _tb = MUSIC::mg_profile::tic();
		MUSIC::mg::mg_begin<T>( mg_dist_order, (int)ilevel, *uf, *ff );
		MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_MG_BEGIN_END, _tb);
	}

	//... do smoothing sweeps with specified solver
	for( unsigned i=0; i<m_npresmooth; ++i ){

		if( ilevel > m_ilevelmin ){
			double _ti = MUSIC::mg_profile::tic();
			interp().interp_coarse_fine(ilevel,*uc,*uf);
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_INTERP_CF, _ti);
		}

		if( mg_dist_active ){
			double _ts = MUSIC::mg_profile::tic();
			MUSIC::mg::mg_scatter_uf<T>( *uf );
			MUSIC::mg::mg_gs_sweep();
			MUSIC::mg::mg_gather_uf<T>( *uf );
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_SMOOTH_DIST, _ts);
		}
		else if( m_smoother == opt::sm_gauss_seidel ){
			double _ts = MUSIC::mg_profile::tic();
			GaussSeidel( h, uf, ff );
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_SMOOTH_LOCAL, _ts);
		}

		else if( m_smoother == opt::sm_jacobi ){
			double _ts = MUSIC::mg_profile::tic();
			Jacobi( h, uf, ff);
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_SMOOTH_LOCAL, _ts);
		}

		else if( m_smoother == opt::sm_sor ){
			double _ts = MUSIC::mg_profile::tic();
			SOR( h, uf, ff );
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_SMOOTH_LOCAL, _ts);
		}

		if( m_bperiodic && ilevel <= m_ilevelmin ){
			double _tp = MUSIC::mg_profile::tic();
			make_periodic( uf );
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_PERIODIC, _tp);
		}
	}


	{
		double _tr = MUSIC::mg_profile::tic();
		m_gridop.restrict( *uf, *uc );
		MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_RESTRICT_U, _tr);
	}
	
	//... essential!!
	if( m_bperiodic && ilevel <= m_ilevelmin ){
		double _tp = MUSIC::mg_profile::tic();
		make_periodic( uc );
		MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_PERIODIC, _tp);
	}
	else if( ilevel > m_ilevelmin ){
		double _ti = MUSIC::mg_profile::tic();
		interp().interp_coarse_fine(ilevel,*uc,*uf);
		MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_INTERP_CF, _ti);
	}

	
	//....................................................................
	//... we now use hard-coded restriction+operatore app, see below
	/*meshvar_bnd Lu(*uf,false);
	Lu.zero();

	#pragma omp parallel for
	for( int ix=0; ix<nx; ++ix )
		for( int iy=0; iy<ny; ++iy )
			for( int iz=0; iz<nz; ++iz )
				Lu(ix,iy,iz) = m_scheme.apply( (*uf), ix, iy, iz )/h2;
	
	meshvar_bnd tLu(*uc,false);
	
	
	//... restrict Lu
	m_gridop.restrict( Lu, tLu );
	Lu.deallocate();*/
	//.................................................................... 
	
	int 
		oxp = uf->offset(0),
		oyp = uf->offset(1),
		ozp = uf->offset(2);
	
	meshvar_bnd tLu(*uc,false);
	if( mg_dist_active ){
		// Distributed apply+restrict (Phase A.2). Workers compute their slab's
		// share of tLu (via on_apply_restrict) and rank-0 gathers + writes into
		// tLu at offset (oxp, oyp, ozp). The required pre-scatter of *uf is
		// done inside mg_apply_restrict via mg_scatter_uf.
		double _ta = MUSIC::mg_profile::tic();
		MUSIC::mg::mg_apply_restrict<T>( *uf, tLu, oxp, oyp, ozp );
		MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_APPLY_REST_DIST, _ta);
	}
	else
	{
	double _ta = MUSIC::mg_profile::tic();
	#pragma omp parallel for
	for( int ix=0; ix<nx/2; ++ix )
	{
		int iix=2*ix;
		for( int iy=0,iiy=0; iy<ny/2; ++iy,iiy+=2 )


			for( int iz=0,iiz=0; iz<nz/2; ++iz,iiz+=2 )
				tLu(ix+oxp,iy+oyp,iz+ozp) = 0.125 * (
							 m_scheme.apply( (*uf), iix, iiy, iiz )
							+m_scheme.apply( (*uf), iix, iiy, iiz+1 )
							+m_scheme.apply( (*uf), iix, iiy+1, iiz )
							+m_scheme.apply( (*uf), iix, iiy+1, iiz+1 )
							+m_scheme.apply( (*uf), iix+1, iiy, iiz )
							+m_scheme.apply( (*uf), iix+1, iiy, iiz+1 )
							+m_scheme.apply( (*uf), iix+1, iiy+1, iiz )
							+m_scheme.apply( (*uf), iix+1, iiy+1, iiz+1 )
						)/h2;
	}
	MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_APPLY_REST_LOCAL, _ta);
	}

	//... restrict source term
	{
		double _trf = MUSIC::mg_profile::tic();
		m_gridop.restrict( *ff, *fc );

		int oi, oj, ok;
		oi = ff->offset(0);
		oj = ff->offset(1);
		ok = ff->offset(2);

		#pragma omp parallel for
		for( int ix=oi; ix<oi+(int)ff->size(0)/2; ++ix )
			for( int iy=oj; iy<oj+(int)ff->size(1)/2; ++iy )
				for( int iz=ok; iz<ok+(int)ff->size(2)/2; ++iz )
					(*fc)(ix,iy,iz) += ((tLu( ix, iy, iz ) - (m_scheme.apply( *uc, ix, iy, iz )/(4.0*h2))));
		MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_RESTRICT_F, _trf);
	}

	tLu.deallocate();
	
	meshvar_bnd ucsave(*uc,true);
						
	//... have we reached the end of the recursion or do we need to go up one level?
	if( ilevel == 1 )
		if( m_bperiodic )
			(*uc)(0,0,0) = 0.0;
		else 
			(*uc)(0,0,0) = (m_scheme.rhs( (*uc), 0, 0, 0 ) + 4.0 * h2 * (*fc)(0,0,0))*c0;
	else
		twoGrid( ilevel-1 );
	
	meshvar_bnd cc(*uc,false);

	{
		double _tc = MUSIC::mg_profile::tic();
		//... compute correction on coarse grid
		#pragma omp parallel for
		for( int ix=0; ix<(int)cc.size(0); ++ix )
			for( int iy=0; iy<(int)cc.size(1); ++iy )
				for( int iz=0; iz<(int)cc.size(2); ++iz )
					cc(ix,iy,iz) = (*uc)(ix,iy,iz) - ucsave(ix,iy,iz);

		ucsave.deallocate();

		if( m_bperiodic && ilevel <= m_ilevelmin )
			make_periodic( &cc );

		m_gridop.prolong_add( cc, *uf );
		MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_PROLONG_CC, _tc);
	}

	//... interpolate and apply coarse-fine boundary conditions on fine level
	if( m_bperiodic && ilevel <= m_ilevelmin ){
		double _tp = MUSIC::mg_profile::tic();
		make_periodic( uf );
		MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_PERIODIC, _tp);
	}
	else if(!m_bperiodic)
		setBC( ilevel );

	//... do smoothing sweeps with specified solver
	for( unsigned i=0; i<m_npostsmooth; ++i ){

		if( ilevel > m_ilevelmin ){
			double _ti = MUSIC::mg_profile::tic();
			interp().interp_coarse_fine(ilevel,*uc,*uf);
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_INTERP_CF, _ti);
		}

		if( mg_dist_active ){
			double _ts = MUSIC::mg_profile::tic();
			MUSIC::mg::mg_scatter_uf<T>( *uf );
			MUSIC::mg::mg_gs_sweep();
			MUSIC::mg::mg_gather_uf<T>( *uf );
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_SMOOTH_DIST, _ts);
		}
		else if( m_smoother == opt::sm_gauss_seidel ){
			double _ts = MUSIC::mg_profile::tic();
			GaussSeidel( h, uf, ff );
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_SMOOTH_LOCAL, _ts);
		}

		else if( m_smoother == opt::sm_jacobi ){
			double _ts = MUSIC::mg_profile::tic();
			Jacobi( h, uf, ff);
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_SMOOTH_LOCAL, _ts);
		}

		else if( m_smoother == opt::sm_sor ){
			double _ts = MUSIC::mg_profile::tic();
			SOR( h, uf, ff );
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_SMOOTH_LOCAL, _ts);
		}

		if( m_bperiodic && ilevel <= m_ilevelmin ){
			double _tp = MUSIC::mg_profile::tic();
			make_periodic( uf );
			MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_PERIODIC, _tp);
		}

	}

	if( mg_dist_active ){
		double _tb = MUSIC::mg_profile::tic();
		MUSIC::mg::mg_end<T>( *uf );
		MUSIC::mg_profile::add(ilevel, MUSIC::mg_profile::P_MG_BEGIN_END, _tb);
	}

}

// --------------------------------------------------------------------------
// Phase D.3.2: multi-box V-cycle branch. At fine level `ilevel` each child
// box restricts/prolongs through its OWN parent box at level-1 (looked up
// via parent_box_index). The parent level then dispatches recursively —
// if num_boxes(L-1)>1 it re-enters twoGrid_multibox; if single it falls
// through to the standard twoGrid path.
//
// FAS source update + apply_restrict: when the parent is single-box, the
// union uc/fc carry data at "gap-between-clusters" cells so the apply(uc)
// stencil reads sensible values. When the parent is itself multi-box,
// each child's parent IS one specific per-box parent mesh, and we use the
// parent's own halo (filled by interp_coarse_fine from the grandparent in
// the previous frame of the recursion) as the gap source.
// --------------------------------------------------------------------------
template< class S, class I, class O, typename T >
void solver<S,I,O,T>::twoGrid_multibox( unsigned ilevel )
{
	size_t nb = m_pu->num_boxes(ilevel);
	if( nb <= 1 ){ twoGrid(ilevel); return; }

	// Phase G.2b B.2b.2.1 → B.2b.2.2.1.d → D.6 follow-up: SPMD dispatcher.
	//
	// Route to twoGrid_multibox_spmd only when BOTH:
	//   (1) user opted in via setup.zoom_slab_spmd_multigrid
	//   (2) we are currently executing under wpd_spmd (thread_local flag set
	//       by MUSIC::poisson::with_pbox_distributed_spmd entry)
	//
	// The second condition is the safety gate. The SPMD V-cycle broadcasts
	// B.2b.* collective ops over WORLD; if we entered solve() via the classic
	// with_pbox_distributed (workers parked in worker_pump) those ops surface
	// as `worker_pump: unknown op N` on the workers. This fired for the 2LPT
	// solve at main.cc:1946 — the outer wrapper there is classic wpd because
	// the lambda body contains union-mesh ops (compute_2LPT_source, output
	// writes) that are not SPMD-safe. We let those classic-wpd sites fall
	// through to twoGrid_multibox (rank-0 composite MG) regardless of flag,
	// and only dispatch SPMD where the call site has explicitly wrapped in
	// wpd_spmd (e.g. main.cc:1916 1LPT solve via with_pbox_distributed_maybe_spmd).
	if( MUSIC::zoom_slab::spmd_multigrid_enabled()
	    && MUSIC::poisson::spmd_mg_is_active() ){
		static bool s_spmd_dispatch_logged = false;
		if( !s_spmd_dispatch_logged ){
			s_spmd_dispatch_logged = true;
			LOGINFO("G.2b B.2b.2.1: spmd_multigrid dispatcher routing to twoGrid_multibox_spmd "
			        "at ilevel=%u nb=%zu (np=%d)",
			        (unsigned)ilevel, nb, MUSIC::mpi::size());
		}
		twoGrid_multibox_spmd( ilevel );
		return;
	}

	const bool parent_multi = (m_pu->num_boxes(ilevel-1) > 1);

	double
		h  = 1.0/(1<<ilevel),
		c0 = -1.0/m_scheme.ccoeff(),
		h2 = h*h;
	(void)c0;

	// Gather child pointers and resolve each child's parent.
	std::vector< MeshvarBnd<T>* > ufs(nb), ffs(nb);
	std::vector< MeshvarBnd<T>* > ucs(nb), fcs(nb);
	std::vector< size_t > pidx(nb);
	for( size_t b=0; b<nb; ++b ){
		ufs[b] = m_pu->get_grid(ilevel, b);
		ffs[b] = m_pf->get_grid(ilevel, b);
		if( parent_multi ){
			pidx[b] = m_pu->parent_box_index(ilevel, b);
			ucs[b]  = m_pu->get_grid(ilevel-1, pidx[b]);
			fcs[b]  = m_pf->get_grid(ilevel-1, pidx[b]);
		} else {
			pidx[b] = 0;
			ucs[b]  = m_pu->get_grid(ilevel-1);
			fcs[b]  = m_pf->get_grid(ilevel-1);
		}
	}

	// Phase G.2b A'/B.2a: decide once per V-cycle level whether the per-box GS
	// sites route through the z-slab bridge (gs_z_neg_meshvarbnd) instead of the
	// in-place GaussSeidel formula.
	//
	// Gate envelope:
	//   A' (np==1): bridge active. sub_size=1, bridge allocs its own buffer.
	//   B.2a (np>1 + policy=="self"): bridge active. sub_comm=MPI_COMM_SELF for
	//      every box → sub_size=1, bridge still purely local per rank. The
	//      multibox composite MG remains rank-0-only (workers in worker_pump);
	//      only rank 0 actually executes any per-box GS, so the COMM_SELF call
	//      is just a single-rank local op (no collectives, no deadlock).
	//   B.2b/later (np>1 + policy=="world"|"round_robin"): GATED OFF here.
	//      Those require lifting with_pbox_distributed's rank-0-only execution
	//      so workers also enter twoGrid_multibox and engage in collectives on
	//      their sub_comm. Falls through to in-place GaussSeidel for now.
	const bool use_zoom_slab_gs =
		MUSIC::zoom_slab::smoother_enabled()
		&& (S::order == 2)
		&& ( MUSIC::mpi::size() == 1
		     || MUSIC::zoom_slab::subcomm_policy() == "self" );
	// B.1: per-cluster sub_comm is looked up from MUSIC::zoom_slab registry per box
	// inside the GS loops below. Registry policy ("world"|"self"|"round_robin") is
	// set in densities.cc from setup.zoom_slab_subcomm. When no registry is built
	// (e.g. flag off or non-multibox), subcomm_for_box returns MPI_COMM_WORLD.
	const int zoom_slab_halo_w = 1;
	{
		static bool s_bridge_logged = false;
		if( use_zoom_slab_gs && !s_bridge_logged ){
			s_bridge_logged = true;
			LOGINFO("G.2b A'/B.2a: zoom-slab GS bridge active at ilevel=%u nb=%zu "
			        "(np=%d, S::order=2, policy=\"%s\")",
			        (unsigned)ilevel, nb, MUSIC::mpi::size(),
			        MUSIC::zoom_slab::subcomm_policy().c_str());
		}
	}

	// --- pre-smoothing sweeps (per box) -------------------------------
	for( unsigned i=0; i<m_npresmooth; ++i ){
		for( size_t b=0; b<nb; ++b ){
			interp().interp_coarse_fine(ilevel, *(ucs[b]), *(ufs[b]));
			if( m_smoother == opt::sm_gauss_seidel ){
				if( use_zoom_slab_gs )
					MUSIC::zoom_slab::gs_z_neg_meshvarbnd(
						/*box_owner=*/0, ufs[b], ffs[b], h, /*n_sweeps=*/1,
						MUSIC::zoom_slab::subcomm_for_box(ilevel, b),
						zoom_slab_halo_w);
				else
					GaussSeidel( h, ufs[b], ffs[b] );
			}
			else if( m_smoother == opt::sm_jacobi )
				Jacobi( h, ufs[b], ffs[b] );
			else if( m_smoother == opt::sm_sor )
				SOR( h, ufs[b], ffs[b] );
		}
	}

	// --- restrict each per-box uf into its parent uc ------------------
	for( size_t b=0; b<nb; ++b )
		m_gridop.restrict( *(ufs[b]), *(ucs[b]) );

	// Refresh halos so apply(uc) reads coherent values.
	if( !parent_multi && m_bperiodic && ilevel <= m_ilevelmin )
		make_periodic( ucs[0] );
	for( size_t b=0; b<nb; ++b )
		interp().interp_coarse_fine(ilevel, *(ucs[b]), *(ufs[b]));

	// --- apply Laplacian on fine + restrict (per-box tLu scratch) -----
	std::vector< meshvar_bnd* > tLus(nb, NULL);
	{
		// Per parent mesh, allocate tLu once and zero, then write child
		// contributions. Multiple children may share the same parent (when
		// parent is single-box) so we key by parent pointer.
		std::map< MeshvarBnd<T>*, meshvar_bnd* > tlu_for_parent;
		for( size_t b=0; b<nb; ++b ){
			MeshvarBnd<T> * uc = ucs[b];
			auto it = tlu_for_parent.find(uc);
			if( it == tlu_for_parent.end() ){
				meshvar_bnd * t = new meshvar_bnd(*uc, false);
				t->zero();
				tlu_for_parent[uc] = t;
				tLus[b] = t;
			} else {
				tLus[b] = it->second;
			}
		}

		for( size_t b=0; b<nb; ++b ){
			MeshvarBnd<T> * uf = ufs[b];
			meshvar_bnd & tLu = *(tLus[b]);
			int nx = uf->size(0), ny = uf->size(1), nz = uf->size(2);
			int oxp = uf->offset(0), oyp = uf->offset(1), ozp = uf->offset(2);
			#pragma omp parallel for
			for( int ix=0; ix<nx/2; ++ix ){
				int iix=2*ix;
				for( int iy=0,iiy=0; iy<ny/2; ++iy,iiy+=2 )
					for( int iz=0,iiz=0; iz<nz/2; ++iz,iiz+=2 )
						tLu(ix+oxp,iy+oyp,iz+ozp) = 0.125 * (
							 m_scheme.apply( *uf, iix,   iiy,   iiz   )
							+m_scheme.apply( *uf, iix,   iiy,   iiz+1 )
							+m_scheme.apply( *uf, iix,   iiy+1, iiz   )
							+m_scheme.apply( *uf, iix,   iiy+1, iiz+1 )
							+m_scheme.apply( *uf, iix+1, iiy,   iiz   )
							+m_scheme.apply( *uf, iix+1, iiy,   iiz+1 )
							+m_scheme.apply( *uf, iix+1, iiy+1, iiz   )
							+m_scheme.apply( *uf, iix+1, iiy+1, iiz+1 )
						)/h2;
			}
		}

		// --- restrict per-box source ff into its parent fc ------------
		for( size_t b=0; b<nb; ++b )
			m_gridop.restrict( *(ffs[b]), *(fcs[b]) );

		// --- FAS source update at child cells of fc -------------------
		for( size_t b=0; b<nb; ++b ){
			MeshvarBnd<T> * ff = ffs[b];
			MeshvarBnd<T> * uc = ucs[b];
			MeshvarBnd<T> * fc = fcs[b];
			meshvar_bnd & tLu = *(tLus[b]);
			int oi = ff->offset(0), oj = ff->offset(1), ok = ff->offset(2);
			int sx = (int)ff->size(0)/2, sy = (int)ff->size(1)/2, sz = (int)ff->size(2)/2;
			#pragma omp parallel for
			for( int ix=oi; ix<oi+sx; ++ix )
				for( int iy=oj; iy<oj+sy; ++iy )
					for( int iz=ok; iz<ok+sz; ++iz )
						(*fc)(ix,iy,iz) += ( tLu(ix,iy,iz)
						                    - (m_scheme.apply(*uc, ix,iy,iz)/(4.0*h2)) );
		}

		for( auto & kv : tlu_for_parent ){
			kv.second->deallocate();
			delete kv.second;
		}
	}

	// --- save parent uc state per parent mesh -------------------------
	std::map< MeshvarBnd<T>*, meshvar_bnd* > ucsave_for_parent;
	for( size_t b=0; b<nb; ++b ){
		MeshvarBnd<T> * uc = ucs[b];
		if( ucsave_for_parent.find(uc) == ucsave_for_parent.end() )
			ucsave_for_parent[uc] = new meshvar_bnd(*uc, true);
	}

	// --- recurse on the parent level ----------------------------------
	if( ilevel == 1 ){
		// Only single-box parent reachable at ilevel==1.
		(*(ucs[0]))(0,0,0) = 0.0;
	} else {
		// Standard descent: twoGrid() will itself dispatch to multibox
		// when num_boxes(ilevel-1) > 1.
		twoGrid( ilevel-1 );
	}

	// --- prolong correction back per-box ------------------------------
	std::map< MeshvarBnd<T>*, meshvar_bnd* > cc_for_parent;
	for( size_t b=0; b<nb; ++b ){
		MeshvarBnd<T> * uc = ucs[b];
		if( cc_for_parent.find(uc) != cc_for_parent.end() ) continue;
		meshvar_bnd * cc = new meshvar_bnd(*uc, false);
		meshvar_bnd & ucsave = *(ucsave_for_parent[uc]);
		#pragma omp parallel for
		for( int ix=0; ix<(int)cc->size(0); ++ix )
			for( int iy=0; iy<(int)cc->size(1); ++iy )
				for( int iz=0; iz<(int)cc->size(2); ++iz )
					(*cc)(ix,iy,iz) = (*uc)(ix,iy,iz) - ucsave(ix,iy,iz);
		if( !parent_multi && m_bperiodic && ilevel <= m_ilevelmin )
			make_periodic( cc );
		cc_for_parent[uc] = cc;
	}

	for( size_t b=0; b<nb; ++b ){
		meshvar_bnd & cc = *(cc_for_parent[ ucs[b] ]);
		m_gridop.prolong_add( cc, *(ufs[b]) );
	}

	for( auto & kv : ucsave_for_parent ){ kv.second->deallocate(); delete kv.second; }
	for( auto & kv : cc_for_parent     ){ kv.second->deallocate(); delete kv.second; }

	// --- post-smoothing sweeps (per box) ------------------------------
	for( unsigned i=0; i<m_npostsmooth; ++i ){
		for( size_t b=0; b<nb; ++b ){
			interp().interp_coarse_fine(ilevel, *(ucs[b]), *(ufs[b]));
			if( m_smoother == opt::sm_gauss_seidel ){
				if( use_zoom_slab_gs )
					MUSIC::zoom_slab::gs_z_neg_meshvarbnd(
						/*box_owner=*/0, ufs[b], ffs[b], h, /*n_sweeps=*/1,
						MUSIC::zoom_slab::subcomm_for_box(ilevel, b),
						zoom_slab_halo_w);
				else
					GaussSeidel( h, ufs[b], ffs[b] );
			}
			else if( m_smoother == opt::sm_jacobi )
				Jacobi( h, ufs[b], ffs[b] );
			else if( m_smoother == opt::sm_sor )
				SOR( h, ufs[b], ffs[b] );
		}
	}
}

// --------------------------------------------------------------------------
// Phase G.2b B.2b.2.1: SPMD twin of twoGrid_multibox.
//
// At np==1 every cross-rank primitive (broadcast/accumulate/release) is a
// no-op (see GridHierarchy<T>::broadcast_parents_to_child_owners — early
// returns when MUSIC::mpi::size()<=1) and owner_of_box(L,b) returns
// MUSIC::mpi::rank() for every b. Therefore the function reduces to the
// existing twoGrid_multibox body but with parent_idx-keyed scratch maps
// instead of pointer-keyed. Bit-identical at np=1.
//
// At np>1 the function executes its full SPMD body, but the call-site
// dispatcher (twoGrid_multibox + wpd outer scope) currently still routes
// np>1 through the existing rank-0-only path. B.2b.2.2 lifts that gate.
// --------------------------------------------------------------------------
template< class S, class I, class O, typename T >
void solver<S,I,O,T>::twoGrid_multibox_spmd( unsigned ilevel )
{
	size_t nb = m_pu->num_boxes(ilevel);
	if( nb <= 1 ){ twoGrid(ilevel); return; }

	const int my_rank = MUSIC::mpi::rank();

	const bool parent_multi = (m_pu->num_boxes(ilevel-1) > 1);

	#define MB_SPMD_TRACE(tag) do { \
		if( const char* _t = ::getenv("MB_SPMD_TRACE") ) if( _t[0]=='1' ){ \
			fprintf(stderr, "[rk=%d] mb_spmd L=%u %s\n", my_rank, ilevel, tag); fflush(stderr); \
		} \
	} while(0)
	MB_SPMD_TRACE("enter");

	double
		h  = 1.0/(1<<ilevel),
		c0 = -1.0/m_scheme.ccoeff(),
		h2 = h*h;
	(void)c0;

	// --- Phase 1: broadcast parent meshes to child_owners ---------------
	// At np==1: no-op. At np>1: parent_owner Isends; child_owners allocate
	// replicas in m_parent_replicas_[ilevel][parent_idx].
	//
	// IMPORTANT: only multi-box parent levels need replicas. Single-box
	// parents reside in the UNION mesh (m_pgrids[L-1]) — the per-box mesh
	// at single-box levels is an independent (non-aliased) buffer NOT
	// touched by the union-based recurse `twoGrid(L-1)`, so reading
	// parent_for_box at single-box levels yields stale state and the cc =
	// new_uc - ucsave correction collapses to ~0 → wrong solution.
	if( parent_multi ){
		MB_SPMD_TRACE("bcast_parents_u/begin");
		m_pu->broadcast_parents_to_child_owners(ilevel);
		MB_SPMD_TRACE("bcast_parents_u/end");
		MB_SPMD_TRACE("bcast_parents_f/begin");
		m_pf->broadcast_parents_to_child_owners(ilevel);
		MB_SPMD_TRACE("bcast_parents_f/end");
	}

	// Resolve per-box pointers (NULL on ranks that don't own that box).
	std::vector< MeshvarBnd<T>* > ufs(nb, NULL), ffs(nb, NULL);
	std::vector< MeshvarBnd<T>* > ucs(nb, NULL), fcs(nb, NULL);
	std::vector< size_t > pidx(nb, 0);
	for( size_t b=0; b<nb; ++b ){
		pidx[b] = parent_multi ? m_pu->parent_box_index(ilevel, b) : 0;
		if( m_pu->owner_of_box(ilevel, b) != my_rank ) continue;
		ufs[b] = m_pu->get_grid(ilevel, b);
		ffs[b] = m_pf->get_grid(ilevel, b);
		if( parent_multi ){
			ucs[b] = m_pu->parent_for_box(ilevel, b);  // per-box: local OR replica
			fcs[b] = m_pf->parent_for_box(ilevel, b);
		} else {
			ucs[b] = m_pu->get_grid(ilevel-1);  // UNION mesh — recurse operates here
			fcs[b] = m_pf->get_grid(ilevel-1);
		}
	}

	// Bridge gate (SPMD path).
	//
	// B.2b.3: policy=="world" now supported. All ranks reach twoGrid_multibox_spmd
	// via wpd_spmd, so a COMM_WORLD collective on the bridge no longer deadlocks.
	// The per-box loops below detect sub_size>1 and enter the bridge collectively;
	// non-owners pass NULL u/f and join the dims broadcast / scatter / gather.
	//
	// policy=="round_robin" is still gated off — sub_comm splits ranks, and the
	// owner rank (m_pu->owner_of_box returns WORLD rank) doesn't map cleanly to
	// the sub_comm's rank space without a translation table.
	const std::string& zoom_slab_policy = MUSIC::zoom_slab::subcomm_policy();
	const bool use_zoom_slab_gs =
		MUSIC::zoom_slab::smoother_enabled()
		&& (S::order == 2)
		&& ( MUSIC::mpi::size() == 1
		     || zoom_slab_policy == "self"
		     || zoom_slab_policy == "world" );
	const int zoom_slab_halo_w = 1;
	{
		static bool s_bridge_logged_spmd = false;
		if( use_zoom_slab_gs && !s_bridge_logged_spmd ){
			s_bridge_logged_spmd = true;
			LOGINFO("G.2b B.2b.6: zoom-slab GS+restrict+prolong_add+apply bridges active (SPMD path) at "
			        "ilevel=%u nb=%zu (np=%d, S::order=2, policy=\"%s\")",
			        (unsigned)ilevel, nb, MUSIC::mpi::size(),
			        zoom_slab_policy.c_str());
		}
	}

	// --- pre-smoothing sweeps (per box, owner-gated; bridge can be collective)
	// B.5.4.a: when keep-in-slab smoothing is enabled, attempt the N-iter mega-
	// primitive once per box (collective on sub_comm). On success, skip that box
	// in the subsequent per-iter loop. Bit-identical to N iterations of
	// (interp_coarse_fine_meshvarbnd + gs_z_neg_meshvarbnd).
	//
	// B.5.4.b: when keep-in-slab u-restrict is *also* enabled, use the
	// keep_slab variant — the padded-cluster interior buffer is retained per
	// box and consumed by restrict_meshvarbnd_from_padded_slab at the u-restrict
	// site immediately below, eliminating one gather+scatter round-trip.
	std::vector<bool> pre_keep_slab_done(nb, false);
	std::vector<bool> pre_keep_slab_padded(nb, false);
	std::vector< MUSIC::zoom_slab::ZoomSlabLayout > pre_padded_Lf(nb);
	std::vector< std::vector<T> > pre_padded_uf_int(nb);
	const bool keep_slab_urestrict_on =
		MUSIC::zoom_slab::keep_slab_smooth_enabled()
		&& MUSIC::zoom_slab::keep_slab_urestrict_enabled();
	if( MUSIC::zoom_slab::keep_slab_smooth_enabled() && m_npresmooth > 0 ){
		for( size_t b=0; b<nb; ++b ){
			MPI_Comm sub_comm = MUSIC::zoom_slab::subcomm_for_box(ilevel, b);
			int sub_size = 1;
#ifdef USE_MPI
			if( use_zoom_slab_gs && m_smoother == opt::sm_gauss_seidel )
				MPI_Comm_size(sub_comm, &sub_size);
#endif
			const bool collective_bridge =
				use_zoom_slab_gs && m_smoother == opt::sm_gauss_seidel && sub_size > 1;
			if( !collective_bridge ) continue;
			const int owner_world = (int)m_pu->owner_of_box(ilevel, b);
			if( keep_slab_urestrict_on ) {
				pre_keep_slab_done[b] = MUSIC::zoom_slab::smooth_pre_post_n_meshvarbnd_keep_slab(
					owner_world, ucs[b], ufs[b], ffs[b], h,
					(int)m_npresmooth, sub_comm,
					pre_padded_Lf[b], pre_padded_uf_int[b]);
				pre_keep_slab_padded[b] = pre_keep_slab_done[b];
			} else {
				pre_keep_slab_done[b] = MUSIC::zoom_slab::smooth_pre_post_n_meshvarbnd(
					owner_world, ucs[b], ufs[b], ffs[b], h,
					(int)m_npresmooth, sub_comm);
			}
		}
	}
	for( unsigned i=0; i<m_npresmooth; ++i ){
		for( size_t b=0; b<nb; ++b ){
			if( pre_keep_slab_done[b] ) continue;
			MPI_Comm sub_comm = MUSIC::zoom_slab::subcomm_for_box(ilevel, b);
			int sub_size = 1;
#ifdef USE_MPI
			if( use_zoom_slab_gs && m_smoother == opt::sm_gauss_seidel )
				MPI_Comm_size(sub_comm, &sub_size);
#endif
			const bool collective_bridge =
				use_zoom_slab_gs && m_smoother == opt::sm_gauss_seidel && sub_size > 1;

			if( collective_bridge ){
				// All sub_comm ranks must enter the bridge. Owner runs interp
				// + supplies real u/f; non-owners pass NULL.
				const int owner_world = (int)m_pu->owner_of_box(ilevel, b);
				// B.5.3c: dispatch interp_coarse_fine through the z-slab
				// composite bridge when geometry allows; falls back to owner-
				// local interp_coarse_fine if the bridge declines (parity,
				// alignment, or uc reach insufficient).
				bool did_interp = MUSIC::zoom_slab::interp_coarse_fine_meshvarbnd(
					owner_world, ucs[b], ufs[b], sub_comm);
				if( !did_interp && ufs[b] )
					interp().interp_coarse_fine(ilevel, *(ucs[b]), *(ufs[b]));
				// For policy=="world" sub_comm == COMM_WORLD so sub-rank == world-rank.
				MUSIC::zoom_slab::gs_z_neg_meshvarbnd(
					owner_world, ufs[b], ffs[b], h, /*n_sweeps=*/1,
					sub_comm, zoom_slab_halo_w);
			} else {
				if( !ufs[b] ) continue;
				interp().interp_coarse_fine(ilevel, *(ucs[b]), *(ufs[b]));
				if( m_smoother == opt::sm_gauss_seidel ){
					if( use_zoom_slab_gs )
						MUSIC::zoom_slab::gs_z_neg_meshvarbnd(
							/*box_owner=*/0, ufs[b], ffs[b], h, /*n_sweeps=*/1,
							sub_comm, zoom_slab_halo_w);
					else
						GaussSeidel( h, ufs[b], ffs[b] );
				}
				else if( m_smoother == opt::sm_jacobi )
					Jacobi( h, ufs[b], ffs[b] );
				else if( m_smoother == opt::sm_sor )
					SOR( h, ufs[b], ffs[b] );
			}
		}
	}

	// --- restrict per-box uf into its parent uc (writes to replica or local)
	//
	// B.2b.4: when sub_size>1 and the bridge accepts the box geometry, the
	// per-box fine mesh is scattered across sub_comm, restricted in z-slab form,
	// and gathered back to the box_owner — which then writes into Vc at the
	// fine-box's parent offset. Falls back to the local op (on box_owner only)
	// when sub_size==1 or alignment is unsuitable (bridge returns false).
	//
	// B.5.4.b: when pre-smooth's keep_slab variant succeeded, consume the
	// padded-cluster interior buffer directly via restrict_meshvarbnd_from_-
	// padded_slab (perimeter=2). Skips the otherwise-needed fine gather +
	// fine re-scatter that the un-padded bridge would do.
	for( size_t b=0; b<nb; ++b ){
		MPI_Comm sub_comm = MUSIC::zoom_slab::subcomm_for_box(ilevel, b);
		int sub_size = 1;
#ifdef USE_MPI
		if( use_zoom_slab_gs )
			MPI_Comm_size(sub_comm, &sub_size);
#endif
		const bool collective_bridge = use_zoom_slab_gs && sub_size > 1;
		if( collective_bridge ){
			const int owner_world = (int)m_pu->owner_of_box(ilevel, b);
			bool did_bridge = false;
			if( pre_keep_slab_padded[b] ) {
				// Broadcast fine-mesh offsets (= coarse-write offsets for restrict
				// per the multibox uf.offset() convention) from owner to sub_comm.
				int voff[3] = {0, 0, 0};
				if( ufs[b] ) {
					voff[0] = ufs[b]->offset(0);
					voff[1] = ufs[b]->offset(1);
					voff[2] = ufs[b]->offset(2);
				}
#ifdef USE_MPI
				MPI_Bcast(voff, 3, MPI_INT, owner_world, sub_comm);
#endif
				did_bridge = MUSIC::zoom_slab::restrict_meshvarbnd_from_padded_slab(
					owner_world, pre_padded_Lf[b], pre_padded_uf_int[b].data(),
					/*perimeter=*/2, ucs[b],
					voff[0], voff[1], voff[2], sub_comm);
				if( did_bridge ) {
					// Padded slab is consumed; drop the (potentially large)
					// buffer now to reclaim RAM before u-restrict at coarser
					// levels recurses.
					std::vector<T>().swap(pre_padded_uf_int[b]);
				}
			}
			if( !did_bridge ) {
				did_bridge = MUSIC::zoom_slab::restrict_meshvarbnd(
					owner_world, ufs[b], ucs[b], sub_comm);
			}
			if( !did_bridge && ufs[b] )
				m_gridop.restrict( *(ufs[b]), *(ucs[b]) );
		} else {
			if( !ufs[b] ) continue;
			m_gridop.restrict( *(ufs[b]), *(ucs[b]) );
		}
	}
	// Multi-box parent: consolidate child writes back to parent_owner, then
	// re-broadcast so non-owners see the consolidated parent for halo reads.
	// Single-box parent: writes landed in union mesh directly — no replica
	// machinery involved.
	if( parent_multi ){
		MB_SPMD_TRACE("accumulate_u/begin");
		m_pu->accumulate_children_to_parents(ilevel);
		MB_SPMD_TRACE("accumulate_u/end");
		MB_SPMD_TRACE("rebcast_u/begin");
		m_pu->broadcast_parents_to_child_owners(ilevel);
		MB_SPMD_TRACE("rebcast_u/end");
	}

	// Refresh halos so apply(uc) reads coherent values.
	if( !parent_multi && m_bperiodic && ilevel <= m_ilevelmin )
		make_periodic( m_pu->get_grid(ilevel-1) );
	// B.5.3c: collective dispatch — every rank in sub_comm must enter the
	// bridge for every box (non-owners pass NULL). Re-resolve ucs[b] on the
	// owner first since broadcast_parents_to_child_owners ran above and may
	// have rebound the per-box parent.
	for( size_t b=0; b<nb; ++b ){
		MPI_Comm sub_comm = MUSIC::zoom_slab::subcomm_for_box(ilevel, b);
		int sub_size = 1;
#ifdef USE_MPI
		if( use_zoom_slab_gs )
			MPI_Comm_size(sub_comm, &sub_size);
#endif
		const bool collective_bridge = use_zoom_slab_gs && sub_size > 1;

		if( ufs[b] && parent_multi )
			ucs[b] = m_pu->parent_for_box(ilevel, b);

		if( collective_bridge ){
			const int owner_world = (int)m_pu->owner_of_box(ilevel, b);
			bool did_interp = MUSIC::zoom_slab::interp_coarse_fine_meshvarbnd(
				owner_world, ucs[b], ufs[b], sub_comm);
			if( !did_interp && ufs[b] )
				interp().interp_coarse_fine(ilevel, *(ucs[b]), *(ufs[b]));
		} else {
			if( !ufs[b] ) continue;
			interp().interp_coarse_fine(ilevel, *(ucs[b]), *(ufs[b]));
		}
	}

	// --- apply Laplacian + restrict (per-parent tLu scratch, parent_idx-keyed)
	std::vector< meshvar_bnd* > tLus(nb, NULL);
	{
		std::map< size_t, meshvar_bnd* > tlu_for_parent;
		for( size_t b=0; b<nb; ++b ){
			if( !ufs[b] ) continue;
			const size_t pi = pidx[b];
			auto it = tlu_for_parent.find(pi);
			if( it == tlu_for_parent.end() ){
				meshvar_bnd * t = new meshvar_bnd(*(ucs[b]), false);
				t->zero();
				tlu_for_parent[pi] = t;
				tLus[b] = t;
			} else {
				tLus[b] = it->second;
			}
		}

		// B.2b.6: apply Laplacian + restrict (8-cell average) → tLu at parent
		// offset. When the collective bridge is active, owner computes apply(uf)/h2
		// into a scratch MeshvarBnd<T> (same geometry as uf — inherits offset),
		// then dispatches through restrict_meshvarbnd which scatters the scratch
		// across sub_comm, averages 8 children, and writes coarse cells back to
		// tLu at (ic+ox, jc+oy, kc+oz). Non-owners pass NULL scratch and tLu.
		// When the bridge is inactive (sub_size==1) or falls back (alignment
		// failure), we use the original inlined fused apply+average loop on
		// owner — bit-identical and avoids the scratch allocation.
		for( size_t b=0; b<nb; ++b ){
			MPI_Comm sub_comm = MUSIC::zoom_slab::subcomm_for_box(ilevel, b);
			int sub_size = 1;
#ifdef USE_MPI
			if( use_zoom_slab_gs )
				MPI_Comm_size(sub_comm, &sub_size);
#endif
			const bool collective_bridge = use_zoom_slab_gs && sub_size > 1;

			if( collective_bridge ){
				const int owner_world = (int)m_pu->owner_of_box(ilevel, b);

				// B.5.2-prod: try the fused apply_meshvarbnd_to_slab +
				// restrict_meshvarbnd_from_slab path. Replaces the gather→scatter
				// round-trip between apply_meshvarbnd and restrict_meshvarbnd
				// with an MPI_Alltoallv redistribute over per-rank slabs.
				// Returns false on order!=2, dim mismatch, or alignment failure
				// — caller falls through to the existing B.5.1 + B.2b.4 path.
				bool did_fused = false;
				if( S::order == 2 ){
					int sub_rank = 0;
#ifdef USE_MPI
					MPI_Comm_rank(sub_comm, &sub_rank);
#endif
					int hdr[6] = {0,0,0,0,0,0};
					if( ufs[b] && sub_rank == owner_world ){
						hdr[0] = (int)ufs[b]->size(0);
						hdr[1] = (int)ufs[b]->size(1);
						hdr[2] = (int)ufs[b]->size(2);
						hdr[3] = ufs[b]->offset(0);
						hdr[4] = ufs[b]->offset(1);
						hdr[5] = ufs[b]->offset(2);
					}
#ifdef USE_MPI
					MPI_Bcast(hdr, 6, MPI_INT, owner_world, sub_comm);
#endif
					const int nxf = hdr[0], nyf = hdr[1], nzf = hdr[2];
					const int oxc = hdr[3], oyc = hdr[4], ozc = hdr[5];
					if( nxf > 0 && nyf > 0 && nzf > 0
					    && (nxf & 1) == 0 && (nyf & 1) == 0 && (nzf & 1) == 0
					    && nzf % (2 * sub_size) == 0
					    && (nzf / (2 * sub_size)) >= 1
					    && zoom_slab_halo_w >= 1
					    && zoom_slab_halo_w <= ((nzf + 2) / sub_size) )
					{
						MUSIC::zoom_slab::ZoomSlabLayout Lf =
							MUSIC::zoom_slab::make_layout(
								sub_comm, /*cluster_id=*/0, ilevel,
								0, 0, 0, nxf, nyf, nzf, zoom_slab_halo_w);
						std::vector<T> my_apply_int(
							MUSIC::zoom_slab::local_interior_size(Lf), T(0));
						const bool did_apply = MUSIC::zoom_slab::apply_meshvarbnd_to_slab(
							owner_world,
							(sub_rank == owner_world) ? ufs[b] : (const MeshvarBnd<T>*)NULL,
							Lf, my_apply_int.data(), h2, sub_comm);
						if( did_apply ){
							const bool did_restr = MUSIC::zoom_slab::restrict_meshvarbnd_from_slab(
								owner_world, Lf, my_apply_int.data(),
								(sub_rank == owner_world) ? tLus[b] : (MeshvarBnd<T>*)NULL,
								oxc, oyc, ozc, sub_comm);
							did_fused = did_restr;
						}
					}
					static bool s_b52_prod_logged = false;
					if( !s_b52_prod_logged && did_fused ){
						LOGINFO("G.2b B.5.2-prod: fused apply_to_slab + "
						        "restrict_from_slab active (SPMD path) at "
						        "ilevel=%u nb=%zu (sub_size=%d) — gather/scatter "
						        "round-trip eliminated",
						        ilevel, nb, sub_size);
						s_b52_prod_logged = true;
					}
				}

				if( !did_fused ){
					MeshvarBnd<T> * apply_uf = NULL;
					if( ufs[b] ){
						MeshvarBnd<T> * uf = ufs[b];
						apply_uf = new MeshvarBnd<T>(*uf, false);
						apply_uf->zero();
					}
					// B.5.1: try sub_comm-collective slab apply (apply_z_slab via
					// apply_meshvarbnd bridge). Falls back to the rank-0-local
					// m_scheme.apply loop on box_owner if the bridge returns false
					// (sub_size==1 path doesn't reach here; halo_w too large is
					// the main fallback trigger).
					bool did_slab_apply = false;
					if( S::order == 2 ){
						did_slab_apply = MUSIC::zoom_slab::apply_meshvarbnd(
							owner_world, ufs[b], apply_uf, h2,
							sub_comm, zoom_slab_halo_w);
					}
					if( !did_slab_apply && ufs[b] && apply_uf ){
						MeshvarBnd<T> * uf = ufs[b];
						const int nxf = (int)uf->size(0);
						const int nyf = (int)uf->size(1);
						const int nzf = (int)uf->size(2);
						#pragma omp parallel for
						for( int i=0; i<nxf; ++i )
							for( int j=0; j<nyf; ++j )
								for( int k=0; k<nzf; ++k )
									(*apply_uf)(i,j,k) = m_scheme.apply( *uf, i, j, k ) / h2;
					}
					const bool did_bridge = MUSIC::zoom_slab::restrict_meshvarbnd(
						owner_world, apply_uf, tLus[b], sub_comm);
					if( !did_bridge && ufs[b] && apply_uf ){
						m_gridop.restrict( *apply_uf, *(tLus[b]) );
					}
					if( apply_uf ){ apply_uf->deallocate(); delete apply_uf; }
				}
			} else {
				if( !ufs[b] ) continue;
				MeshvarBnd<T> * uf = ufs[b];
				meshvar_bnd & tLu = *(tLus[b]);
				int nx = uf->size(0), ny = uf->size(1), nz = uf->size(2);
				int oxp = uf->offset(0), oyp = uf->offset(1), ozp = uf->offset(2);
				#pragma omp parallel for
				for( int ix=0; ix<nx/2; ++ix ){
					int iix=2*ix;
					for( int iy=0,iiy=0; iy<ny/2; ++iy,iiy+=2 )
						for( int iz=0,iiz=0; iz<nz/2; ++iz,iiz+=2 )
							tLu(ix+oxp,iy+oyp,iz+ozp) = 0.125 * (
								 m_scheme.apply( *uf, iix,   iiy,   iiz   )
								+m_scheme.apply( *uf, iix,   iiy,   iiz+1 )
								+m_scheme.apply( *uf, iix,   iiy+1, iiz   )
								+m_scheme.apply( *uf, iix,   iiy+1, iiz+1 )
								+m_scheme.apply( *uf, iix+1, iiy,   iiz   )
								+m_scheme.apply( *uf, iix+1, iiy,   iiz+1 )
								+m_scheme.apply( *uf, iix+1, iiy+1, iiz   )
								+m_scheme.apply( *uf, iix+1, iiy+1, iiz+1 )
							)/h2;
				}
			}
		}

		// restrict ff into fc (writes to parent local or replica)
		//
		// B.2b.4 bridge dispatch (same gating pattern as the u-restrict above).
		for( size_t b=0; b<nb; ++b ){
			MPI_Comm sub_comm = MUSIC::zoom_slab::subcomm_for_box(ilevel, b);
			int sub_size = 1;
#ifdef USE_MPI
			if( use_zoom_slab_gs )
				MPI_Comm_size(sub_comm, &sub_size);
#endif
			const bool collective_bridge = use_zoom_slab_gs && sub_size > 1;
			if( collective_bridge ){
				const int owner_world = (int)m_pu->owner_of_box(ilevel, b);
				const bool did_bridge = MUSIC::zoom_slab::restrict_meshvarbnd(
					owner_world, ffs[b], fcs[b], sub_comm);
				if( !did_bridge && ufs[b] )
					m_gridop.restrict( *(ffs[b]), *(fcs[b]) );
			} else {
				if( !ufs[b] ) continue;
				m_gridop.restrict( *(ffs[b]), *(fcs[b]) );
			}
		}
		// FAS source update on fc child-region (reads uc, tLu; writes fc)
		for( size_t b=0; b<nb; ++b ){
			if( !ufs[b] ) continue;
			MeshvarBnd<T> * ff = ffs[b];
			MeshvarBnd<T> * uc = ucs[b];
			MeshvarBnd<T> * fc = fcs[b];
			meshvar_bnd & tLu = *(tLus[b]);
			int oi = ff->offset(0), oj = ff->offset(1), ok = ff->offset(2);
			int sx = (int)ff->size(0)/2, sy = (int)ff->size(1)/2, sz = (int)ff->size(2)/2;
			#pragma omp parallel for
			for( int ix=oi; ix<oi+sx; ++ix )
				for( int iy=oj; iy<oj+sy; ++iy )
					for( int iz=ok; iz<ok+sz; ++iz )
						(*fc)(ix,iy,iz) += ( tLu(ix,iy,iz)
						                    - (m_scheme.apply(*uc, ix,iy,iz)/(4.0*h2)) );
		}

		for( auto & kv : tlu_for_parent ){
			kv.second->deallocate();
			delete kv.second;
		}
	}
	// Multi-box parent: consolidate fc contributions back to parent_owner of fh.
	// Single-box parent: fc writes landed in union mesh directly.
	if( parent_multi ){
		MB_SPMD_TRACE("accumulate_f/begin");
		m_pf->accumulate_children_to_parents(ilevel);
		MB_SPMD_TRACE("accumulate_f/end");
	}
	// NOTE: uh parent state must also reflect any non-restrict accumulate.
	// In current logic per-box ops above only ADD into fc; uc was only read.
	// So uh has no pending child writes here — DO NOT re-broadcast uh
	// (already current from the earlier re-broadcast post halo refresh).

	// --- save parent uc state (per parent_idx, on whoever holds the parent)
	std::map< size_t, meshvar_bnd* > ucsave_for_parent;
	for( size_t b=0; b<nb; ++b ){
		if( !ufs[b] ) continue;
		const size_t pi = pidx[b];
		if( ucsave_for_parent.find(pi) == ucsave_for_parent.end() ){
			// Save from the local view of parent (replica or local).
			MeshvarBnd<T> * uc = ucs[b];
			ucsave_for_parent[pi] = new meshvar_bnd(*uc, true);  // deep copy
		}
	}

	// --- recurse on the parent level ---------------------------------
	// B.2b.2.2.1.b: coordinated single-box-parent recurse.
	//   - ilevel == 1   : pin gauge at corner cell on the box owner; nothing to broadcast.
	//   - parent_multi  : SPMD recurse — every rank calls twoGrid(L-1). The
	//                     parent state is held in per-box meshes / replicas;
	//                     non-owners participate via the replica machinery.
	//   - single-box    : only parent_owner runs the SERIAL twoGrid(L-1) on
	//     parent       the union mesh. Non-owners barrier (implicit in the
	//                     subsequent broadcast_union_mesh_from_owner collective).
	//                     After recurse, the union u must be republished so
	//                     downstream reads (cc computation, post-smooth halo
	//                     interp) see the corrected state.
	//
	// B.2b.2.2.1.d: at np>1 single-box-parent levels, each child's restrict +
	// FAS write went into its OWNER's local copy of m_pu/m_pf->get_grid(L-1).
	// Before the serial recurse fires on parent_owner_L1, gather each
	// non-owned child's parent-footprint contributions onto parent_owner_L1.
	// At np==1 these primitives are no-ops; at multi-box parent the replica
	// accumulate path (above) already handled consolidation.
	if( ilevel == 1 ){
		MB_SPMD_TRACE("pin_gauge");
		if( m_pu->owner_of_box(0, 0) == my_rank ){
			MeshvarBnd<T> * uc0 = m_pu->get_grid(0);
			(*uc0)(0,0,0) = 0.0;
		}
	} else if( parent_multi ){
		MB_SPMD_TRACE("recurse_multi/begin");
		twoGrid( ilevel-1 );
		MB_SPMD_TRACE("recurse_multi/end");
	} else {
		const int parent_owner_L1 = m_pu->owner_of_box(ilevel-1, 0);
		MB_SPMD_TRACE("consolidate_u/begin");
		m_pu->consolidate_child_writes_to_parent_owner(ilevel);
		MB_SPMD_TRACE("consolidate_u/end");
		MB_SPMD_TRACE("consolidate_f/begin");
		m_pf->consolidate_child_writes_to_parent_owner(ilevel);
		MB_SPMD_TRACE("consolidate_f/end");
		if( parent_owner_L1 == my_rank ){
			MB_SPMD_TRACE("serial_twoGrid_parent/begin");
			twoGrid( ilevel-1 );
			MB_SPMD_TRACE("serial_twoGrid_parent/end");
		} else {
			MB_SPMD_TRACE("await_parent_bcast");
		}
		MB_SPMD_TRACE("bcast_union_parent/begin");
		m_pu->broadcast_union_mesh_from_owner(ilevel-1);
		MB_SPMD_TRACE("bcast_union_parent/end");
	}

	// --- re-broadcast updated parent uc to child_owners (multi-box parent only)
	// At multi-box parent: each cross-owner rank needs a fresh replica of the
	// updated parent to compute cc = new_uc - ucsave and feed prolong_add /
	// halo refreshes in post-smoothing.
	// At single-box parent: ucs[b] still points at the UNION mesh which the
	// recurse already updated in place — nothing to do.
	if( parent_multi ){
		MB_SPMD_TRACE("post_recurse_rebcast/begin");
		m_pu->broadcast_parents_to_child_owners(ilevel);
		MB_SPMD_TRACE("post_recurse_rebcast/end");
		for( size_t b=0; b<nb; ++b ){
			if( !ufs[b] ) continue;
			ucs[b] = m_pu->parent_for_box(ilevel, b);
		}
	}

	// --- compute cc = new_uc - ucsave and prolong_add (per box) --------
	std::map< size_t, meshvar_bnd* > cc_for_parent;
	for( size_t b=0; b<nb; ++b ){
		if( !ufs[b] ) continue;
		const size_t pi = pidx[b];
		if( cc_for_parent.find(pi) != cc_for_parent.end() ) continue;
		MeshvarBnd<T> * uc = ucs[b];
		meshvar_bnd * cc = new meshvar_bnd(*uc, false);
		meshvar_bnd & ucsave = *(ucsave_for_parent[pi]);
		#pragma omp parallel for
		for( int ix=0; ix<(int)cc->size(0); ++ix )
			for( int iy=0; iy<(int)cc->size(1); ++iy )
				for( int iz=0; iz<(int)cc->size(2); ++iz )
					(*cc)(ix,iy,iz) = (*uc)(ix,iy,iz) - ucsave(ix,iy,iz);
		if( !parent_multi && m_bperiodic && ilevel <= m_ilevelmin )
			make_periodic( cc );
		cc_for_parent[pi] = cc;
	}

	// B.5.4.c: when fusing prolong_add into the post-smoother, defer the
	// collective-box prolong_add to the post-smooth site (a single fused
	// scatter/gather). Requires keep-slab smoothing on and post-sweeps > 0.
	const bool fuse_prolong =
		MUSIC::zoom_slab::keep_slab_prolong_enabled() &&
		MUSIC::zoom_slab::keep_slab_smooth_enabled() &&
		m_npostsmooth > 0;

	// B.2b.5 bridge dispatch (same gating pattern as the restrict sites above).
	// When sub_size>1 the per-box coarse correction cc is scattered across
	// sub_comm, prolong_add'd onto the per-box fine mesh in z-slab form, and
	// the fine result is gathered back to the box_owner. Non-owners pass NULL
	// for both cc and ufs[b] — the bridge broadcasts dimensions from the owner.
	for( size_t b=0; b<nb; ++b ){
		MPI_Comm sub_comm = MUSIC::zoom_slab::subcomm_for_box(ilevel, b);
		int sub_size = 1;
#ifdef USE_MPI
		if( use_zoom_slab_gs )
			MPI_Comm_size(sub_comm, &sub_size);
#endif
		const bool collective_bridge = use_zoom_slab_gs && sub_size > 1;
		if( collective_bridge ){
			// B.5.4.c: fused path handles prolong_add at the post-smooth site.
			if( fuse_prolong ) continue;
			const int owner_world = (int)m_pu->owner_of_box(ilevel, b);
			meshvar_bnd * cc_ptr = NULL;
			if( ufs[b] ){
				auto it = cc_for_parent.find(pidx[b]);
				if( it != cc_for_parent.end() ) cc_ptr = it->second;
			}
			const bool did_bridge = MUSIC::zoom_slab::prolong_add_meshvarbnd(
				owner_world, cc_ptr, ufs[b], sub_comm);
			if( !did_bridge && ufs[b] && cc_ptr )
				m_gridop.prolong_add( *cc_ptr, *(ufs[b]) );
		} else {
			if( !ufs[b] ) continue;
			meshvar_bnd & cc = *(cc_for_parent[ pidx[b] ]);
			m_gridop.prolong_add( cc, *(ufs[b]) );
		}
	}

	for( auto & kv : ucsave_for_parent ){ kv.second->deallocate(); delete kv.second; }
	// B.5.4.c: cc_for_parent stays alive past here when fusing — the fused
	// prolong_add_then_smooth call at the post-smooth site still needs it.
	// Deleted below after the keep-slab post-smooth site.

	// --- post-smoothing sweeps (per box, owner-gated; bridge can be collective)
	// B.5.4.a: see pre-smoothing comment.
	std::vector<bool> post_keep_slab_done(nb, false);
	if( MUSIC::zoom_slab::keep_slab_smooth_enabled() && m_npostsmooth > 0 ){
		for( size_t b=0; b<nb; ++b ){
			MPI_Comm sub_comm = MUSIC::zoom_slab::subcomm_for_box(ilevel, b);
			int sub_size = 1;
#ifdef USE_MPI
			if( use_zoom_slab_gs && m_smoother == opt::sm_gauss_seidel )
				MPI_Comm_size(sub_comm, &sub_size);
#endif
			const bool collective_bridge =
				use_zoom_slab_gs && m_smoother == opt::sm_gauss_seidel && sub_size > 1;
			if( !collective_bridge ) continue;
			const int owner_world = (int)m_pu->owner_of_box(ilevel, b);
			if( fuse_prolong ){
				// B.5.4.c: fused prolong_add + post-smooth in one scatter/gather.
				meshvar_bnd * cc_ptr = NULL;
				if( ufs[b] ){
					auto it = cc_for_parent.find(pidx[b]);
					if( it != cc_for_parent.end() ) cc_ptr = it->second;
				}
				const bool fused = MUSIC::zoom_slab::prolong_add_then_smooth_n_meshvarbnd(
					owner_world, cc_ptr, ucs[b], ufs[b], ffs[b], h,
					(int)m_npostsmooth, sub_comm);
				if( fused ){
					post_keep_slab_done[b] = true;
				} else {
					// Fused fell back: do the prolong_add now (collective bridge,
					// matching the deferred site), leave smoothing to the per-iter
					// loop below (post_keep_slab_done[b] stays false).
					const bool did_bridge = MUSIC::zoom_slab::prolong_add_meshvarbnd(
						owner_world, cc_ptr, ufs[b], sub_comm);
					if( !did_bridge && ufs[b] && cc_ptr )
						m_gridop.prolong_add( *cc_ptr, *(ufs[b]) );
				}
			} else {
				post_keep_slab_done[b] = MUSIC::zoom_slab::smooth_pre_post_n_meshvarbnd(
					owner_world, ucs[b], ufs[b], ffs[b], h,
					(int)m_npostsmooth, sub_comm);
			}
		}
	}
	for( auto & kv : cc_for_parent ){ kv.second->deallocate(); delete kv.second; }
	for( unsigned i=0; i<m_npostsmooth; ++i ){
		for( size_t b=0; b<nb; ++b ){
			if( post_keep_slab_done[b] ) continue;
			MPI_Comm sub_comm = MUSIC::zoom_slab::subcomm_for_box(ilevel, b);
			int sub_size = 1;
#ifdef USE_MPI
			if( use_zoom_slab_gs && m_smoother == opt::sm_gauss_seidel )
				MPI_Comm_size(sub_comm, &sub_size);
#endif
			const bool collective_bridge =
				use_zoom_slab_gs && m_smoother == opt::sm_gauss_seidel && sub_size > 1;

			if( collective_bridge ){
				const int owner_world = (int)m_pu->owner_of_box(ilevel, b);
				// B.5.3c: collective interp_coarse_fine dispatch (see pre-smoothing).
				bool did_interp = MUSIC::zoom_slab::interp_coarse_fine_meshvarbnd(
					owner_world, ucs[b], ufs[b], sub_comm);
				if( !did_interp && ufs[b] )
					interp().interp_coarse_fine(ilevel, *(ucs[b]), *(ufs[b]));
				MUSIC::zoom_slab::gs_z_neg_meshvarbnd(
					owner_world, ufs[b], ffs[b], h, /*n_sweeps=*/1,
					sub_comm, zoom_slab_halo_w);
			} else {
				if( !ufs[b] ) continue;
				interp().interp_coarse_fine(ilevel, *(ucs[b]), *(ufs[b]));
				if( m_smoother == opt::sm_gauss_seidel ){
					if( use_zoom_slab_gs )
						MUSIC::zoom_slab::gs_z_neg_meshvarbnd(
							/*box_owner=*/0, ufs[b], ffs[b], h, /*n_sweeps=*/1,
							sub_comm, zoom_slab_halo_w);
					else
						GaussSeidel( h, ufs[b], ffs[b] );
				}
				else if( m_smoother == opt::sm_jacobi )
					Jacobi( h, ufs[b], ffs[b] );
				else if( m_smoother == opt::sm_sor )
					SOR( h, ufs[b], ffs[b] );
			}
		}
	}

	// --- final release: drain any lingering replicas from the last broadcast
	// (no-op when parent_multi=false since broadcast was skipped above).
	if( parent_multi )
		m_pu->release_parent_replicas(ilevel);
}

template< class S, class I, class O, typename T >
double solver<S,I,O,T>::compute_error( const MeshvarBnd<T>& u, const MeshvarBnd<T>& f, int ilevel )
{
	int 
		nx = u.size(0), 
		ny = u.size(1), 
		nz = u.size(2);
	
	double err = 0.0, err2 = 0.0;
	size_t count = 0;

	double h = 1.0/(1ul<<ilevel), h2=h*h;
	
	#pragma omp parallel for reduction(+:err,count)
	for( int ix=0; ix<nx; ++ix )
		for( int iy=0; iy<ny; ++iy )
			for( int iz=0; iz<nz; ++iz )
			  if( true )//fabs(unew(ix,iy,iz)) > 0.0 )//&& u(ix,iy,iz) != unew(ix,iy,iz) )
				{
				  //err += fabs(1.0 - (double)u(ix,iy,iz)/(double)unew(ix,iy,iz));
				  /*err += fabs(((double)m_scheme.apply( u, ix, iy, iz )/h2 + (double)(f(ix,iy,iz)) ));
				    err2 += fabs((double)f(ix,iy,iz));*/

				  err += fabs( (double)m_scheme.apply( u, ix, iy, iz )/h2/(double)(f(ix,iy,iz)) + 1.0 );
					++count;
				}
	
	  if( count != 0 )
	    err /= count; 
	  
	return err;
}

template< class S, class I, class O, typename T >
double solver<S,I,O,T>::compute_error( const GridHierarchy<T>& uh, const GridHierarchy<T>& fh, bool verbose )
{
	double maxerr = 0.0;

	for( unsigned ilevel=uh.levelmin(); ilevel <= uh.levelmax(); ++ilevel )
	{
		double h = 1.0/(1ul<<ilevel), h2=h*h;

		// D.3.3: at multi-box levels iterate per-box; at single-box levels use union.
		const size_t nb = uh.num_boxes(ilevel);
		const bool multi = (nb > 1);
		// B.2b.2.2.1.d: only when SPMD MG is active do all ranks enter solve()
		// concurrently — then workers' union F at single-box levels is empty
		// (only the level-owner holds it). Under classic wpd, only rank 0 reaches
		// here at all, so no skip / no Allreduce is needed.
		const bool spmd_mg = MUSIC::poisson::spmd_mg_is_active();
		const int  single_box_owner = multi ? -1 : (int)uh.owner_of_box(ilevel, 0);
		const bool single_box_skip  = spmd_mg && (!multi) && (single_box_owner != MUSIC::mpi::rank());

		double err = 0.0, mean_res = 0.0;
		size_t count = 0;

		if( !single_box_skip ) for( size_t b=0; b < (multi ? nb : 1); ++b ){
			const MeshvarBnd<T> * pu = multi ? uh.get_grid(ilevel, b) : uh.get_grid(ilevel);
			const MeshvarBnd<T> * pf = multi ? fh.get_grid(ilevel, b) : fh.get_grid(ilevel);
			// At multi-box levels each rank owns only its subset of boxes;
			// non-owners see NULL slots.
			if( !pu || !pf ) continue;
			int nx = pu->size(0), ny = pu->size(1), nz = pu->size(2);
			double berr = 0.0, bres = 0.0; size_t bcnt = 0;
			#pragma omp parallel for reduction(+:berr,bres,bcnt)
			for( int ix=0; ix<nx; ++ix )
			  for( int iy=0; iy<ny; ++iy )
			    for( int iz=0; iz<nz; ++iz )
				{
				  double res =  (double)m_scheme.apply( *pu, ix, iy, iz ) + h2 * (double)((*pf)(ix,iy,iz));
				  double val = (*pu)( ix, iy, iz );
				  if( fabs(val) > 0.0 ){
				      berr += fabs( res/val );
				      bres += fabs(res);
				      ++bcnt;
				    }
				}
			err += berr; mean_res += bres; count += bcnt;
		}

#ifdef USE_MPI
		// Allreduce only when SPMD MG is active (all ranks reached here).
		// Under classic wpd only rank 0 runs the solver → collective would deadlock.
		if( spmd_mg && MUSIC::mpi::size() > 1 ){
			double ge=0.0, gm=0.0; long long gc=0;
			long long lc = (long long)count;
			MPI_Allreduce(&err,     &ge, 1, MPI_DOUBLE, MPI_SUM, MUSIC::mpi::world());
			MPI_Allreduce(&mean_res,&gm, 1, MPI_DOUBLE, MPI_SUM, MUSIC::mpi::world());
			MPI_Allreduce(&lc,      &gc, 1, MPI_LONG_LONG, MPI_SUM, MUSIC::mpi::world());
			err = ge; mean_res = gm; count = (size_t)gc;
		}
#endif

		if( count != 0 )
		  {
		    err /= count;
		    mean_res /= count;
		  }
		if( verbose )
			std::cout << "      Level " << std::setw(6) << ilevel << ",   Error = " << err << std::endl;

		LOGDEBUG("[mg]      level %3d,  residual %g,  rel. error %g",ilevel, mean_res, err);

		maxerr = std::max(maxerr,err);

	}
	return maxerr;
}

template< class S, class I, class O, typename T >
double solver<S,I,O,T>::compute_RMS_resid( const GridHierarchy<T>& uh, const GridHierarchy<T>& fh, bool verbose )
{
	if( m_is_ini )
		m_residu_ini.assign( uh.levelmax()+1, 0.0 );
	
	double maxerr=0.0;
	
	for( unsigned ilevel=uh.levelmin(); ilevel <= uh.levelmax(); ++ilevel )
	{
		double h = 1.0/(1<<ilevel), h2=h*h;

		// D.3.3: at multi-box levels iterate per-box; at single-box levels use union.
		const size_t nb = uh.num_boxes(ilevel);
		const bool multi = (nb > 1);
		// B.2b.2.2.1.d: see compute_error for the spmd_mg rationale.
		const bool spmd_mg = MUSIC::poisson::spmd_mg_is_active();
		const int  single_box_owner = multi ? -1 : (int)uh.owner_of_box(ilevel, 0);
		const bool single_box_skip  = spmd_mg && (!multi) && (single_box_owner != MUSIC::mpi::rank());

		double sum = 0.0, sumd2 = 0.0;
		size_t count = 0;

		if( !single_box_skip ) for( size_t b=0; b < (multi ? nb : 1); ++b ){
			const MeshvarBnd<T> * pu = multi ? uh.get_grid(ilevel, b) : uh.get_grid(ilevel);
			const MeshvarBnd<T> * pf = multi ? fh.get_grid(ilevel, b) : fh.get_grid(ilevel);
			if( !pu || !pf ) continue;
			int nx = pu->size(0), ny = pu->size(1), nz = pu->size(2);
			double bsum=0.0, bsumd2=0.0; size_t bcnt=0;
			#pragma omp parallel for reduction(+:bsum,bsumd2,bcnt)
			for( int ix=0; ix<nx; ++ix )
				for( int iy=0; iy<ny; ++iy )
					for( int iz=0; iz<nz; ++iz )
					{
						double d = (double)(*pf)(ix,iy,iz);
						bsumd2 += d*d;
						double r = ((double)m_scheme.apply( *pu, ix, iy, iz )/h2 + d);
						bsum += r*r;
						++bcnt;
					}
			sum += bsum; sumd2 += bsumd2; count += bcnt;
		}

#ifdef USE_MPI
		if( spmd_mg && MUSIC::mpi::size() > 1 ){
			double gs=0.0, gd2=0.0; long long gc=0;
			long long lc = (long long)count;
			MPI_Allreduce(&sum,   &gs,  1, MPI_DOUBLE,    MPI_SUM, MUSIC::mpi::world());
			MPI_Allreduce(&sumd2, &gd2, 1, MPI_DOUBLE,    MPI_SUM, MUSIC::mpi::world());
			MPI_Allreduce(&lc,    &gc,  1, MPI_LONG_LONG, MPI_SUM, MUSIC::mpi::world());
			sum = gs; sumd2 = gd2; count = (size_t)gc;
		}
#endif

		if( count == 0 ) continue;

		if( m_is_ini )
			m_residu_ini[ilevel] =  sqrt(sum)/count;

		double err_abs = sqrt(sum/count);
		double err_rel = err_abs / sqrt(sumd2/count);
		
		if( verbose && !m_is_ini )
			std::cout << "      Level " << std::setw(6) << ilevel << ",   Error = " << err_rel << std::endl;		
		
		LOGDEBUG("[mg]      level %3d,  rms residual %g,  rel. error %g",ilevel, err_abs, err_rel);
		
		if( err_rel > maxerr )
			maxerr = err_rel;
		
	}
	
	if( m_is_ini )
		m_is_ini = false;
	
	return maxerr;
}


template< class S, class I, class O, typename T >
double solver<S,I,O,T>::solve( GridHierarchy<T>& uh, double acc, double h, bool verbose )
{

	double err, maxerr = 1e30;
	unsigned niter = 0;

	bool fullverbose = false;

	m_pu = &uh;

	// D.5: tell the user once what the multibox MPI parallelism profile is.
	// The per-cluster V-cycle (twoGrid_multibox) is communication-free and
	// runs on rank-0 only; workers idle during the multibox Poisson solve
	// (cluster-to-rank parallelization deferred). Single-box (zoom) levels
	// still distribute the finest-level GS smoother via mg_dist.
	{
		static bool s_logged_multibox_mpi = false;
		if( !s_logged_multibox_mpi ){
			s_logged_multibox_mpi = true;
			bool any_multi = false;
			for( unsigned L=uh.levelmin(); L<=uh.levelmax(); ++L )
				if( uh.num_boxes(L) > 1 ){ any_multi = true; break; }
			if( any_multi && MUSIC::mpi::size() > 1 ){
				LOGINFO("Multibox + MPI: per-cluster V-cycle runs on rank-0 only "
				        "(%d worker rank(s) idle during multibox Poisson solve). "
				        "Per-box meshes too small for slab distribute to amortize; "
				        "cluster-to-rank parallelization deferred to D.6.",
				        MUSIC::mpi::size()-1);
			}
		}
	}

	//err = compute_RMS_resid( *m_pu, *m_pf, fullverbose );

	MUSIC::mg_profile::reset();
	double _tsolve = MUSIC::mg_profile::tic();

	//... iterate ...//
	while (true)
	{

		LOGUSER("Performing multi-grid V-cycle...");
		twoGrid( uh.levelmax() );

		//err = compute_RMS_resid( *m_pu, *m_pf, fullverbose );
		err = compute_error( *m_pu, *m_pf, fullverbose );
		++niter;

		if( fullverbose ){
			LOGUSER("  multigrid iteration %3d, maximum RMS residual = %g", niter, err );
			std::cout << "   - Step No. " << std::setw(3) << niter << ", Max Err = " << err << std::endl;
			std::cout << "     ---------------------------------------------------\n";
		}

		if( err < maxerr )
			maxerr = err;

		if( (niter > 1) && ((err < acc) || (niter > 20)) )
			break;
	}

	{
		double _tw = MUSIC::mg_profile::tic() - _tsolve;
		char tag[64];
		snprintf(tag, sizeof(tag), "solve niter=%u wall=%.4fs", niter, _tw);
		MUSIC::mg_profile::report(tag, uh.levelmin(), uh.levelmax());
		MUSIC::mg_profile::reset();
	}
	
	if( err > acc )
	{
		std::cout << "Error : no convergence in Poisson solver" << std::endl;
		LOGERR("No convergence in Poisson solver, final error: %g.",err);
	}
	else if( verbose )
	{
		if( MUSIC::mpi::is_root() ) std::cout << " - Converged in " << niter << " steps to " << maxerr << std::endl;
		LOGUSER("Poisson solver converged to max. error of %g in %d steps.",err,niter);
	}

	//.. make sure that the RHS does not contain the FAS corrections any more
	// D.3.3: at multi-box levels, each per-box ff restricts into its own parent
	// fc (looked up via parent_box_index). At single-box levels, fall through
	// to union restrict.
	// B.2b.2.2.1.d: under SPMD MG every rank enters here; per-box meshes only
	// live on owners (owner_of_box=b%np). NULL-guard skips non-owners.
	for( int i=m_pf->levelmax(); i>0; --i ){
		size_t nb = m_pf->num_boxes(i);
		if( nb > 1 ){
			bool parent_multi = (m_pf->num_boxes(i-1) > 1);
			for( size_t b=0; b<nb; ++b ){
				MeshvarBnd<T> * ff = m_pf->get_grid(i, b);
				MeshvarBnd<T> * fc = parent_multi
					? m_pf->get_grid(i-1, m_pf->parent_box_index(i, b))
					: m_pf->get_grid(i-1);
				if( !ff || !fc ) continue;
				m_gridop.restrict( *ff, *fc );
			}
		} else {
			MeshvarBnd<T> * ff = m_pf->get_grid(i);
			MeshvarBnd<T> * fc = m_pf->get_grid(i-1);
			if( !ff || !fc ) continue;
			m_gridop.restrict( *ff, *fc );
		}
	}


	return err;
}



//TODO: this only works for 2nd order! (but actually not needed)
template< class S, class I, class O, typename T >
void solver<S,I,O,T>::setBC( unsigned ilevel )
{
	//... set only on level before additional refinement starts
	if( ilevel == m_ilevelmin )
	{
		MeshvarBnd<T> *u = m_pu->get_grid(ilevel);
		int
			nx = u->size(0), 
			ny = u->size(1), 
			nz = u->size(2);
			
		for( int iy=0; iy<ny; ++iy )
			for( int iz=0; iz<nz; ++iz )
			{
				(*u)(-1,iy,iz) = 2.0*(*m_pubnd)(-1,iy,iz) - (*u)(0,iy,iz);
				(*u)(nx,iy,iz) = 2.0*(*m_pubnd)(nx,iy,iz) - (*u)(nx-1,iy,iz);;
			}
		
		for( int ix=0; ix<nx; ++ix )
			for( int iz=0; iz<nz; ++iz )
			{
				(*u)(ix,-1,iz) = 2.0*(*m_pubnd)(ix,-1,iz) - (*u)(ix,0,iz);
				(*u)(ix,ny,iz) = 2.0*(*m_pubnd)(ix,ny,iz) - (*u)(ix,ny-1,iz);
			}
		
		for( int ix=0; ix<nx; ++ix )
			for( int iy=0; iy<ny; ++iy )
			{
				(*u)(ix,iy,-1) = 2.0*(*m_pubnd)(ix,iy,-1) - (*u)(ix,iy,0);
				(*u)(ix,iy,nz) = 2.0*(*m_pubnd)(ix,iy,nz) - (*u)(ix,iy,nz-1);
			}		
		
		
		
	}
}



//... enforce periodic boundary conditions
template< class S, class I, class O, typename T >
void solver<S,I,O,T>::make_periodic( MeshvarBnd<T> *u )
{
	

	int
		nx = u->size(0), 
		ny = u->size(1), 
		nz = u->size(2);
	int nb = u->m_nbnd;
	
		
	//if( u->offset(0) == 0 )
		for( int iy=-nb; iy<ny+nb; ++iy )
			for( int iz=-nb; iz<nz+nb; ++iz )
			{
				int iiy( (iy+ny)%ny ), iiz( (iz+nz)%nz );
				
				for( int i=-nb; i<0; ++i )
				{
					(*u)(i,iy,iz) = (*u)(nx+i,iiy,iiz);
					(*u)(nx-1-i,iy,iz) = (*u)(-1-i,iiy,iiz);	
				}
				
			}
	
	//if( u->offset(1) == 0 )
		for( int ix=-nb; ix<nx+nb; ++ix )
			for( int iz=-nb; iz<nz+nb; ++iz )
			{
				int iix( (ix+nx)%nx ), iiz( (iz+nz)%nz );
				
				for( int i=-nb; i<0; ++i )
				{
					(*u)(ix,i,iz) = (*u)(iix,ny+i,iiz);
					(*u)(ix,ny-1-i,iz) = (*u)(iix,-1-i,iiz);
				}
			}
	
	//if( u->offset(2) == 0 )
		for( int ix=-nb; ix<nx+nb; ++ix )
			for( int iy=-nb; iy<ny+nb; ++iy )
			{
				int iix( (ix+nx)%nx ), iiy( (iy+ny)%ny );
				
				for( int i=-nb; i<0; ++i )
				{
					(*u)(ix,iy,i) = (*u)(iix,iiy,nz+i);
					(*u)(ix,iy,nz-1-i) = (*u)(iix,iiy,-1-i);
				}
			}
	
}


END_MULTIGRID_NAMESPACE
 
#endif
