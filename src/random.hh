/*
 
 random.hh - This file is part of MUSIC -
 a code to generate multi-scale initial conditions 
 for cosmological simulations 
 
 Copyright (C) 2010  Oliver Hahn
 
*/

//... for testing purposes.............
//#define DEGRADE_RAND1
//#define DEGRADE_RAND2
//.....................................

#ifndef __RANDOM_HH
#define __RANDOM_HH

#define DEF_RAN_CUBE_SIZE	32

#include <fstream>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>
#include <sys/types.h>
#include <omp.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "general.hh"
#include "mesh.hh"
#include "mg_operators.hh"
#include "constraints.hh"


class RNG_plugin{
protected:
  config_file *pcf_;		//!< pointer to config_file from which to read parameters
public:
  explicit RNG_plugin( config_file& cf )
  : pcf_( &cf )
  { }
  virtual ~RNG_plugin() { }
  virtual bool is_multiscale() const  = 0; 
};


struct RNG_plugin_creator
{
  virtual RNG_plugin * create( config_file& cf ) const = 0;
  virtual ~RNG_plugin_creator() { }
};

std::map< std::string, RNG_plugin_creator *>&
get_RNG_plugin_map();

void print_RNG_plugins( void );

template< class Derived >
struct RNG_plugin_creator_concrete : public RNG_plugin_creator
{
  //! register the plugin by its name
  RNG_plugin_creator_concrete( const std::string& plugin_name )
  {
    get_RNG_plugin_map()[ plugin_name ] = this;
  }

  //! create an instance of the plugin
  RNG_plugin* create( config_file& cf ) const
  {
    return new Derived( cf );
  }
};

typedef RNG_plugin RNG_instance;
RNG_plugin *select_RNG_plugin( config_file& cf );


/*!
 * @brief encapsulates all things random number generator related
 */
template< typename T >
class random_numbers
{
public:
	unsigned 
		res_,		//!< resolution of the full mesh
		cubesize_,	//!< size of one independent random number cube
		ncubes_;	//!< number of random number cubes to cover the full mesh
	long baseseed_;	//!< base seed from which cube seeds are computed 
	
    
protected:
	//! vector of 3D meshes (the random number cubes) with random numbers
	std::vector< Meshvar<T>* > rnums_;
    
    //! map of 3D indices to cube index
    std::map<size_t,size_t> cubemap_;
    
    typedef std::map<size_t,size_t>::iterator cubemap_iterator;
	
protected:
    
    //! register a cube with the hash map
    void register_cube( int i, int j, int k);
	
	//! fills a subcube with random numbers
	double fill_cube( int i, int j, int k);
	
	//! subtract a constant from an entire cube
	void subtract_from_cube( int i, int j, int k, double val );
	
	//! copy random numbers from a cube to a full grid array
	template< class C >
	void copy_cube( int i, int j, int k, C& dat )
	{
		int offi, offj, offk;
		
		offi = i*cubesize_;
		offj = j*cubesize_;
		offk = k*cubesize_;
		
		i = (i+ncubes_)%ncubes_;
		j = (j+ncubes_)%ncubes_;
		k = (k+ncubes_)%ncubes_;
		
		size_t icube = (i*ncubes_+j)*ncubes_+k;
        cubemap_iterator it = cubemap_.find( icube );
        
        if( it == cubemap_.end() )
        {
            LOGERR("attempting to copy data from non-existing RND cube %d,%d,%d",i,j,k);
            throw std::runtime_error("attempting to copy data from non-existing RND cube");
        }
        
        size_t cubeidx = it->second;
		
		for( int ii=0; ii<(int)cubesize_; ++ii )
			for( int jj=0; jj<(int)cubesize_; ++jj )
				for( int kk=0; kk<(int)cubesize_; ++kk )
					dat(offi+ii,offj+jj,offk+kk) = (*rnums_[cubeidx])(ii,jj,kk);
	}
	
	//! free the memory associated with a subcube
	void free_cube( int i, int j, int k );
	
	//! initialize member variables and allocate memory
	void initialize( void );
	
	//! fill a cubic subvolume of the full grid with random numbers
	double fill_subvolume( int *i0, int *n );
	
	//! fill an entire grid with random numbers
	double fill_all( void );
	
	//! fill an external array instead of the internal field
	template< class C >
	double fill_all( C& dat )
	{
		double sum = 0.0;
		
        for( int i=0; i<(int)ncubes_; ++i )
			for( int j=0; j<(int)ncubes_; ++j )
				for( int k=0; k<(int)ncubes_; ++k )
				{
					int ii(i),jj(j),kk(k);
                    register_cube(ii,jj,kk);
                }
        
		#pragma omp parallel for reduction(+:sum)
		for( int i=0; i<(int)ncubes_; ++i )
			for( int j=0; j<(int)ncubes_; ++j )
				for( int k=0; k<(int)ncubes_; ++k )
				{
					int ii(i),jj(j),kk(k);
					
					ii = (ii+ncubes_)%ncubes_;
					jj = (jj+ncubes_)%ncubes_;
					kk = (kk+ncubes_)%ncubes_;
					
					sum+=fill_cube(ii, jj, kk);
					copy_cube(ii,jj,kk,dat);
					free_cube(ii, jj, kk);
				}
		
		return sum/(ncubes_*ncubes_*ncubes_);
	}
	
	//! write the number of allocated random number cubes to stdout
	void print_allocated( void );
	
public:
	
	//! constructor
	random_numbers( unsigned res, unsigned cubesize, long baseseed, int *x0, int *lx );	
	
	//! constructor for constrained fine field
	random_numbers( random_numbers<T>& rc, unsigned cubesize, long baseseed, 
			bool kspace=false, bool isolated=false, int *x0_=NULL, int *lx_=NULL, bool zeromean=true );
	
	//! constructor
	random_numbers( unsigned res, unsigned cubesize, long baseseed, bool zeromean=true );
	
	
	//! constructor to read white noise from file
	random_numbers( unsigned res, std::string randfname, bool rndsign );

	//! SPMD slab-aware constructor: only fills cubes whose x-cube-index is in
	//! [cube_x0, cube_x0+cube_nx) (no periodic wrap supported — caller must keep
	//! the range within [0, ncubes)). Does NOT subtract any zero-mean. The
	//! caller is responsible for cross-rank Allreduce of the cell sum and then
	//! calling subtract_from_x_slab_cells(cube_x0, cube_nx, mean).
	//! Distinguished from the (res, cubesize, baseseed, bool zeromean) overload
	//! by the unsigned types of the slab arguments.
	random_numbers( unsigned res, unsigned cubesize, long baseseed,
	                unsigned cube_x0, unsigned cube_nx );

	//! copy constructor for averaged field (not copying) hence explicit!
	explicit random_numbers( /*const*/ random_numbers <T>& rc, bool kdegrade = true );
	
	//! destructor
	~random_numbers()
	{
		for( unsigned i=0; i<rnums_.size(); ++i )
			if( rnums_[i] != NULL )
				delete rnums_[i];
		rnums_.clear();
	}
	
	//! access a random number, this allocates a cube and fills it with consistent random numbers
	inline T& operator()( int i, int j, int k, bool fillrand=true )
	{
		int ic, jc, kc, is, js, ks;
		
		if( ncubes_ == 0 )
			throw std::runtime_error("random_numbers: internal error, not properly initialized");
		
		//... determine cube
		ic = (int)((double)i/cubesize_ + ncubes_) % ncubes_;
		jc = (int)((double)j/cubesize_ + ncubes_) % ncubes_;
		kc = (int)((double)k/cubesize_ + ncubes_) % ncubes_;
		
		size_t icube = ((size_t)ic*ncubes_+(size_t)jc)*ncubes_+(size_t)kc;
		
        cubemap_iterator it = cubemap_.find( icube );
        
        if( it == cubemap_.end() )
        {
            LOGERR("Attempting to copy data from non-existing RND cube %d,%d,%d @ %d,%d,%d",ic,jc,kc,i,j,k);
            throw std::runtime_error("attempting to copy data from non-existing RND cube");
            
        }
        
        size_t cubeidx = it->second;
        
		if( rnums_[ cubeidx ] == NULL )
		{
            LOGERR("Attempting to access data from non-allocated RND cube %d,%d,%d",ic,jc,kc);
            throw std::runtime_error("attempting to access data from non-allocated RND cube");
		}
		
		//... determine cell in cube
		is = (i - ic * cubesize_ + cubesize_) % cubesize_;
		js = (j - jc * cubesize_ + cubesize_) % cubesize_;
		ks = (k - kc * cubesize_ + cubesize_) % cubesize_;
		
        return (*rnums_[ cubeidx ])(is,js,ks);
	}
	
	//! SPMD: register and fill x-slab cubes [cube_x0, cube_x0+cube_nx) × all y × all z.
	//! No periodic wrap (caller keeps range in [0, ncubes_)). Returns the local
	//! sum of cell values in the owned cubes (NOT the mean).
	double fill_x_slab_cubes( unsigned cube_x0, unsigned cube_nx );

	//! SPMD: subtract `val` from every cell of owned x-slab cubes.
	void subtract_from_x_slab_cells( unsigned cube_x0, unsigned cube_nx, double val );

	//! F.2a: SPMD slab-aware subvolume fill. Registers and fills cubes covering
	//! the subvolume [i0, i0+n) but restricted in x to absolute cube indices
	//! [cube_x0_abs, cube_x0_abs + cube_nx_local). Mirrors fill_subvolume's
	//! padding semantics (ncube[d] = n[d]/cubesize + 2). Returns the local cell
	//! count's contribution to the mean; caller may Allreduce a true mean.
	double fill_subvolume_x_slab( int *i0, int *n,
	                              unsigned cube_x0_abs, unsigned cube_nx_local );

	//! free all cubes
	void free_all_mem( void )
	{
		for( unsigned i=0; i<rnums_.size(); ++i )
			if( rnums_[i] != NULL )
			{
				delete rnums_[i];	
				rnums_[i] = NULL;
			}
	}
	
	
};


/*!
 * @brief encapsulates all things for multi-scale white noise generation
 */
template< typename rng, typename T >
class random_number_generator
{
protected:
	config_file						* pcf_;
	refinement_hierarchy			* prefh_;
	constraint_set					constraints;
	
	int								levelmin_, 
									levelmax_, 
									levelmin_seed_;
	std::vector<long>				rngseeds_;
	std::vector<std::string>		rngfnames_;
	
	bool							disk_cached_;
	bool							restart_;
	std::vector< std::vector<T>* >	mem_cache_;
	
	unsigned						ran_cube_size_;
	

protected:
	
	//! checks if the specified string is numeric
	bool is_number(const std::string& s);
	
	//! parses the random number parameters in the conf file
	void parse_rand_parameters( void );
	
	//! correct coarse grid averages for the change in small scale when using Fourier interpolation
	void correct_avg( int icoarse, int ifine );

	//! F.3: SPMD slab-distributed correct_avg for the disk_cached_ path.
	//! Each rank pread's its coarse-x slab, oct-averages the corresponding
	//! fine-x slab in place, and pwrite's its coarse slab back. Header is
	//! preserved (existing file is not truncated). Returns true if executed
	//! the slab path; false if caller must fall back to rank-0 correct_avg.
	bool correct_avg_slab( int icoarse, int ifine );

	//! F.3: all-ranks entry point that runs the correct_avg pass after
	//! compute_random_numbers / compute_random_numbers_slab_unigrid. Honours
	//! the legacy `setup.kspace_TF` gate (only fires when kspace_TF=no).
	//! Workers participate in SPMD slab calls; rank 0 handles rank-0 fallback.
	void apply_correct_avg_pass( void );
	
	//! the main driver routine for multi-scale white noise generation
	//! When `stop_at_levelmin=true`, only levelmin is generated + stored,
	//! and the refinement loop is skipped (F.2b SPMD pass handles refinement).
	void compute_random_numbers( bool stop_at_levelmin = false );

	//! SPMD slab-distributed driver for the unigrid (levelmin==levelmax) case
	//! with shift-aligned cube boundaries. Each rank fills only its own
	//! x-cube slab and writes only its own output x-rows to wnoise_NNNN.bin.
	void compute_random_numbers_slab_unigrid( void );

	//! F.2b: all-ranks SPMD refinement-level wnoise pass. For each ilevel in
	//! [levelmin+1, levelmax], reads parent wnoise_<ilevel-1>.bin from disk,
	//! builds the child rng x-slab via F.2a (build_child_rng_subvolume_slab_kspace),
	//! and pwrites the slab to wnoise_<ilevel>.bin (rank 0 writes header +
	//! ftruncate; barrier; all ranks pwrite their planes).
	//!
	//! Eligibility (returns true if executed full pass, false if any level
	//! ineligible — caller must fall back to rank-0 refinement loop):
	//!   - all of: mpi_size > 1, disk_cached, shift=0
	//!   - per ilevel: lx[i] even > 0; lx[0] % nproc == 0;
	//!     (lx[0]/nproc) % cubesize == 0; x0[0] % cubesize == 0;
	//!     coarse subvolume fits in parent dims (no periodic wrap)
	bool apply_refinement_pass_slab( void );

	//! F.2b precheck (no I/O, no MPI): returns true iff apply_refinement_pass_slab
	//! would succeed end-to-end. Used by the dispatcher to decide whether
	//! rank-0 compute_random_numbers should `stop_at_levelmin` (handing
	//! refinement off to the SPMD pass) or run the full legacy refinement loop.
	bool refinement_pass_slab_eligible( void ) const;

	//! F.5-A precheck (no I/O, no MPI): returns true iff
	//! apply_refinement_pass_per_cluster would succeed end-to-end. Opt-in via
	//! setup.rng_per_cluster=yes (default no — preserves baseline bit-identical).
	//! Eligibility: mpi_size > 1, disk_cached, shift=0, at least one refinement
	//! level has num_boxes >= 2, and per cluster the coarse subvolume fits in
	//! parent (no periodic wrap). Single-cluster levels are still handled (with
	//! a degenerate b=0 cluster) because correctness requires every cluster of
	//! every refinement level to be regenerated with the new gap-cell semantics.
	bool refinement_pass_per_cluster_eligible( void ) const;

	//! F.5-A SPMD pass: per-cluster RNG via round-robin owner dispatch.
	//! For each refinement level L in [levelmin+1, levelmax]:
	//!   1) rank 0 writes a raw cube-fill union wnoise to wnoise_<L>.bin
	//!      (deterministic Gaussian samples from rngseeds_[L]; this populates
	//!      the gap cells in the union bbox so the F.3 correct_avg pass at
	//!      level L produces valid Gaussians at the parent's coarse cells)
	//!   2) barrier
	//!   3) per cluster b: owner = b % nproc constructs the cluster's child
	//!      rng via a serial kspace builder (parent read from disk) and
	//!      pwrites the cluster cells into wnoise_<L>.bin at union byte
	//!      offsets, overwriting the raw cube-fill in cluster regions
	//!   4) barrier
	//! Cluster cells carry the full kspace correlation with parent; gap cells
	//! contain independent N(0,1) samples — values DIFFER vs the rank-0
	//! serial path's union kspace ctor (acknowledged trade-off).
	bool apply_refinement_pass_per_cluster( void );

	//! SPMD slab-aware store: each rank pwrites its own x-slab planes;
	//! rank 0 writes the header. Caller passes the rank's output slab range
	//! and the cube range that backs it.
	void store_rnd_slab_unigrid( int ilevel, rng* prng,
	                             unsigned out_x_start, unsigned out_nx,
	                             unsigned cube_x0, unsigned cube_nx,
	                             int i0_x, int i0_y, int i0_z );

	//! store the white noise fields in memory or on disk
	void store_rnd( int ilevel, rng* prng );
	

public:
	
	//! constructor
	random_number_generator( config_file& cf, refinement_hierarchy& refh, transfer_function *ptf = NULL );	
	
	//! destructor
	~random_number_generator();
	
	//! load random numbers to a new array
	template< typename array >
	void load( array& A, int ilevel )
	{
		if( restart_ )
			LOGINFO("Attempting to restart using random numbers for level %d\n     from file \'wnoise_%04d.bin\'.",ilevel,ilevel);
		
		if( disk_cached_ )
		{
			char fname[128];
			sprintf(fname,"wnoise_%04d.bin",ilevel);
			
			LOGUSER("Loading white noise from file \'%s\'...",fname);
			
			std::ifstream ifs( fname, std::ios::binary );
			if( !ifs.good() )
			{	
				LOGERR("White noise file \'%s\'was not found.",fname);
				throw std::runtime_error("A white noise file was not found. This is an internal inconsistency and bad.");
				
			}
			
			int nx,ny,nz;
			ifs.read( reinterpret_cast<char*> (&nx), sizeof(int) );
			ifs.read( reinterpret_cast<char*> (&ny), sizeof(int) );
			ifs.read( reinterpret_cast<char*> (&nz), sizeof(int) );
			
			if( nx!=(int)A.size(0) || ny!=(int)A.size(1) || nz!=(int)A.size(2) )
			{	

			  if( nx==(int)A.size(0)*2 && ny==(int)A.size(1)*2 && nz==(int)A.size(2)*2 )
			    {
			      std::cerr << "CHECKPOINT" << std::endl;


			      int ox = nx/4, oy = ny/4, oz = nz/4;
			      std::vector<T> slice( ny*nz, 0.0 );

			      for( int i=0; i<nx; ++i )
				{
				  ifs.read( reinterpret_cast<char*> ( &slice[0] ), ny*nz*sizeof(T) );
			      
				  if( i<ox ) continue;
				  if( i>=3*ox ) break;

                                  #pragma omp parallel for
				  for( int j=oy; j<3*oy; ++j )
				    for( int k=oz; k<3*oz; ++k )
				      A(i-ox,j-oy,k-oz) = slice[j*nz+k];
				}		
			  
			      ifs.close();	
			    }
			  else
			    {
			      LOGERR("White noise file is not aligned with array. File: [%d,%d,%d]. Mem: [%d,%d,%d].",
				     nx,ny,nz,A.size(0),A.size(1),A.size(2));
			      throw std::runtime_error("White noise file is not aligned with array. This is an internal inconsistency and bad.");
			    }
			}else{
			
			  for( int i=0; i<nx; ++i )
			    {
			      std::vector<T> slice( ny*nz, 0.0 );
			      ifs.read( reinterpret_cast<char*> ( &slice[0] ), ny*nz*sizeof(T) );
			      
                              #pragma omp parallel for
			      for( int j=0; j<ny; ++j )
				for( int k=0; k<nz; ++k )
				  A(i,j,k) = slice[j*nz+k];
			      
			    }		
			  
			  ifs.close();	
			}
		}
		else
		{
			LOGUSER("Copying white noise from memory cache...");

			if( mem_cache_[ilevel-levelmin_] == NULL )
				LOGERR("Tried to access mem-cached random numbers for level %d. But these are not available!\n",ilevel);

			int nx( A.size(0) ), ny( A.size(1) ), nz( A.size(2) );

			if ( (size_t)nx*(size_t)ny*(size_t)nz != mem_cache_[ilevel-levelmin_]->size() )
			{
				LOGERR("White noise file is not aligned with array. File: [%d,%d,%d]. Mem: [%d,%d,%d].",nx,ny,nz,A.size(0),A.size(1),A.size(2));
				throw std::runtime_error("White noise file is not aligned with array. This is an internal inconsistency and bad");
			}

			#pragma omp parallel for
			for( int i=0; i<nx; ++i )
				for( int j=0; j<ny; ++j )
					for( int k=0; k<nz; ++k )
						A(i,j,k) = (*mem_cache_[ilevel-levelmin_])[((size_t)i*ny+(size_t)j)*nz+(size_t)k];

			std::vector<T>().swap( *mem_cache_[ilevel-levelmin_] );
			delete mem_cache_[ilevel-levelmin_];
			mem_cache_[ilevel-levelmin_] = NULL;

		}


	}

	//! Distributed-load: each rank fills only its own x-slab from the disk-cached
	//! wnoise file. Writes into a buffer laid out as (local_nx, gny, gnz_padded)
	//! with row-major indexing — same layout as DensityGrid / make_slab_meshvar
	//! (fftw_inplace_pad=true). Padding cells in the inner dim are not touched.
	//! Pre: random/disk_cached must be true (workers cannot share mem_cache_).
	//! Pre: rank 0's compute_random_numbers() must have completed (the constructor
	//! barriers on this when USE_MPI is defined).
	template< typename real_t >
	void load_slab( real_t* slab_data, std::size_t local_x_start, std::size_t local_nx,
	                std::size_t gny, std::size_t gnz, std::size_t gnz_padded, int ilevel )
	{
		if( local_nx == 0 ) return;

		if( !disk_cached_ ){
			LOGERR("random_number_generator::load_slab requires random/disk_cached=yes (workers cannot share mem cache).");
			throw std::runtime_error("load_slab requires disk_cached white noise");
		}

		char fname[128];
		std::sprintf(fname, "wnoise_%04d.bin", ilevel);
		std::FILE* fp = std::fopen(fname, "rb");
		if( !fp ){
			LOGERR("load_slab: cannot open '%s'", fname);
			throw std::runtime_error("load_slab: cannot open wnoise file");
		}

		unsigned hnx = 0, hny = 0, hnz = 0;
		if( std::fread(&hnx, sizeof(unsigned), 1, fp) != 1
		 || std::fread(&hny, sizeof(unsigned), 1, fp) != 1
		 || std::fread(&hnz, sizeof(unsigned), 1, fp) != 1 ){
			std::fclose(fp);
			LOGERR("load_slab: failed to read header from '%s'", fname);
			throw std::runtime_error("load_slab: header read failed");
		}
		if( hny != (unsigned)gny || hnz != (unsigned)gnz ){
			std::fclose(fp);
			LOGERR("load_slab: wnoise file dim mismatch (file=%u,%u,%u vs slab y,z=%zu,%zu)",
			       hnx, hny, hnz, gny, gnz);
			throw std::runtime_error("load_slab: wnoise dim mismatch");
		}
		if( local_x_start + local_nx > (std::size_t)hnx ){
			std::fclose(fp);
			LOGERR("load_slab: slab x-range [%zu,%zu) exceeds file nx=%u",
			       local_x_start, local_x_start + local_nx, hnx);
			throw std::runtime_error("load_slab: slab out of range");
		}

		const off_t header_bytes = (off_t)(3 * sizeof(unsigned));
		const off_t plane_bytes  = (off_t)((std::size_t)gny * (std::size_t)gnz * sizeof(real_t));
		if( fseeko(fp, header_bytes + (off_t)local_x_start * plane_bytes, SEEK_SET) != 0 ){
			std::fclose(fp);
			LOGERR("load_slab: fseeko failed on '%s'", fname);
			throw std::runtime_error("load_slab: fseeko failed");
		}

		std::vector<real_t> plane((std::size_t)gny * (std::size_t)gnz);
		for( std::size_t i = 0; i < local_nx; ++i ){
			const std::size_t want = (std::size_t)gny * (std::size_t)gnz;
			const std::size_t got  = std::fread(&plane[0], sizeof(real_t), want, fp);
			if( got != want ){
				std::fclose(fp);
				LOGERR("load_slab: short read on plane %zu (got %zu of %zu)", i, got, want);
				throw std::runtime_error("load_slab: short read");
			}
			if( gnz_padded == gnz ){
				std::memcpy(&slab_data[i * gny * gnz_padded], &plane[0],
				            want * sizeof(real_t));
			} else {
				#pragma omp parallel for
				for( std::size_t j = 0; j < gny; ++j )
					for( std::size_t k = 0; k < gnz; ++k )
						slab_data[(i * gny + j) * gnz_padded + k] = plane[j * gnz + k];
			}
		}

		std::fclose(fp);
	}
};

typedef random_numbers<real_t> rand_nums;
typedef random_number_generator< rand_nums,real_t> rand_gen;


namespace MUSIC { namespace rng_slab {

//! F.2a: SPMD child-rng builder (kspace coarse-mode replacement path).
//!
//! Each MPI rank builds an x-slab of the refinement-level child rng for
//! subvolume [x0, x0+lx) (in child fine-grid coords). Parent rng cells are
//! read from `parent_file` (plane-major + 3-unsigned header, like wnoise_*.bin).
//! parent_file must hold a full parent_res^3 array (no subvolume header).
//!
//! The slab partition is FFTW3-MPI's choice (via make_slab_meshvar). Caller
//! receives a Meshvar<real_t>* with FFTW-padded inner dim (nz+2). m_pdata
//! holds the per-rank x-slab. Caller deletes the Meshvar*.
//!
//! Eligibility (returns NULL if not satisfied):
//!   - lx[0] % nproc == 0
//!   - (lx[0] / nproc) % cubesize == 0
//!   - x0[0] % cubesize == 0
//!   - parent coarse subvolume [parent_x0_abs..+lx[0]/2) etc. fits within
//!     [0, parent_res) (no periodic wrap supported in F.2a)
//!
//! Limitations of F.2a: kspace coarse-mode replacement path only. The
//! Hoffman-Ribak / isolated paths are not handled — caller falls back.
//!
//! Non-templated: FFTW3-MPI plans are typed on fftw_real (== real_t at
//! runtime in all real builds). Caller is responsible for ensuring its data
//! type matches fftw_real; the smoke test enforces this.
//!
//! parent_{nx,ny,nz} are the parent file's stored dims (allow rectangular
//! for refinement-level parents stored by store_rnd). parent_{x0,y0,z0}_abs
//! is the parent file's plane index of the coarse subvolume base — caller
//! computes via (absolute_coord_of_first_coarse_cell - parent_file_i0).
//! prng_res_parent is the parent's RNG resolution res_ (2^(L-1) in production;
//! must equal parent file dim for unigrid, but may be smaller than the file
//! dim when the refinement region tiles the parent's prng space — see B.6).
//! Coord wraparound is handled internally via read_coarse_subvolume_periodic.
Meshvar<fftw_real>* build_child_rng_subvolume_slab_kspace(
    long child_seed, unsigned cubesize,
    int x0[3], int lx[3],
    const char *parent_file,
    unsigned parent_nx, unsigned parent_ny, unsigned parent_nz,
    unsigned prng_res_parent,
    int parent_x0_abs, int parent_y0_abs, int parent_z0_abs,
    ptrdiff_t &out_local_nx, ptrdiff_t &out_local_x_start);

//! F.2a smoke test driver. On rank 0: generates parent rng (res=parent_res,
//! seed=parent_seed) and child rng (existing serial constructor) as reference.
//! Writes parent to a temp file. All ranks call build_child_rng_subvolume_slab_kspace
//! and gather the result to rank 0 for cell-by-cell comparison. Throws on
//! verification failure. No-op if MPI size == 1.
//!
//! Returns max |slab - serial| (rank 0 only; other ranks return 0).
double run_child_rng_slab_smoke_test(
    unsigned parent_res, long parent_seed,
    unsigned cubesize, long child_seed,
    int x0[3], int lx[3]);

//! F.5-A smoke test driver. Validates the per-cluster pwrite layout used
//! by the production multibox RNG SPMD path.
//!
//! Test scenario: N disjoint clusters inside a synthetic union bbox.
//! - All ranks: build parent rng deterministically (parent_res, parent_seed).
//! - Reference: rank 0 builds each cluster's child rng (existing serial
//!   kspace constructor) and assembles values into an in-memory union array.
//! - SPMD: clusters round-robin to ranks (b % nproc). Each owner constructs
//!   its cluster's child rng then pwrites cluster cells into a union-layout
//!   file at byte-disjoint offsets.
//! - Rank 0 reads back the SPMD file and compares cell-by-cell to the
//!   reference union (in cluster regions) and verifies gap cells are zero.
//!
//! u_x0[3], u_lx[3]            : union bbox origin and extent (fine cells)
//! clu_x0[n_clusters][3]       : per-cluster absolute origin (fine cells)
//! clu_lx[n_clusters][3]       : per-cluster extent (fine cells)
//!
//! All cluster bboxes must be contained in the union bbox AND disjoint.
//! Returns max |spmd - ref| inside cluster regions (rank 0 only; other
//! ranks return broadcast of rank-0 result). Throws on FAIL.
double run_per_cluster_rng_smoke_test(
    unsigned parent_res, long parent_seed,
    unsigned cubesize, long child_seed,
    const int u_x0[3], const int u_lx[3],
    int n_clusters,
    const int (*clu_x0)[3], const int (*clu_lx)[3]);

//! F.5-A: serial per-cluster kspace child rng builder. Used by F.5-A
//! production wiring (apply_refinement_pass_per_cluster); called only by
//! the cluster's owner rank (round-robin b % nproc dispatch). Mirrors
//! F.2a's parent_file pread + r2c + k-blend + c2r pipeline, but with serial
//! FFTW3 plans (no MPI) and produces the FULL child cube (not an x-slab).
//!
//! On success fills `out_cube_packed` (size lx[0]*lx[1]*lx[2], row-major
//! packed, no FFT padding) with the child rng cells and returns true.
//! On failure returns false (out buffer left undefined).
//!
//! Constraints (returns false if violated):
//!   - lx[d] > 0, even, for d in {0,1,2}
//!   - parent_nx == parent_ny == parent_nz (cubic parent)
//!   - parent_{x0,y0,z0}_abs >= 0 and coarse subvolume [px..+lx/2) fits
//!     within [0, parent_n*) (no periodic wrap supported)
//!
//! Non-templated: uses fftw_real (== real_t at runtime in all real builds).
bool build_child_rng_per_cluster_serial_kspace(
    long child_seed, unsigned cubesize,
    int x0[3], int lx[3],
    const char *parent_file,
    unsigned parent_nx, unsigned parent_ny, unsigned parent_nz,
    unsigned prng_res_parent,
    int parent_x0_abs, int parent_y0_abs, int parent_z0_abs,
    fftw_real *out_cube_packed);

#ifdef USE_MPI
//! G.3b: per-cluster sub_comm child-rng builder. Same structure as F.2a
//! (build_child_rng_subvolume_slab_kspace) but FFTW3-MPI plans run on a
//! caller-provided sub_comm rather than world. Used by G.3c production
//! wiring: ALL ranks in sub_comm participate in one cluster's FFT in
//! parallel; each rank pwrites its own x-slab into the wnoise file.
//!
//! Eligibility (returns NULL if not satisfied) — same as F.2a but on sub_comm:
//!   - sub_size := MPI_Comm_size(sub_comm)
//!   - lx[0] % sub_size == 0
//!   - (lx[0] / sub_size) % cubesize == 0
//!   - x0[0] % cubesize == 0
//!   - parent coarse subvolume must fit within parent_n* (no wrap)
//!   - cubic parent
//!
//! Returns Meshvar* with FFTW-padded slab on success; caller deletes.
//! out_local_nx / out_local_x_start give the sub_comm-local x-partition
//! (so caller can compute byte offsets for pwrite into a union file).
Meshvar<fftw_real>* build_child_rng_per_cluster_subcomm_kspace(
    MPI_Comm sub_comm,
    long child_seed, unsigned cubesize,
    int x0[3], int lx[3],
    const char *parent_file,
    unsigned parent_nx, unsigned parent_ny, unsigned parent_nz,
    unsigned prng_res_parent,
    int parent_x0_abs, int parent_y0_abs, int parent_z0_abs,
    ptrdiff_t &out_local_nx, ptrdiff_t &out_local_x_start);

//! G.3b smoke test driver. Builds parent rng on rank 0, runs serial builder
//! as reference (rank 0), then splits world comm into K sub_comms (where
//! K = world_size / target_sub_size) and has each sub_comm run the subcomm
//! builder. Each sub_comm gathers its result to its sub-root for comparison
//! with the serial reference. Throws on mismatch.
//!
//! Returns max |subcomm - serial| (rank 0).
double run_per_cluster_rng_subcomm_smoke_test(
    int target_sub_size,
    unsigned parent_res, long parent_seed,
    unsigned cubesize, long child_seed,
    int x0[3], int lx[3]);
#endif

}} // namespace MUSIC::rng_slab

#endif //__RANDOM_HH

