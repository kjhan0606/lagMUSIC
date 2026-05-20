/*
 
 convolution_kernel.hh - This file is part of MUSIC -
 a code to generate multi-scale initial conditions 
 for cosmological simulations 
 
 Copyright (C) 2010-19  Oliver Hahn
 
*/

#ifndef __CONVOLUTION_KERNELS_HH
#define __CONVOLUTION_KERNELS_HH

#include <string>
#include <map>

#include "config_file.hh"
#include "densities.hh"
#include "transfer_function.hh"

#define ACC_RF(i, j, k) (((((size_t)(i) + nx) % nx) * ny + (((size_t)(j) + ny) % ny)) * 2 * (nz / 2 + 1) + (((size_t)(k) + nz) % nz))
#define ACC_RC(i, j, k) (((((size_t)(i) + nxc) % nxc) * nyc + (((size_t)(j) + nyc) % nyc)) * 2 * (nzc / 2 + 1) + (((size_t)(k) + nzc) % nzc))

namespace convolution
{

//! encapsulates all parameters required for transfer function convolution
struct parameters
{
	int nx, ny, nz;
	double lx, ly, lz; //,boxlength;
	config_file *pcf;
	transfer_function *ptf;
	unsigned coarse_fact;
	bool deconvolve;
	bool is_finest;
	bool smooth;
};

/////////////////////////////////////////////////////////////////

//! abstract base class for a transfer function convolution kernel
class kernel
{
public:
	//! all parameters (physical/numerical)
	parameters cparam_;

	config_file *pcf_;
	transfer_function *ptf_;
	refinement_hierarchy *prefh_;
	tf_type type_;

	//! constructor
	kernel(config_file &cf, transfer_function *ptf, refinement_hierarchy &refh, tf_type type)
		: pcf_(&cf), ptf_(ptf), prefh_(&refh), type_(type) //cparam_( cp )
	{
	}

	//! dummy constructor
	/*kernel( void )
      {	}*/

	//! compute/load the kernel. `distributed=true` requests an MPI-slab kernel
	//! buffer (only meaningful for kernels backed by a real-space cache; ksampled
	//! kernels ignore the flag because they synthesize values on the fly).
	virtual kernel *fetch_kernel(int ilevel, bool isolated = false, bool distributed = false) = 0;

	//! virtual destructor
	virtual ~kernel(){};

	//! purely virtual method to obtain a pointer to the underlying data
	virtual void *get_ptr() = 0;

	//! purely virtual method to determine whether the kernel is k-sampled or not
	virtual bool is_ksampled() = 0;

	//! purely virtual vectorized method to compute the kernel value if is_ksampled
	virtual void at_k(size_t len, const double *in_k, double *out_Tk) = 0;

	//! free memory
	virtual void deallocate() = 0;
};

//! abstract factory class to create convolution kernels
struct kernel_creator
{
	//! creates a convolution kernel object
	virtual kernel *create(config_file &cf, transfer_function *ptf, refinement_hierarchy &refh, tf_type type) const = 0;

	//! destructor
	virtual ~kernel_creator() {}
};

//! access map to the various kernel classes through the factory
std::map<std::string, kernel_creator *> &get_kernel_map();

//! actual implementation of the factory class for kernel objects
template <class Derived>
struct kernel_creator_concrete : public kernel_creator
{
	//! constructor inserts the kernel class in the map
	kernel_creator_concrete(const std::string &kernel_name)
	{
		get_kernel_map()[kernel_name] = this;
	}

	//! creates an instance of the kernel object
	kernel *create(config_file &cf, transfer_function *ptf, refinement_hierarchy &refh, tf_type type) const
	{
		return new Derived(cf, ptf, refh, type);
	}
};

//! actual implementation of the FFT convolution (independent of the actual kernel)
template <typename real_t>
void perform(kernel *pk, void *pd, bool shift, bool fix, bool flip);

//! MPI / slab-distributed convolution. Forward FFT, k-space multiply with the
//! ksampled kernel, inverse FFT — all collective across ranks via FFTW3-MPI.
//! `pdist` must have been allocated via MUSIC::dist::make_slab_meshvar with
//! fftw_inplace_pad=true and matching global (nx,ny,nz).
//! Only ksampled kernels (kernel::is_ksampled()==true) are supported right now;
//! the cached real-space kernel path will be added once the kernel cache is
//! also slab-distributed.
template <typename real_t>
void perform_mpi(kernel *pk, class Meshvar<real_t> *pdist, bool shift, bool fix, bool flip);

//! Scatter/perform_mpi/gather wrapper around a root-owned full padded grid.
//! On rank 0 `root_data` must point to a contiguous nx*ny*2*(nz/2+1) real buffer
//! (the in-place r2c FFTW layout used by DensityGrid). On other ranks it is
//! ignored and may be NULL. All ranks must call this collectively. The buffer
//! on rank 0 is overwritten with the convolution result.
//! Falls back to plain perform() when USE_MPI is undefined or MPI size == 1.
template <typename real_t>
void perform_dist(kernel *pk, real_t *root_data, size_t gnx, size_t gny, size_t gnz,
                  bool shift, bool fix, bool flip);

} //namespace convolution

#endif //__CONVOLUTION_KERNELS_HH
