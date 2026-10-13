// Direct RNG tests, independent of convolution/Poisson/2LPT tolerances.
#include "random.hh"
#include "mpi_helper.hh"
#include "convolution_kernel.hh"
#include <iostream>

// Definitions normally supplied by MUSIC's main.cc (excluded for this test).
transfer_function *TransferFunction_real::ptf_ = NULL;
transfer_function *TransferFunction_k::ptf_ = NULL;
tf_type TransferFunction_k::type_;
tf_type TransferFunction_real::type_;
real_t TransferFunction_real::nspec_ = -1.0;
real_t TransferFunction_k::nspec_ = -1.0;

static void check(int lx, int ly, int lz, int ox, int oy, int oz)
{
    const int period = 32;
    int origin[3] = {ox, oy, oz}, size[3] = {lx, ly, lz};
    std::vector<double> src((size_t)lx*ly*lz), a(src.size()), b(src.size());
    for (size_t q=0; q<src.size(); ++q) src[q] = (double)q;

    omp_set_num_threads(1);
    random_numbers<double> serial(period, 8, 12345, origin, size);
    serial.unpack_subvolume_from_buffer(ox,oy,oz,lx,ly,lz,src.data(),ly,lz);
    serial.pack_subvolume_to_buffer(ox,oy,oz,lx,ly,lz,a.data(),ly,lz);
    for (int repeat=0; repeat<4; ++repeat) {
        omp_set_num_threads(4);
        random_numbers<double> threaded(period, 8, 12345, origin, size);
        threaded.unpack_subvolume_from_buffer(ox,oy,oz,lx,ly,lz,src.data(),ly,lz);
        threaded.pack_subvolume_to_buffer(ox,oy,oz,lx,ly,lz,b.data(),ly,lz);
        if(a != b) throw std::runtime_error("OMP periodic writeback changed");
    }

    // Compare against cell-order reference, not just two implementations
    // sharing the same block-order bug. The last visit wins on each axis.
    for(int i=0; i<lx; ++i) for(int j=0; j<ly; ++j) for(int k=0; k<lz; ++k) {
        int ii=i%period+(lx-1-i%period)/period*period;
        int jj=j%period+(ly-1-j%period)/period*period;
        int kk=k%period+(lz-1-k%period)/period*period;
        const size_t q=((size_t)i*ly+j)*lz+k;
        if(a[q] != src[((size_t)ii*ly+jj)*lz+kk])
            throw std::runtime_error("periodic last-write reference changed");
    }

    // Duplicate cube filling must preserve the seeded realization too.
    omp_set_num_threads(1);
    random_numbers<double> seed1(period, 8, 12345, origin, size);
    seed1.pack_subvolume_to_buffer(ox,oy,oz,lx,ly,lz,a.data(),ly,lz);
    omp_set_num_threads(4);
    random_numbers<double> seed4(period, 8, 12345, origin, size);
    seed4.pack_subvolume_to_buffer(ox,oy,oz,lx,ly,lz,b.data(),ly,lz);
    if(a != b) throw std::runtime_error("OMP periodic cube filling changed");
}

int main(int argc, char **argv)
{
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
#endif
    int failed=0;
    try {
        check(40,40,40,-4,-4,-4);  // alias in all directions
        check(40,24,24,3,3,3);     // alias in x
        check(24,40,24,-4,-4,-4);  // alias in y
        check(24,24,40,3,3,3);     // alias in z
        check(72,40,40,-4,-4,-4);  // more than two visits
        check(24,24,24,-4,-4,-4);  // boundary crossing, but unique cells
        check(32,32,32,3,3,3);     // exactly one period
        std::cout << "RNG_PERIODIC_UNPACK=passed" << std::endl;
    } catch(const std::exception &e) {
        std::cerr << e.what() << std::endl;
        failed=1;
    }
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return failed;
}
