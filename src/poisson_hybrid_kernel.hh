#ifndef MUSIC_POISSON_HYBRID_KERNEL_HH
#define MUSIC_POISSON_HYBRID_KERNEL_HH

#include <cmath>

template<int order>
inline double poisson_hybrid_kernel( int /*idir*/, int /*i*/, int /*j*/, int /*k*/, int /*n*/ )
{
	return 1.0;
}

template<>
inline double poisson_hybrid_kernel<2>(int idir, int i, int j, int k, int n )
{
	if(i==0&&j==0&&k==0)
		return 0.0;

	double
	ki(M_PI*(double)i/(double)n),
	kj(M_PI*(double)j/(double)n),
	kk(M_PI*(double)k/(double)n),
	kr(sqrt(ki*ki+kj*kj+kk*kk));

	double grad = 1.0, laplace = 1.0;

	if( idir==0 )
		grad = sin(ki);
	else if( idir==1 )
		grad = sin(kj);
	else
		grad = sin(kk);

	laplace = 2.0*((-cos(ki)+1.0)+(-cos(kj)+1.0)+(-cos(kk)+1.0));

	double kgrad = 1.0;
	if( idir==0 )
		kgrad = ki;
	else if( idir ==1)
		kgrad = kj;
	else if( idir ==2)
		kgrad = kk;

	return kgrad/kr/kr-grad/laplace;
}

template<>
inline double poisson_hybrid_kernel<4>(int idir, int i, int j, int k, int n )
{
	if(i==0&&j==0&&k==0)
		return 0.0;

	double
	ki(M_PI*(double)i/(double)n),
	kj(M_PI*(double)j/(double)n),
	kk(M_PI*(double)k/(double)n),
	kr(sqrt(ki*ki+kj*kj+kk*kk));

	double grad = 1.0, laplace = 1.0;

	if( idir==0 )
		grad = 0.166666666667*(-sin(2.*ki)+8.*sin(ki));
	else if( idir==1 )
		grad = 0.166666666667*(-sin(2.*kj)+8.*sin(kj));
	else if( idir==2 )
		grad = 0.166666666667*(-sin(2.*kk)+8.*sin(kk));

	laplace = 0.1666666667*((cos(2*ki)-16.*cos(ki)+15.)
						   +(cos(2*kj)-16.*cos(kj)+15.)
						   +(cos(2*kk)-16.*cos(kk)+15.));

	double kgrad = 1.0;
	if( idir==0 )
		kgrad = ki;
	else if( idir ==1)
		kgrad = kj;
	else if( idir ==2)
		kgrad = kk;

	return kgrad/kr/kr-grad/laplace;
}

template<>
inline double poisson_hybrid_kernel<6>(int idir, int i, int j, int k, int n )
{
	double
	ki(M_PI*(double)i/(double)n),
	kj(M_PI*(double)j/(double)n),
	kk(M_PI*(double)k/(double)n),
	kr(sqrt(ki*ki+kj*kj+kk*kk));

	if(i==0&&j==0&&k==0)
		return 0.0;

	double grad = 1.0, laplace = 1.0;

	if( idir==0 )
		grad = 0.0333333333333*(sin(3.*ki)-9.*sin(2.*ki)+45.*sin(ki));
	else if( idir==1 )
		grad = 0.0333333333333*(sin(3.*kj)-9.*sin(2.*kj)+45.*sin(kj));
	else if( idir==2 )
		grad = 0.0333333333333*(sin(3.*kk)-9.*sin(2.*kk)+45.*sin(kk));

	laplace = 0.01111111111111*(
								(-2.*cos(3.0*ki)+27.*cos(2.*ki)-270.*cos(ki)+245.)
								+(-2.*cos(3.0*kj)+27.*cos(2.*kj)-270.*cos(kj)+245.)
								+(-2.*cos(3.0*kk)+27.*cos(2.*kk)-270.*cos(kk)+245.));

	double kgrad = 1.0;
	if( idir==0 )
		kgrad = ki;
	else if( idir ==1)
		kgrad = kj;
	else if( idir ==2)
		kgrad = kk;

	return kgrad/kr/kr-grad/laplace;
}

#endif
