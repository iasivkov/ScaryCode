#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <cmath>
#include <time.h>
#include "bessel.h"
#include "expintegrals.h"
#include <fstream>

//массы компонент
const float PI = 3.1415926;

float Md = 1.0f;
float Mh = 5.8f;
float Mb = 0.333333f;
float Ms = 0.1*Md;

//характерные размеры диска
float h = 1.0f;
float z0 = 0.2f;
float r0 = 1.0f/3.5f;
int NforDisc = 16384;
int NforBulge = 16384;
int NforHalo = 16384;
int NforSatellite =0;

float psi = PI/6.0f;
float xs = 6.0f*h;
float ys = 6.0f*h;
float zs = sqrt(xs*xs + ys*ys) * sin(psi)/cos(psi); 
float4 rs = {xs, ys, zs, 0.0f };
float rabs = sqrtf(xs*xs + ys*ys)/cos(psi); 

float orb;







//-----------------------------------------------------------------------------


float Qt =  2.0f;
float rho0= Md/(4*PI*h*h*z0);
//хар-ки гало
float gam = 1.0f;
float rc = 10.0f;//радиус обрезания
const float q = gam/rc;
float alpha =1.0f/(1 - sqrt(PI)*q*exp(q*q)*(1 - erf(q)));
//ха-рки балджа
float a = 0.05f;
float c = 0.05f;

float4 rref = {sqrt(2.95)*h, sqrt(2.95)*h, 0.0f, 0.0f};

// число частиц
int NParticles = (NforDisc + NforHalo + NforBulge + NforSatellite);
int p=512;

// радиус системы
float  R = 10.0f;

float Rs = 5.0f;

float Rb = 10.0f;
// параметр точности
float  theta = 0.25f;

// шаг по времени
float  TimeStep = 0.08f;

// гравитационная постоянная
float	G = 1.0f;



float4 *X;
float4 *A;
float4 *V;
float Bin[1000];
float Bin2[1000];
float Bin3[1000];
float Bin4[1000];
float Bin5[1000];
float Bin6[1000];


//-----------------------------------------------------------------------------
inline float  random()
{
	
    return  (float)rand() / (float)RAND_MAX ;
}

//-----------------------------------------------------------------------------

//ядро для подсчета взаимодействия
__device__ float4
bodyBodyInteraction(float4 bi, float4 bj, float4 ai, int NforDisc, int NforHalo, int NforBulge, int tile, int p)
{
	float3 r;
	int gtid = blockIdx.x * blockDim.x + threadIdx.x;
	float eps = 0.0f;
	

	/*
	if( gtid <  NforHalo + NforDisc  && gtid >= NforDisc )
	{
		ai.x += 0.0f;
		ai.y += 0.0f;
		ai.z += 0.0f;
		//ai.w += bj.w*bi.w/sqrt(distSqr);
		return ai;
	}
	*/

	if(gtid < NforDisc)
	{
		eps = 0.08f;
	}
	else if(gtid < NforHalo + NforDisc)
	{
		eps = 0.4f;
	}
	else if(gtid < NforHalo + NforDisc + NforBulge)
	{ 
		eps = 0.06f;
	}
	else 
	{
		eps = 0.08f;
	}
	
	r.x = bj.x - bi.x;
	r.y = bj.y - bi.y;
	r.z = bj.z - bi.z;
	// distSqr = dot(r_ij, r_ij) + EPS^2  
	float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
	float R = sqrtf(distSqr);
	float u =R/eps; 
	float invDistCube = 1.0f;
	float fr=1.0f;
	if(u <= 1.0)
	{ 
		invDistCube = 1.0f/eps/eps/eps*(4.0f/3.0f - 6.0f/5.0f*u*u + 1.0f/20.0f*u*u*u*u*u);
		fr = -2.0f/eps*(1.0f/3.0f*u*u - 3.0f/20.0f*u*u*u*u);
	}
	else if(u <= 2.0)
	{
		invDistCube =1/R/R/R *(-1.0f/15.0f + 8.0f/3.0f*u*u*u - 3.0f*u*u*u*u + 6.0f/5.0f*u*u*u*u*u - 1.0f/6.0f*u*u*u*u*u*u);
		fr = -1.0f/15.0f*R - 1.0f/eps*(4.0f/3.0f*u*u - u*u*u + 3.0f/10.0f*u*u*u*u - 1.0f/30.0f*u*u*u*u*u) + 8.0f/5.0f*eps;
	}
	else 
	{
		float distSixth = distSqr * distSqr * distSqr;
		invDistCube = 1.0f/sqrtf(distSixth);
		fr = 1/R;
	}


	float s = bj.w * invDistCube;
	// a_i =  a_i + s * r_ij [6 FLOPS]
	ai.x += r.x * s;
	ai.y += r.y * s;
	ai.z += r.z * s;
	//потенциал
	ai.w += bj.w*bi.w*fr;


	return ai;
}


//-----------------------------------------------------------------------------

//подсчет тайла
__device__ float4
tile_calculation(float4 myPosition, float4 accel,float4 *shPos, int NforDisc, int NforHalo, int NforBulge, int tile, int p)
{

	int i;
	
	for (i = 0; i < blockDim.x; i++) {
	accel = bodyBodyInteraction(myPosition,  shPos[i], accel, NforDisc,  NforHalo,  NforBulge, tile, p); 
}
return accel;
}


//-----------------------------------------------------------------------------

//подсчет ускорения

 void __global__ 
calculate_forces(float4 *devX, float4 *devA,float4 *devV, float4 *globalX, float4 *globalA, float4 *globalV, int N, int p, float timeStep,int NforDisc,int NforHalo,int NforBulge)
{

__shared__ float4 shPosition[512];
//globalX = devX;
//globalA = devA;
float4 myPosition;
int i, tile;
float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};


int gtid = blockIdx.x * blockDim.x + threadIdx.x;

myPosition = devX[gtid];

for (i = 0, tile = 0; i < N; i += p, tile++) 
{
	int idx = tile * blockDim.x + threadIdx.x;
	shPosition[threadIdx.x] = devX[idx];

	__syncthreads();
	
	acc = tile_calculation(myPosition, acc, shPosition, NforDisc, NforHalo, NforBulge,tile,p);
	__syncthreads();
}

// Save the result in global memory for the integration step.

float4 acc4 = {acc.x, acc.y, acc.z, acc.w};
globalA[gtid] = acc4;

float b = devX[gtid].w * ((devV[gtid].x + globalA[gtid].x * timeStep/2.0)*(devV[gtid].x + globalA[gtid].x * timeStep/2.0) + (devV[gtid].y + globalA[gtid].y * timeStep/2.0)*(devV[gtid].y + globalA[gtid].y * timeStep/2.0) + (devV[gtid].z + globalA[gtid].z * timeStep/2.0)*(devV[gtid].z + globalA[gtid].z * timeStep/2.0))/2.0;
float velx = devV[gtid].x + globalA[gtid].x * timeStep ;
float vely = devV[gtid].y + globalA[gtid].y * timeStep ;
float velz = devV[gtid].z + globalA[gtid].z * timeStep ;
float4 vel4 = {velx, vely, velz, b };
globalV[gtid] = vel4;

float posx = devX[gtid].x + globalV[gtid].x * timeStep;
float posy = devX[gtid].y + globalV[gtid].y * timeStep;
float posz = devX[gtid].z + globalV[gtid].z * timeStep;
float4 pos = {posx, posy, posz, devX[gtid].w };
globalX[gtid] = pos;

}

//-----------------------------------------------------------------------------





//-----------------------------------------------------------------------------
//куммулятивная масса диска
float Mu(float r,float z)
{
	//float sech = 2.0/(exp(z/z0)+exp(-z/z0));
	float cons=Md/h/h;

	if(z<0.0)z*=-1.0;
	if (z == 0.0)
	{
		//z=0.001;
		return cons*(h*h-h*h*exp(-r/h)-h*exp(-r/h)*r);
	}
	/*if (z == 0.0)
	{
		z=0.001;
		(h*h-h*h*exp(-r/h)-h*exp(-r/h)*r)*
	}*/
	
	return cons *(exp(z/z0)- exp(-z/z0))/(exp(z/z0)+ exp(-z/z0));
	
}

//макс масса по r

//потенциал диска
/*float Q(float R, float z)
{
	float cons =  G * Md ;
	float K = 10.0f;
	float sum = (besselj0(0) + besselj0(K*R) * exp(-K*fabs(z))/ pow(1 + (K*K*h*h),1.5f) )/2.0f;
	for(int i = 0; i<30; i++)
	{
		float k = K/30.0f*i;
		sum += besselj0(k*R)*exp(-k*fabs(z))/pow(1+(k*k*h*h),1.5f);
	}
	return cons * sum * K/30.0f;
}*/


float Vc2(float r)
{
	float y = r/2.0/h;
	return 4.0*PI*G*rho0*2.0*z0*h*y*y*(besseli0(y)*besselk0(y) - besseli1(y)*besselk1(y));
}


//dQ/dR
float dQ(float R, float z)
{
	float cons =  -G * Md ;
	float K = 10.0f;
	float sum = -( besselj1(K*R) *K* exp(-K*fabs(z))/ pow(1 + (K*K*h*h),1.5f) )/2.0f;
	for(int i = 0; i<30; i++)
	{
		float k = K/30.0f*i;
		sum += -besselj1(k*R)*k*exp(-k*fabs(z))/pow(1+(k*k*h*h),1.5f);
	}
	return cons * sum * 1.0f/3.0f;
}
//d^2Q/dR^2
float ddQ(float R, float z)
{
	if(R == 0.0f)return 0.0f;
	float cons =  -G * Md ;
	float K = 10.0f;
	float sum = -1.0 *((besselj0(K*R)*K*K - besselj1(K*R) *K / R)* exp(-K*fabs(z))/ pow(1 + (K*K*h*h),1.5f) )/2.0f;
	for(int i = 0; i<30; i++)
	{
		float k = 1.0f/3.0f*i;
		sum += -1.0*((besselj0(k*R)*k*k - besselj1(10*R) *k / R)* exp(-k*fabs(z))/pow(1+(k*k*h*h),1.5f));
	
	}
return cons * sum * K/30.0f;
}


//куммулятивная масса гало
float fh(float x)
{
	return x*x*exp(-x*x)/(x*x + q*q);
}

float Muh(float r)
{
	float sum = (fh(0)+fh(r/rc))/2.0;
	int n = 30;
	for(int i=1; i<29;i++)
	{
		sum+=fh(i*r/rc/n);
	}


	return 2*Mh*alpha/sqrt(PI)*sum*(r/rc/30.0);
}


//куммулятивная масса балджа
/*float Mub(float m)
{
	return Mb * m * m /(1 + m)/(1 + m);
}
float Rhob(float m)
{
	return Mb/2.0f/PI/a/c/m/(1 + m)/(1 + m)/(1 + m);
}*/
float Mub(float r)
{
	return Mb*r*r /(r + a)/(r + a); 
}



float Rhob(float r)
{
	return Mb/2.0f/PI*a/r/(r + a)/(r + a)/(r + a);
}
//распределеие гаусса


float F(float v,float sig )
{
	//4*PI*pow(2.0*PI*sig*sig,-3.0/2.0)*v*v*exp(-v*v/2.0/sig/sig)
	
	return 4.0*PI*pow(2.0*PI*sig*sig,-3.0/2.0)*(-1.0*v*exp(-v*v/2.0/sig/sig)*sig*sig + 1.25331413*sig*sig*sig*erf(v/sqrt(2.0)/sig)  ) ;
	//return  0.5*(1 + erf(v/sig));
}
float F2(float v,float sig )
{
	
	return  0.5*(1 + erf(v/sig));
}


float veldistr(float sig, float Vesc)
{
 float v=random()*Vesc;
 float maxf=F(10.0,sig);
 float fr=random()*maxf;
 while(fr<F(v,sig))
 {
	v=random()*Vesc;
	fr=random()*maxf;
 }
  return v;
}
 
float gauss(float sig)
{
	
	float v=random()*10.0;
	//if(v < 0.0)t=-1.0;
	float maxf=1.0;
	float fr=(0.5*random() + 0.5)*maxf;
	while(fr<F2(v,sig))
	{
		v=random()*10.0;
		float fr=(0.5*random() + 0.5)*maxf;
		//if(v < 0.0)t=-1.0;
		
	}
	if(random()<0.5)return -v;
	else return v;
    
}


//экспоненциальный интеграл
float Ei(float x)
{
	return	exponentialintegralei(x);
}

//функция плотности гало
float Rho(float r)
{
	return Mh*alpha/2.0/pow(PI,1.5f)/rc*exp(-r*r/rc/rc)/(r*r + gam*gam);
}

//потенциал гало
float Qh(float r)
{
	
	//гало, балдж, диск все компоненты
	
	return -G*(Muh(r) + Mub(r/a) + Mu(r,r))/r;
	
	
}




float Q(float r)
{			
	
			//гало, балдж, диск все компоненты
			//if(r<0.01)return -r;
			int L=r/R*1000;
			if(r==R)L=999;
			r=(L+1.0)*R/1000.0;
			float sum = 0;
			for(int i = 0; i<=L; i++)
				{
					sum += Bin[i];
				}
			return -G*sum/r;
		
	

	
}

float Q2(float r)
{			
	//гало, балдж
			//if(r<0.01)return -r;
			int L=r/R*1000;
			if(r == R)L=999;
			r=(L+1.0)*R/1000.0;
			float sum = 0;
			for(int i = 0; i<=L; i++)
				{
					sum += Bin2[i];
				}
			return -G*sum/r;
		

	
}

float Q3(float r)
{			
	// балдж
			//if(r<0.01)return -r;
			int L=r/R*1000;
			if(r == R)L=999;
			r=(L+1.0)*R/1000.0;
			float sum = 0;
			for(int i = 0; i<=L; i++)
				{
					sum += Bin3[i];
				}
			return -G*sum/r;
		

	
}

float Qs(float r)
{
			//if(sqrtr<0.01)return -r;
			int L=r /Rs*1000;
			if(r == Rs)L=999;
			r=(L+1.0)*Rs/1000.0;
			float sum = 0;
			for(int i = 0; i<=L; i++)
				{
					sum += Bin4[i];
				}
			return -G*sum/r;

}

float Q5(float r)
{
			//if(sqrtr<0.01)return -r;
			int L=r /R*1000;
			if(r == R)L=999;
			float sum = 0;
			for(int i = 0; i<=L; i++)
				{
					sum += Bin5[i];
				}
			return -G*sum/r;

}

float Q6(float r)
{
			//if(sqrtr<0.01)return -r;
			int L=r /R*1000;
			if(r == R)L=999;
			float sum = 0;
			for(int i = 0; i<=L; i++)
				{
					sum += Bin6[i];
				}
			return -G*sum/r;

}
//распределение плотности спутника (Яффе)
float Rhos(float r)
{
	return Ms / (4 * PI * r0 * r0 * r0) * (r0 / r) * (r0 / r) / (1 + r/r0 ) / (1 + r/r0 );
}

//куммулятивная масса спутника
float Mus(float r)
{
	return Ms*r / (r0 + r); 
}



//sigma^2
float sigma2(float r,int p)
{	
	float sum=0;
	if(p>R)p=R;

	//if(r < 4.0*h/3.0 && p==1) sum =( Rho(r) * G * fabsf(Q3(r))/r + Rho(R) * G * fabsf(Q3(R))/R)/2.0; 
	if(p==1) sum =( Rho(r)  * (-Q3(r))/r + Rho(R) *  (-Q3(R))/R)/2.0;
    else if(p==2) sum =( Rhob(r)  * (-Q2(r))/r + Rhob(R) *  (-Q2(R))/R)/2.0;
	else if(p==3) sum = ( Rhos(r)  * (-Qs(r))/r + Rhos(Rs) *  (-Qs(Rs))/Rs)/2.0;
	
	int n = 30;
	for(int i=1; i<29;i++)
	{
		//if(r < 4.0*h/3.0 && p==1) sum += Rho(i*(R-r)/n + r) * G * fabsf(Q3(i*(R-r)/n + r))/(i*(R-r)/n + r);
		if(p==1) sum += Rho(i*(R-r)/n + r)  * (-Q3(i*(R-r)/n + r))/(i*(R-r)/n + r);
		else if(p==2) sum += Rhob(i*(R-r)/n + r/a) * (-Q2(i*(R-r)/n + r))/(i*(R-r)/n + r);
		else if(p==3) sum += Rhos(i*(Rs-r)/n + r) * (-Qs(i*(Rs-r)/n + r))/(i*(Rs-r)/n + r);
	}
		
	if(p == 1) return 1.0/Rho(r) * sum * (R-r)/n;
	else if(p == 2)  return 1.0/Rhob(r) * sum * (R-r)/n;
	else if(p == 3) return 1.0/Rhos(r) * sum * (Rs-r)/n;
	
}







// начальное распределение частиц диска, гало, балджа
void    InitParticles()
{	
	char FileName4[32];
	sprintf(FileName4,"Vr.txt");
	FILE * out4 = fopen(FileName4, "w+");
	srand(time(NULL));
	X = new float4[NParticles];
	A = new float4[NParticles];
	V = new float4[NParticles];
	
	for(int i=0; i<1000 ;i++)
	{
		Bin[i] = 0;
	}
	for(int i=0; i<1000 ;i++)
	{
		Bin2[i] = 0;
	}
	for(int i=0; i<1000 ;i++)
	{
		Bin3[i] = 0;
	}
	for(int i=0; i<1000 ;i++)
	{
		Bin4[i] = 0;
	}
	for(int i=0; i<1000 ;i++)
	{
		Bin5[i] = 0;
	}
	for(int i=0; i<1000 ;i++)
	{
		Bin6[i] = 0;
	}



	float ref = 0.0f;
	int numref = 0;
	
	
	float Mumh = Muh(R);
	float Mumb = 0;
	float Mums = Mus(Rs);
	float r = 0;
	float phi = 0;
	float teta = 0;
	float x = 0;
	float y = 0;
	float z = 0;
	float m = 0;

	
		Mumb = Mub(Rb);

//	printf("Mumb %f\n", Mumb);
    
	float Mur = 0;
	float Murz = 0;
    //задаем координаты частиц методом отбора-отказа(фон Неймана)
    for (int k=0; k<NParticles; k++)
    {	
		if(k<NforDisc)
		{
			
			float Mum2 = Mu(R,0);
			phi = random()*2.0*PI;
			r = random()*R;
			
			
			Mur = random()* Mum2;
		
			while( Mur <=Mu(r, 0) )
			{
				//phi = random()*2*PI;		
				r = random()*R;
				
				Mur = random() * Mum2;
			
			}

			//if (r==0.0)r+=0.000001;
			float Mum3 = Mu(r,0.6);
			
			Murz = random()* Mum3;
			z = (random()*2.0f - 1.0f)*2.0;
			//printf("Mumb %f\t%f\n", r , z);
			while( Murz <= Mu(r, z) )
			{
				z = (random()*2.0 - 1.0) * 2.0;
				Murz = random() * Mum3;
				
			}
			
			
			X[k].x = r * cos(phi); 
			A[k].x = 0.0f;
			
			
        
			X[k].y = r * sin(phi);
			A[k].y = 0.0f;
           
			
			
			X[k].z = z;
			A[k].z = 0.0f;
			
			
			
			
			X[k].w = Md * 1.0f/NforDisc;
			A[k].w = 0.0f;
			
			
			
			
			int l = r/R*1000;
			
			//int l2=r*0.35/R*1000;
			//Bin3[l2] += X[k].w;
			//if(r==R)l=999;
			if(1000-l<=8)
			{
				for(int i=-8;i<1000-l;i++)
				{	
				int l=(r/R*1000  + i);

				Bin[l] += X[k].w/(8.0 + 1000.0 - l);
				if(r>0.35*h*4.0)Bin3[l] += X[k].w/(8.0 + 1000.0 - l);
				
				}
			}
			else if(l < 8)
			{
			
				for(int i=-l;i<=8;i++)
				{	
					int l=(r/R*1000  + i);
					Bin[l] += X[k].w/(l + 1.0 + 8.0 );
					if(r>0.35*h*4.0) Bin3[l] += X[k].w/(l + 1.0 + 8.0 );
				}
			}
			else
			{
				for(int i=-8;i<=8;i++)
				{	
					int l=(r/R*1000  + i);
					Bin[l] += X[k].w/17.0;
					if(r>0.35*h*4.0)Bin3[l] += X[k].w/17.0;
				}
			}
			//int l3=fabsf(r-rabs)/Rs*1000;
			//if(l3==1000)l3=999;
			//Bin5[l]+=X[k].w;
			//Bin[l] += X[k].w;
			//if(l3<1000)Bin4[l3] += X[k].w;
			
		}
		else if(k< (NforHalo + NforDisc))
		{
			r = random()*R;
			Mur = random()* Mumh;
			while(Mur < Muh(r/rc))
			{
			  r = random()*R;
			  Mur = random()*Mumh;

			}

			
		
			
			phi = phi = random()*2.0*PI;;
			teta = asin(random()*2.0f - 1.0f);
			X[k].x = r * cos(phi)*cos(teta); 
			A[k].x = 0.0f;
			
			
        
			X[k].y = r * sin(phi)*cos(teta);
			A[k].y = 0.0f;
            
			
			
			X[k].z = r * sin(teta);
			A[k].z = 0.0f;
			
			
			
			
			X[k].w = Mh * 1.0f/NforHalo;
			A[k].w = 0.0f;
			
			int l=r/R*1000;
			//if(r==R)l=999;
			//printf("Mumb %d\t%f\t%f\n",l, r , r/R);
			//int l3=fabsf(r-rabs)/Rs*1000;
			//if(l3==1000)l3=999;
			if(1000-l<=40)
			{
				for(int i=-40;i<1000-l;i++)
				{	
				int l=(r/R*1000  + i);

				Bin[l] += X[k].w/(40.0 + 1000.0 - l);
				Bin2[l] += X[k].w/(40.0 + 1000.0 - l);
				Bin3[l] += X[k].w/(40.0 + 1000.0 - l);
				Bin6[l] += X[k].w/(40.0 + 1000.0 - l);

				}
			}
			else if(l<40)
			{
			
				for(int i=-l;i<=40;i++)
				{	
					int l=(r/R*1000  + i);
					Bin[l] += X[k].w/(l + 1.0 + 40.0 );
					Bin2[l] += X[k].w/(l + 1.0 + 40.0 );
					Bin3[l] += X[k].w/(l + 1.0 + 40.0 );
					Bin6[l] += X[k].w/(l + 1.0 + 40.0 );
				}
			}
			else
			{
				for(int i=-40;i<=40;i++)
				{	
					int l=(r/R*1000  + i);
					Bin[l] += X[k].w/81.0;
					Bin2[l] += X[k].w/81.0;
					Bin3[l] += X[k].w/81.0;
					Bin6[l] += X[k].w/81.0;
				}
			}
			
			//Bin[l] += X[k].w;
			//Bin2[l]+= X[k].w;
			
		}
		else if(k < NforDisc + NforHalo + NforBulge)
		{
			
			r = random()*Rb;
			phi = phi = random()*2.0*PI;
			teta = asin(random()*2.0 -1.0);
			x = r * cos(phi)*cos(teta);
			y = r * sin(phi)*cos(teta);
			z = r * sin(teta);
			//m = (x*x+y*y)/a/a + z*z/c/c;
			Mur = random()* Mumb;
			while(Mur <= Mub(r))
			{
				r = random()*Rb;
				//phi = phi = random()*2.0*PI;;
				//teta = asin(random()*2-1);
				//x = r * cos(phi)*cos(teta);
				//y = r * sin(phi)*cos(teta);
				//z = r * sin(teta);
			//	m = (x*x+y*y)/a/a + z*z/c/c;
				Mur = random()* Mumb;
			}

			
			

		
			X[k].x = r * cos(phi)*cos(teta); 
			A[k].x = 0.0f;
			
			
        
			X[k].y = r * sin(phi)*cos(teta);
			A[k].y = 0.0f;
           
			
			
			X[k].z = r * sin(teta);
			A[k].z = 0.0f;
			
			
			
			
			X[k].w = Mb * 1.0f/NforBulge;
			A[k].w = 0.0f;
			
			int l=r/R*1000;
			//int l2=r/R*5000;
			//int l3=fabsf(r-rabs)/Rs*1000;
			//if(l3==1000)l3=999;
			//
			//if(l2=2000)l2=1999;
			//float rabs = sqrtf(xs*xs + ys*ys + zs*zs); 
			if(1000-l<=6)
			{
				for(int i=-6;i<1000-l;i++)
				{	
				int l=(r/R*1000  + i);

				Bin[l] += X[k].w/(12.0 + 1000.0 - l);
				Bin2[l] += X[k].w/(12.0 + 1000.0 - l);
				Bin3[l] += X[k].w/(12.0 + 1000.0 - l);
				Bin5[l] += X[k].w/(12.0 + 1000.0 - l);
				}
			}
			else if(l<6)
			{
			
				for(int i=-l;i<=6;i++)
				{	
					int l=(r/R*1000  + i);
					Bin[l] += X[k].w/(l + 1.0 + 12.0 );
					Bin2[l] += X[k].w/(l + 1.0 + 12.0 );
					Bin3[l] += X[k].w/(l + 1.0 + 12.0 );
					Bin5[l] += X[k].w/(l + 1.0 + 12.0 );
				}
			}
			else
			{
				for(int i=-6;i<=6;i++)
				{	
					int l=(r/R*1000  + i);
					Bin[l] += X[k].w/13.0;
					Bin2[l] += X[k].w/13.0;
					Bin3[l] += X[k].w/13.0;
					Bin5[l] +=  X[k].w/13.0;
				}
			}
			//Bin[l] += X[k].w;
			//Bin2[l] += X[k].w;
			//if(l3<1000)Bin4[l3] += X[k].w;
			
			//Bin3[l] +=  X[k].w;
			//Bin5[l] +=  X[k].w;
		}
		else
		{
			
			r = random()*Rs;
			phi = random()*2.0*PI;
			teta = asin(random()*2.0 -1.0);
			x = xs + r * cos(phi)*cos(teta);
			y = ys + r * sin(phi)*cos(teta);
			z = zs + r * sin(teta);
			
			Mur = random()* Mums;
			while(Mur <= Mus(r))
			{
				r = random()*Rs;
				
				x = xs + r * cos(phi)*cos(teta);
				y = ys + r * sin(phi)*cos(teta);
				z = zs + r * sin(teta);
				Mur = random()* Mums;
			}
			//printf("Mumb %f\t%f\t%f\t%f\n", r , Mums,Mus(r),Mur);
			
			

		
			X[k].x = x;
			A[k].x = 0.0f;
			
			
        
			X[k].y = y;
			A[k].y = 0.0f;
           
			
			
			X[k].z = z;
			A[k].z = 0.0f;
			
			
			
			
			X[k].w = Ms * 1.0f/NforSatellite;
			A[k].w = 0.0f;
			
			int l=r/R*1000;
			

			Bin4[l] +=  X[k].w;
		}
		
		
            
	}

	for(int k=0; k<NforDisc; k++)
	{
		float r = sqrt(X[k].x*X[k].x + X[k].y*X[k].y);
		if( sqrt((X[k].x-rref.x)*(X[k].x-rref.x) + (X[k].y-rref.y)*(X[k].y-rref.y))< h/5.0 )
			{	
				//float t = sqrt(X[k].x*X[k].x + X[k].y*X[k].y + X[k].z*X[k].z);
				//float O = sqrt(dQ(r,0)/r);//sqrt((dQ(r,0) + G*(Mub(m)+Muh(t))/r/r)/r);
				float K =sqrt(fabs(3.0*(dQ(r,0) - Q2(r)/r)/r + (ddQ(r,0)+2.0f*Q2(r)/r/r)));
				//float K =sqrt(fabs(3.0*(dQ(r,X[k].z) + G*(Mub(m)+Muh(t))/r/r)/r + (ddQ(r,X[k].z)-2.0f*G*(Mub(m)+Muh(t))/r/r/r)));;
				ref += 3.36 * G * rho0 * 2.0 * z0 * exp(-r/h)/K;
					numref++;
			}
	}

	float Scrit = ref/numref;


	float c0 = Qt*Scrit*exp(2.428/2.0/h);

   // printf("halo %f\n",c0  );
	system("pause");

	//задаем начальые скорости частиц через БУБ
	float vteta = 0;
	float vphi = 0;
	float vr = 0;
	
	char FileName6[32];
	sprintf(FileName6,"VrH.txt");
	FILE * out6 = fopen(FileName6, "w+");

	for (int k=0; k<NParticles; k++)
	{
		//диск
		if(k<NforDisc)
		{	
			
			r = sqrt(X[k].x * X[k].x + X[k].y * X[k].y);
			float t = sqrt(X[k].x * X[k].x + X[k].y * X[k].y + X[k].z*X[k].z);
				if(r == 0.0)
		{

			V[k].x = 0.0f;
			V[k].y = 0.0f;
			V[k].z = 0.0f;
			V[k].w = 0.0f;
			continue;
		}
			
			

			float O =sqrt(dQ(r,0)/r - Q2(r)/r/r);//sqrt(-Q(r))/r;//sqrt(dQ(r,0)/r - Q2(r)/r/r);//sqrt(-Q(r))/r;
			float K =sqrt(fabs(3.0*(dQ(r,0) - Q2(r)/r)/r + (ddQ(r,0)+2.0f*Q2(r)/r/r)));//sqrt(fabs(3.0*( - Q(r)/r)/r + 2.0f*Q(r)/r/r));//sqrt(fabs(3.0*(dQ(r,0) - Q2(r)/r)/r + (ddQ(r,0)+2.0f*Q2(r)/r/r)));
			float VR2 = c0*c0*exp(-sqrt(r*r)/h);//c0*c0*exp(-sqrt(r*r + h*h/8.0f)/h);//3.36 * G * rho0 * 2.0 * z0 * exp(-r/h)/K*Qt;
			float Vphi2 =VR2*K*K/O/O/4; //VR2*K*K/O/O/4;//
			float Vz2 = PI*G*rho0*2.0*z0*exp(-sqrt(r*r)/h)*z0;//PI*G*rho0*2.0*z0*exp(-sqrt(r*r + h*h/8.0f)/h)*z0;
			float Vphis = sqrt(fabs(Vc2(r)- Q2(r) - VR2 * (1 - K * K /(4.0 * O * O) - 2*sqrt(X[k].x * X[k].x + X[k].y * X[k].y)/h)) );//sqrt(fabs((dQ(r,0) + G*(Mub(m)+Muh(t))/r/r)*r - VR2 * (1 - K * K /(4.0 * O * O) - 2*sqrt(X[k].x * X[k].x + X[k].y * X[k].y)/h)) );
			float Vz = gauss(sqrt(Vz2));
			float VR = gauss(sqrt(VR2));
			float Vphi = gauss(sqrt(Vphi2));
			
			x = -(Vphis + Vphi)*X[k].y/r - VR*X[k].x/r ;
			y = (Vphis + Vphi)*X[k].x/r - VR*X[k].y/r ;
			z = Vz;
			//float vm = sqrtf((Vphis + Vphi) *(Vphis + Vphi)  + Vz * Vz + VR * VR);
			float v = sqrt(x * x + y * y + z * z); 
			/*if(X[k].x>-0.01 && X[k].x<0.01 && X[k].y<0.01 && X[k].y>-0.01)
			{
				x = r*6.0 ;
				y = r*6.0 ;
				z = r*6.0;
			}*/
		
		
			//printf("test %f\t%f\t%f\n",r ,v,Q2(r));
			//printf("test %f\t%f\n",r ,dQ(14.5,0));
			fprintf(out4,"%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n", r, sqrt(Vz2),sqrt(VR2),-Q6(r),K,-Q5(r),Vc2(r),Vc2(r)-Q2(r) );
			V[k].x = x ;
			V[k].y = y ;
			V[k].z = z;
			V[k].w = 0;
			
				if(k== NforDisc-1 )
		{ 
			fclose(out4); 
			system("pause");
		}
		}
	
		//гало
		else if(k < (NforHalo + NforDisc) )
		{
			
			r = sqrtf(X[k].x * X[k].x + X[k].y * X[k].y + X[k].z * X[k].z);
		//	printf("halo %f\t%f\t%f\t%f\n", X[k].x, X[k].y, X[k].z, r );
					if(fabs(r) == 0.0)
		{

			V[k].x = 0.0f;
			V[k].y = 0.0f;
			V[k].z = 0.0f;
			V[k].w = 0.0f;
			continue;
		}
			
		
		if(r < 0.003)
		{
			V[k].x = 1.0f/sqrtf(3.0f)* sqrtf(2*fabsf(Q(r)))*random();
			V[k].y = 1.0f/sqrtf(3.0f)* sqrtf(2*fabsf(Q(r)))*random();
			V[k].z = 1.0f/sqrtf(3.0f)* sqrtf(2*fabsf(Q(r)))*random();
			V[k].w = 0;
			continue;
		}


			float VR2 = sigma2(r,1);
			//float Vteta2 = sigma2(r,1);
			//float Vphi2 = sigma2(r,1);

		//	printf("halo disp %f\t%f\t%f\t%f\n", X[k].x, X[k].y, X[k].z, VR2 );
			
			
			float vm = 0.0;
			//phi = acos(X[k].x/r);
			//teta = asin(X[k].z/r);
		
			
			


	/*if(X[k].x>-0.0001 && X[k].x<0.0001 && X[k].y<0.0001 && X[k].y>-0.0001)
			{
				x = -vteta; 
				y = -vphi; 
				z = -vr;
				printf("halo %f\t%f\t%f\t%f\n", X[k].x, X[k].y, X[k].z, x);
			}
				
			else
			{
				
				x = vr * (-X[k].x/sqrtf(r*r - X[k].z*X[k].z)) * sqrtf(r*r - X[k].z*X[k].z)/r + vphi * (X[k].y/sqrtf(r*r - X[k].z*X[k].z)) + vteta * X[k].z/r * (-X[k].x/sqrtf(r*r - X[k].z*X[k].z)); 
				y = vr * (-X[k].y/sqrtf(r*r - X[k].z*X[k].z)) * sqrtf(r*r - X[k].z*X[k].z)/r + vphi * (-X[k].x/sqrtf(r*r - X[k].z*X[k].z)) + vteta * X[k].z/r * (-X[k].y/sqrtf(r*r - X[k].z*X[k].z)); 
				z = vr * (-X[k].z/r)  + vteta * sqrtf(r*r - X[k].z*X[k].z)/r;
			}
			
	*/		 


			//float v = sqrtf(x * x + y * y + z * z); 
			float Vesc = sqrtf(2*fabsf(Q(r)));
			vr = veldistr(sqrtf(VR2),10.0);
			//vteta = veldistr(sqrtf(VR2),10.0);
			//vphi = veldistr(sqrtf(VR2),10.0);
			//float v = sqrtf(vteta * vteta + vphi * vphi + vr * vr);
			
			while(vr > 0.95 * Vesc)
			{
				
				
				vr = veldistr(sqrtf(VR2),10.0);
				//vteta = veldistr(sqrtf(VR2),10.0);
				//vphi = veldistr(sqrtf(VR2),10.0);
				//vr = gauss(sqrtf(VR2));
				//vteta = gauss(sqrtf(VR2));
				//vphi = gauss(sqrtf(VR2));

				//v = sqrtf(vteta * vteta + vphi * vphi + vr * vr);
				
				//	printf("while %f\n", v);
					

			/*if(X[k].x>-0.0001 && X[k].x<0.0001 && X[k].y<0.0001 && X[k].y>-0.0001)
			{
				x = -vteta; 
				y = -vphi; 
				z = -vr;
				//printf("halo %f\t%f\t%f\t%f\n", X[k].x, X[k].y, X[k].z, x);
			}
				
			else
			{
				
				x = vr * (-X[k].x/sqrtf(r*r - X[k].z*X[k].z)) * sqrtf(r*r - X[k].z*X[k].z)/r + vphi * (X[k].y/sqrtf(r*r - X[k].z*X[k].z)) + vteta * X[k].z/r * (-X[k].x/sqrtf(r*r - X[k].z*X[k].z)); 
				y = vr * (-X[k].y/sqrtf(r*r - X[k].z*X[k].z)) * sqrtf(r*r - X[k].z*X[k].z)/r + vphi * (-X[k].x/sqrtf(r*r - X[k].z*X[k].z)) + vteta * X[k].z/r * (-X[k].y/sqrtf(r*r - X[k].z*X[k].z)); 
				z = vr * (-X[k].z/r)  + vteta * sqrtf(r*r - X[k].z*X[k].z)/r;
			}*/
			
				
			
				 //v = sqrtf(x * x + y * y + z * z);
				//v = sqrtf(vteta * vteta + vphi * vphi + vr * vr);
			}
			
			phi  = random()*2.0*PI;
			teta = asin(random()*2.0f - 1.0f);
			//vr=0.0;
			x = vr * cos(phi)*cos(teta); 
			y = vr * sin(phi)*cos(teta);
			z = vr * sin(teta);
			//vm = sqrtf(x * x + y * y + z * z);
			//printf("halo %f\t%f\t%f\t%f\n", r, Vesc, vr,0.0);
			V[k].x = x;
			V[k].y = y;
			V[k].z = z;
			V[k].w = 0;
			//printf("while%i \n",k);
		}
		//балдж
		else if(k < NforDisc + NforHalo + NforBulge)
		{r = sqrtf(X[k].x * X[k].x + X[k].y * X[k].y + X[k].z * X[k].z);
		 
		
			if(r == 0.0)
				{
					V[k].x = 0.0f;
					V[k].y = 0.0f;
					V[k].z = 0.0f;
					V[k].w = 0.0f;
					continue;
				}
		
			/*if(1)
				{
					V[k].x = 0;
			V[k].y = 0;
			V[k].z = 0;
			V[k].w = 0;
			continue;
				}*/


			/*	if(X[k].x>-0.0001 && X[k].x<0.0001 && X[k].y<0.0001 && X[k].y>-0.0001)
			{
				x = -vteta; 
				y = -vphi; 
				z = -vr;
				printf("bulge %f\t%f\t%f\t%f\n", X[k].x, X[k].y, X[k].z, x);
			}
				
			else
			{
				
				x = vr * (-X[k].x/sqrtf(r*r - X[k].z*X[k].z)) * sqrtf(r*r - X[k].z*X[k].z)/r + vphi * (X[k].y/sqrtf(r*r - X[k].z*X[k].z)) + vteta * X[k].z/r * (-X[k].x/sqrtf(r*r - X[k].z*X[k].z)); 
				y = vr * (-X[k].y/sqrtf(r*r - X[k].z*X[k].z)) * sqrtf(r*r - X[k].z*X[k].z)/r + vphi * (-X[k].x/sqrtf(r*r - X[k].z*X[k].z)) + vteta * X[k].z/r * (-X[k].y/sqrtf(r*r - X[k].z*X[k].z)); 
				z = vr * (-X[k].z/r)  + vteta * sqrtf(r*r - X[k].z*X[k].z)/r;
			}*/
			


			
			float VR2 = sigma2(r,2);
			//float Vteta2 = sigma2(r,1);
			//float Vphi2 = sigma2(r,1);

		//	printf("halo disp %f\t%f\t%f\t%f\n", X[k].x, X[k].y, X[k].z, VR2 );
			
			
			float vm = 0.0;
			//phi = acos(X[k].x/r);
			//teta = asin(X[k].z/r);
		
			
			


	/*if(X[k].x>-0.0001 && X[k].x<0.0001 && X[k].y<0.0001 && X[k].y>-0.0001)
			{
				x = -vteta; 
				y = -vphi; 
				z = -vr;
				printf("halo %f\t%f\t%f\t%f\n", X[k].x, X[k].y, X[k].z, x);
			}
				
			else
			{
				
				x = vr * (-X[k].x/sqrtf(r*r - X[k].z*X[k].z)) * sqrtf(r*r - X[k].z*X[k].z)/r + vphi * (X[k].y/sqrtf(r*r - X[k].z*X[k].z)) + vteta * X[k].z/r * (-X[k].x/sqrtf(r*r - X[k].z*X[k].z)); 
				y = vr * (-X[k].y/sqrtf(r*r - X[k].z*X[k].z)) * sqrtf(r*r - X[k].z*X[k].z)/r + vphi * (-X[k].x/sqrtf(r*r - X[k].z*X[k].z)) + vteta * X[k].z/r * (-X[k].y/sqrtf(r*r - X[k].z*X[k].z)); 
				z = vr * (-X[k].z/r)  + vteta * sqrtf(r*r - X[k].z*X[k].z)/r;
			}
			
	*/		 


			//float v = sqrtf(x * x + y * y + z * z); 
			float Vesc = sqrtf(2*fabsf(Q(r)));
			vr = veldistr(sqrtf(VR2),10.0);
			//vteta = veldistr(sqrtf(VR2),2.0);
			//vphi = veldistr(sqrtf(VR2),2.0);
			//float v = sqrtf(vteta * vteta + vphi * vphi + vr * vr);
			
			while(vr > 0.95 * Vesc)
			{
				
				
				vr = veldistr(sqrtf(VR2),10.0);
				//vteta = veldistr(sqrtf(VR2),2.0);
				//vphi = veldistr(sqrtf(VR2),2.0);
				//vr = gauss(sqrtf(VR2));
				//vteta = gauss(sqrtf(VR2));
				//vphi = gauss(sqrtf(VR2));

				//v = sqrtf(vteta * vteta + vphi * vphi + vr * vr);
				
				//	printf("while %f\n", v);
					

			/*if(X[k].x>-0.0001 && X[k].x<0.0001 && X[k].y<0.0001 && X[k].y>-0.0001)
			{
				x = -vteta; 
				y = -vphi; 
				z = -vr;
				//printf("halo %f\t%f\t%f\t%f\n", X[k].x, X[k].y, X[k].z, x);
			}
				
			else
			{
				
				x = vr * (-X[k].x/sqrtf(r*r - X[k].z*X[k].z)) * sqrtf(r*r - X[k].z*X[k].z)/r + vphi * (X[k].y/sqrtf(r*r - X[k].z*X[k].z)) + vteta * X[k].z/r * (-X[k].x/sqrtf(r*r - X[k].z*X[k].z)); 
				y = vr * (-X[k].y/sqrtf(r*r - X[k].z*X[k].z)) * sqrtf(r*r - X[k].z*X[k].z)/r + vphi * (-X[k].x/sqrtf(r*r - X[k].z*X[k].z)) + vteta * X[k].z/r * (-X[k].y/sqrtf(r*r - X[k].z*X[k].z)); 
				z = vr * (-X[k].z/r)  + vteta * sqrtf(r*r - X[k].z*X[k].z)/r;
			}*/
			
				
			
				 //v = sqrtf(x * x + y * y + z * z);
				//v = sqrtf(vteta * vteta + vphi * vphi + vr * vr);
			}
			
			phi  = random()*2.0*PI;
			teta = asin(random()*2.0f - 1.0f);
			//v=0.0;
			x = vr * cos(phi)*cos(teta); 
			y = vr * sin(phi)*cos(teta);
			z = vr * sin(teta);
			//vm = sqrtf(x * x + y * y + z * z);
			//printf("bulge %f\t%f\t%f\t%f\n", r, Vesc, vr,0.0);
			V[k].x = x;
			V[k].y = y;
			V[k].z = z;
			V[k].w = 0;
			//printf("while%i \n",k);
		}
		else
		{
			r = sqrtf((X[k].x-xs) * (X[k].x-xs) + (X[k].y-ys) * (X[k].y-ys) + (X[k].z - zs) * (X[k].z - zs));
			//float rab = sqrt(xs*xs + ys*ys + zs*zs); 
			if(r == 0.0)
		{
			V[k].x = 0.0f + sqrtf(-Q(rabs)) * cosf(psi) * (-ys/rabs);
			V[k].y = 0.0f + sqrtf(-Q(rabs)) * cosf(psi) * (-xs/rabs);
			V[k].z = 0.0f + sqrtf(-Q(rabs)) * sinf(psi) * (-ys/rabs);
			V[k].w = 0.0f;
			continue;
		}
				if(r < 0.001)
				{
					V[k].x = 1.0f/sqrtf(3.0f)* sqrtf(2*fabsf(Q(r))) + 1.0f/sqrtf(3.0f)*gauss(0.5*sqrtf(2*fabsf(Q(r)))) + sqrtf(-Q(rabs)) * cosf(psi) * ys/rabs;
					V[k].y = 1.0f/sqrtf(3.0f)* sqrtf(2*fabsf(Q(r))) + 1.0f/sqrtf(3.0f)*gauss(0.5*sqrtf(2*fabsf(Q(r)))) + sqrtf(-Q(rabs)) * cosf(psi) * (-xs/rabs);
					V[k].z = 1.0f/sqrtf(3.0f)* sqrtf(2*fabsf(Q(r))) + 1.0f/sqrtf(3.0f)*gauss(0.5*sqrtf(2*fabsf(Q(r)))) + sqrtf(-Q(rabs)) * sinf(psi) * (-ys/rabs);
					V[k].w = 0.0f;
					continue;
				}

			//vteta = gauss(sqrtf(sigma2(r,3)));
			//vphi = gauss(sqrtf(sigma2(r,3)));
			float VR2 = sigma2(r,3); 


			//phi = acos(X[k].x/r);
			//teta = asin(X[k].z/r);
			//float Xs = X[k].x - xs;
			//float Ys = X[k].y - ys;
			//float Zs = X[k].z - zs;
			//x = vr * (-Xs/sqrtf(r*r - Zs*Zs)) * sqrtf(r*r - Zs*Zs)/r + vphi * (Ys/sqrtf(r*r - Zs*Zs)) + vteta * fabsf(Zs/r) * (-Xs/sqrtf(r*r - Zs*Zs)); 
			//y = vr * (-Ys/sqrtf(r*r - Zs*Zs)) * sqrtf(r*r - Zs*Zs)/r + vphi * -Xs/sqrtf(r*r - Zs*Zs) + vteta * fabsf(Zs/r) * (-Ys/sqrtf(r*r - Zs*Zs)); 
			//z = vr * (-Zs/r)  + vteta * sqrtf(r*r - Zs*Zs)/r;  
			 
				
		//	float v = sqrtf(x * x + y * y + z * z); 
			//float Vesc = sqrtf(2*fabsf(Qs(r)));
			float Vesc = sqrtf(2*fabsf(Qs(r)));
			float vr = veldistr(sqrtf(VR2),10.0);
		
			while(vr > 0.95*Vesc)
			{
				
				//vteta = gauss(sqrtf(sigma2(r,3)));
				//vphi = gauss(sqrtf(sigma2(r,3)));
				vr = veldistr(sqrtf(VR2),10.0);
				//vteta =vteta;//*random()*0.8;
				//vphi =vphi;//*random()*0.8;;
				//vr = vr;//*random()*0.8;;


				//x = vr * (-Xs/sqrtf(r*r - Zs*Zs)) * sqrtf(r*r - Zs*Zs)/r + vphi * (Ys/sqrtf(r*r - Zs*Zs)) + vteta * (Zs/r) * (-Xs/sqrtf(r*r - Zs*Zs)); 
				//	y = vr * (-Ys/sqrtf(r*r - Zs*Zs)) * sqrtf(r*r - Zs*Zs)/r + vphi * (-Xs/sqrtf(r*r - Zs*Zs)) + vteta * (Zs/r) * (-Ys/sqrtf(r*r - Zs*Zs)); 
				//z = vr * (-Zs/r)  + vteta * sqrtf(r*r - Zs*Zs)/r;  
				//v = sqrtf(x * x + y * y + z * z); 
				//printf("bulge %f\t%f\t%f\t%f\t%f\t%f\n",X[k].x,X[k].y,X[k].z, r,vr ,Vesc);
				//system("pause");
			}
			phi  = random()*2.0*PI;
			teta = asin(random()*2.0f - 1.0f);
			x = vr * cos(phi)*cos(teta); 
			y = vr * sin(phi)*cos(teta);
			z = vr * sin(teta);

			
			V[k].x = x + sqrtf(-Q(rabs))  * (-ys/rabs) * orb;
			V[k].y = y + sqrtf(-Q(rabs))  * (xs/rabs) * orb;
			V[k].z = z ;
			V[k].w = 0;
			
			//printf("satellite %f\t%f\t%f\n",(-ys/rabs)  ,orb, sqrtf(-Q(rabs)) );
		}
	}
  
}

//-----------------------------------------------------------------------------

// запись координат частиц в файл c именем out/iteration_(i).txt
void    WriteData(int i)
{
    char FileName[128];
	char FileName2[128];
	char FileName3[128];
	char FileName4[128];

	
    sprintf(FileName, "D:\\Diplom_Vanya\\CUDA\\Программы\\NbodyGPU\\OutFiles\\disc_%i.dat", i);
	sprintf(FileName2, "D:\\Diplom_Vanya\\CUDA\\Программы\\NbodyGPU\\OutFiles\\HALO_%i.dat", i);
	sprintf(FileName3, "D:\\Diplom_Vanya\\CUDA\\Программы\\NbodyGPU\\OutFiles\\bulge_%i.dat", i);
	sprintf(FileName4, "D:\\Diplom_Vanya\\CUDA\\Программы\\NbodyGPU\\OutFiles\\satellite_%i.dat", i);


    FILE * out = fopen(FileName, "w");
	FILE * out2 = fopen(FileName2, "w");
	FILE * out3 = fopen(FileName3, "w");
	FILE * out4 = fopen(FileName4, "w");
	
    for(int i=0; i<NParticles; i++)
	{
		if(i<NforDisc)
		{
			fprintf(out, "%f\t%f\t%f\n", X[i].x, X[i].y, X[i].z);
		
		}
		else if(i<(NforDisc + NforHalo))
		{
			fprintf(out2,"%f\t%f\t%f\n", X[i].x, X[i].y, X[i].z);
		}
		else if(i <(NforDisc + NforHalo + NforBulge)) 
		{
			fprintf(out3,"%f\t%f\t%f\n", X[i].x, X[i].y, X[i].z);
		}
		else
		{
			fprintf(out4,"%f\t%f\t%f\n", X[i].x, X[i].y, X[i].z);
		}
		
	}
    fclose(out);
	fclose(out2);
	fclose(out3);
	fclose(out4);
	
	

}
//-----------------------------------------------------------------------------
// вычисление новых координат частиц
void    Calculate(float timeStep)
{		
		float4 *devX,*devA,*devV,*newX,*newA,*newV;

		cudaMalloc ( &devX, NParticles * sizeof(float4)); 
		cudaMalloc ( &devA, NParticles * sizeof(float4));
		cudaMalloc ( &devV, NParticles * sizeof(float4));
		cudaMalloc ( &newX, NParticles * sizeof(float4)); 
		cudaMalloc ( &newA, NParticles * sizeof(float4));
		cudaMalloc ( &newV, NParticles * sizeof(float4)); 
		
		cudaMemcpy ( devX, X, NParticles * sizeof(float4), cudaMemcpyHostToDevice ); 
		cudaMemcpy ( devA, A, NParticles * sizeof(float4), cudaMemcpyHostToDevice );
		cudaMemcpy ( devV, V, NParticles * sizeof(float4), cudaMemcpyHostToDevice );
		//printf("Iteration %f,\t%f\t\n", A[1].z,A[1].w);
		//printf("Velocity %f,\t%f\t%f\n", V[1000].x,V[1000].y,V[1000].z);
		// вычисляем новую скорость и позицию
	
		//cudaPrintfInit(25600000);
		
		//system("pause");
		
		calculate_forces<<<dim3((int)NParticles/p),dim3(p)>>> ( devX,  devA, devV, newX, newA, newV, NParticles, p ,timeStep, NforDisc, NforHalo, NforBulge);
        
		

		//cudaPrintfDisplay(stdout, true);
		
		cudaError_t f = cudaThreadSynchronize();
		printf("Velocity %s\n",  cudaGetErrorString( f ));
       // cudaPrintfEnd();
		
		cudaMemcpy ( X, newX, NParticles * sizeof(float4), cudaMemcpyDeviceToHost ); 
		cudaMemcpy ( A, newA, NParticles * sizeof(float4), cudaMemcpyDeviceToHost ); 
		cudaMemcpy ( V, newV, NParticles * sizeof(float4), cudaMemcpyDeviceToHost );
		
		printf("Velocity %f,\t%f\t%f\n", V[1000].x,V[1000].y,V[1000].z);
		printf("Position %f,\t%f\t%f\n", X[1000].x,X[1000].y,X[1000].z);
		float E=0;
		for(int i = 0; i<NParticles; i++)
		{

			E+= A[i].w + V[i].w;



		/*	V[i].x += A[i].x * TimeStep ;
			V[i].y += A[i].y * TimeStep ;
			V[i].z += A[i].z * TimeStep ;
			
			X[i].x += V[i].x * TimeStep;
			X[i].y += V[i].y * TimeStep;
			X[i].z += V[i].z * TimeStep;
		*/
		}
		//printf("Position %f\r", E);
	//	printf("Velocity %f,\t%f\t%f\n", V[1000].x,V[1000].y,V[1000].z);
		// system("pause");
		//printf("\r");
		cudaFree (devX);
		cudaFree (devA);
		cudaFree (devV);
		cudaFree (newX);
		cudaFree (newA);
		cudaFree (newV);
	
    
}

//-----------------------------------------------------------------------------





//-----------------------------------------------------------------------------

int  main(int argc, char *argv[])
{
	FILE *file;

	file = fopen("D:\\Diplom_Vanya\\CUDA\\Программы\\NbodyGPU\\initialparam.txt","r");

	int t=0;
	float input[8];
	char result_sting[20];
 while(fgets(result_sting,sizeof(result_sting),file))
        {

            input[t]=atof ( result_sting  );                 
            t++;   
        }
 
        fclose(file);

		for (int i=0; i<8; i++)
    {
        
		
		printf(" %f\n", input[i]);
        
    }
	
	//параметры спутника
	/*psi = input[0]*PI/180.0;
	rabs = input[1];
	xs = 0.0;
	zs = rabs*sin(psi);
	ys = rabs*cos(psi);
	orb = input[6];
	r0 = input[2];
	//параметры галактики
	Qt = input[3];
	//if(input[7]== 0.0)NforBulge=0;
	Mb = input[4];
	a =0.05;// input[5];
*/
    InitParticles();
	char FileName[32];
	
	
    sprintf(FileName, "test.txt");
	


    FILE * out = fopen(FileName, "w+");

	for (int i=0; i<1000; i++)
    {
        
		
		fprintf(out," %f\n", Bin[i]);
        
    }
	
    
    fclose(out);
 
	
    WriteData(0);
	float GpuTime;
 
	
	cudaEvent_t start,stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);	

	cudaEventRecord(start,0);
    
	for (int i=0; i<300; i++)
    {
		if(i==0) Calculate(TimeStep/2.0);
        else Calculate(TimeStep);
		

		printf("Iteration %i,\n", i);
        
       
      
        WriteData(i + 1);
    }
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GpuTime,start,stop);

		printf("Gpu Time is: %f\n", GpuTime/1000.0);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	system("pause");
    delete [] X;
	delete [] A;
	delete [] V;


	return 0;
}

