#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <mpi.h>
#define MAXN 2000// Maximum Dimension for Matrix 

void initializeMat();
void backSubstitution();
void displayMat();
void printAnswer();
void gaussian_mpi(int N);

int proc,id,N;
double X[MAXN][MAXN],y[MAXN],Z[MAXN];

/* Initialize matrix with random values */
void initializeMat()
{
	int i,j;
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
						
			X[i][j]=rand()/50000.0;
		}
	y[i] = rand()/50000.0;

	}
}

/* Displays the matrix which has being initialized */
void displayMat()
{	
	int i,j;
	printf("Displaying Initial Matrix.\n");
	for (i=0;i<N;i++)
	{
		printf("| ");
		for(j=0;j<N; j++)
		{	
			printf("%lf ",X[i][j]);
		}
		printf("| | %lf |\n",y[i]);
	}
}

/* This function performs the backsubstitution */
void backSubstitution()
{
	int i,j;
	for (i=N-1;i>=0;i--)
	{
		int count=0;
		Z[i] = y[i];
		for (j=i+1;j<N;j++)
		{
			Z[i]-=X[i][j]*Z[j];
			
		}
		Z[i] = Z[i]/X[i][i];
	}
}

/* This function performs gaussian elimination with MPI implementation through static interleave */
void gaussian_mpi(int N)
{	
	double wp_time,wa_time=0;
	MPI_Request request;
	MPI_Status status;
	int p,k,i,j;
	float mp;	

	MPI_Barrier(MPI_COMM_WORLD);// waiting for all processors	
	if(id==0)// Processors starts the MPI Timer i.e MPI_Wtime()
	{
		wa_time = MPI_Wtime();
	}	

	for (k=0;k<N-1;k++)
 	{	
		//Broadcsting X's and Y's matrix from 0th rank processor to all other processors.
		MPI_Bcast(&X[k][0],N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&y[k],1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
		if(id==0)
		{
			for (p=1;p<proc;p++)
			{
		  		for (i=k+1+p;i<N;i+=proc)
		  		{
				/* Sending X and y matrix from oth to all other processors using non blocking send*/
				   MPI_Isend(&X[i], N, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &request);
				   MPI_Wait(&request, &status);
				   MPI_Isend(&y[i], 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &request);
				   MPI_Wait(&request, &status);
		  		}
			}
			// implementing gaussian elimination 
			for (i=k+1 ; i<N ; i += proc)
			{
	  			mp = X[i][k] / X[k][k];
	  			for (j = k; j < N; j++)
	 			{
	   				X[i][j] -= X[k][j] * mp;
	 			}
	   			y[i] -= y[k] * mp;
			}
			// Receiving all the values that are send by 0th processor.
			for (p = 1; p < proc; p++)
			{
			  for (i = k + 1 + p; i < N; i += proc)
			  {
			    MPI_Recv(&X[i], N, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, &status);
			    MPI_Recv(&y[i], 1, MPI_DOUBLE, p, 1, MPI_COMM_WORLD, &status);
			  }
			}
			//Stopping the MPI_Timer
			if (k == N - 2)
			{
 				wp_time = MPI_Wtime();
 				printf("elapsed time = %f\n", wp_time-wa_time);
			}
		}
		
		
		else
		{
			for (i = k + 1 + id; i < N; i += proc)
			{
				MPI_Recv(&X[i], N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);		
				MPI_Recv(&y[i], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
				mp = X[i][k] / X[k][k];
				for (j = k; j < N; j++)
				{
				    X[i][j] -= X[k][j] * mp;
				}
				y[i] -= y[k] * mp;
				MPI_Isend(&X[i], N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &request);						    
				MPI_Wait(&request, &status);		
				MPI_Isend(&y[i], 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &request);
				MPI_Wait(&request, &status);
			}
		}
		 MPI_Barrier(MPI_COMM_WORLD);//Waiting for all processors
	}
}


/*  Printing Solution Matrix Z   */
void printAnswer()
{
	int i;
	printf("\nSolution Vector (x):\n\n");
	for (i=0;i<N;i++)
	{
		printf("|%lf|\n", Z[i]);
	}	
}

int main(int argc,char *argv[])
{
	
	MPI_Init(&argc,&argv);//Initiating MPI
	MPI_Comm_rank(MPI_COMM_WORLD,&id);//Getting rank of current processor.
	MPI_Comm_size(MPI_COMM_WORLD,&proc);//Getting number of processor in MPI_COMM_WORLD
		
	
	if (argc >= 2) 
	{
	    N = atoi(argv[1]);//getting matrix dimension from command line argument
   	}


	struct timeval etstart, etstop;  
 	struct timezone tzdummy;
 	clock_t etstart2, etstop2;  
  	unsigned long long usecstart, usecstop;
  	struct tms cputstart, cputstop;  


	if(id==0)
	{
		initializeMat();//initializing matrix
	//	displayMat();//displaying the matrix
		/* Start Clock */
 		printf("\nStarting clock.\n");
  		gettimeofday(&etstart, &tzdummy);
  		etstart2 = times(&cputstart);					
	}
	
	gaussian_mpi(N);//implementing the gaussian elimination
	
	if(id==0)
	{
		backSubstitution();
		 /* Stop Clock */
  		gettimeofday(&etstop, &tzdummy);
  		etstop2 = times(&cputstop);
  		printf("Stopped clock.\n");
  		usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  		usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;
	
	//	printAnswer(); Displaying the final solution matrix Z
		
		 /* Display timing results */
  		printf("\nElapsed time = %g ms.\n",
	 	(float)(usecstop - usecstart)/(float)1000);

  		printf("(CPU times are accurate to the nearest %g ms)\n",
	 	1.0/(float)CLOCKS_PER_SEC * 1000.0);
	
  		printf("My total CPU time for parent = %g ms.\n",
	 	(float)( (cputstop.tms_utime + cputstop.tms_stime) -
		  (cputstart.tms_utime + cputstart.tms_stime) ) /
	 	(float)CLOCKS_PER_SEC * 1000);
  	
		printf("My system CPU time for parent = %g ms.\n",
	 	(float)(cputstop.tms_stime - cputstart.tms_stime) /
	 	(float)CLOCKS_PER_SEC * 1000);
  	
		printf("My total CPU time for child processes = %g ms.\n",
	 	(float)( (cputstop.tms_cutime + cputstop.tms_cstime) -
		  (cputstart.tms_cutime + cputstart.tms_cstime) ) /
	 	(float)CLOCKS_PER_SEC * 1000);
     	
		  /* Contrary to the man pages, this appears not to include the parent */
		 printf("--------------------------------------------\n");



	}
	MPI_Finalize(); //Finalizing the MPI
  	return 0;
}
