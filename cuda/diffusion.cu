//kernel definition
__global__ void diffusionSolver(double* A, double * old,int n_x,int n_y)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i*(n_x-i-1)*j*(n_y-j-1)!=0)
        A[i+n_y*j] = A[i+n_y*j] + (old[i-1+n_y*j]+old[i+1+n_y*j]+
                       old[i+(j-1)*n_y]+old[i+(j+1)*n_y] -4*old[i+n_y*j])/40;


}

int main()
{


    int i,j ,M;
    M = n_y ;
    phi = (double *) malloc( n_x*n_y* sizeof(double));
    phi_old = (double *) malloc( n_x*n_y* sizeof(double));
    dummy = (double *) malloc( n_x*n_y* sizeof(double));
    int iterationMax =10;
    //phase initialization
    for(j=0;j<n_y ;j++)
    {
        for(i=0;i<n_x;i++)
        {
            if((.4*n_x-i)*(.6*n_x-i)<0)
                phi[i+M*j] = -1;
            else
                phi[i+M*j] = 1;

            phi_old[i+M*j] = phi[i+M*j];
        }
    }

    double *dev_phi;
    cudaMalloc((void **) &dev_phi, n_x*n_y*sizeof(double));

    dim3 threadsPerBlock(100,10);
    dim3 numBlocks(n_x*n_y / threadsPerBlock.x, n_x*n_y / threadsPerBlock.y);

    //start iterating
    for(int z=0; z<iterationMax; z++)
    {
        //copy array on host to device
        cudaMemcpy(dev_phi, phi, n_x*n_y*sizeof(double),
                cudaMemcpyHostToDevice);

        //call kernel
        diffusionSolver<<<numBlocks, threadsPerBlock>>>(dev_phi, phi_old,n_x,n_y);

        //get updated array back on host
        cudaMemcpy(phi, dev_phi,n_x*n_y*sizeof(double), cudaMemcpyDeviceToHost);

        //old values will be assigned new values
        for(j=0;j<n_y ;j++)
        {
            for(i=0;i<n_x;i++)
            {
                phi_old[i+n_y*j] = phi[i+n_y*j];
            }
        }
    }

    return 0;
}
