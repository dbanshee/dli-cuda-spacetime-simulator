/*
 * SpaceTime Simulator
 *   Curso Deep Learning y Cuda - 2020
 *   Autor: Oscar Noel Amaya Garcia   
 *   email: dbanshee@gmail.com
 */
 
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>

#define RUN_MODE_SIM            0
#define RUN_MODE_BENCH          1

#define SP_FILENAME             "sp.json"
#define SP_FILENAME_BUFF1       "sp_0.json"
#define SP_FILENAME_BUFF2       "sp_1.json"
#define SP_FILENAME_BENCH       "sp_bench.json"
#define MAX_BLACK_HOLES         10
#define SOFTENING               1e-9f
#define DT                      0.05f

#define MAX_ASTEROIDS           30
#define AST_FILENAME            "ast.json"
#define AST_FILENAME_BUFF1      "ast_0.json"
#define AST_FILENAME_BUFF2      "ast_1.json"
#define AST_FILENAME_BENCH      "ast_bench.json"

#define MAX_BENCHMARKS          128
#define BENCH_FILENAME          "benchmark.json"
#define BENCH_TIME_SECS         10
#define BENCH_CPU               0
#define BENCH_GPU               1
#define BENCH_REGEN_BH_STEPS    5
#define BENCH_FILE_ACCESS_STEPS 3

#define CUDA_OPT_NLEVELS        4
#define CUDA_OPT_LEVEL_0        0
#define CUDA_OPT_LEVEL_1        1
#define CUDA_OPT_LEVEL_2        2
#define CUDA_OPT_LEVEL_3        3

#define MAX_TIME_SIMULATION_SEC 360
#define REGEN_BLACK_HOLES_SEC   20

#define NUM_BECHMARKS 10

typedef struct blackHole {
    float x, y, g;
} blackHole;

typedef struct spacePoint {
    float x, y, g;
} spacePoint;

typedef struct asteroid { 
    float x, y, vx, vy; 
} asteroid;


typedef struct benchmark {
    char name[1024];
    int number;
    int config;
    int type;   // CPU = 0, GPU = 1
    long time;  // millis
    int steps;
} benchmark;


///////////////
// Global Vars
///////////////

// Runtime
int runMode = RUN_MODE_SIM;
int spCurrentBuff = 0; 
int astCurrentBuff = 0; 
int nBlackHoles = 0;
int nAsteroids = MAX_ASTEROIDS;
blackHole* blackHoles = NULL;
int bhSize;
asteroid* asteroids = NULL;
int astSize;
spacePoint* SPBox = NULL;
int spSize;
float top = 2, left = -2, bottom = -2, right = 2;
float spStep = 0.1;
int nelems;
int rows, cols;
int cudaOptLevel = CUDA_OPT_LEVEL_3;

// BenchMark
int nBenchmark;
int bechmarkRegenBHSteps = MAX_TIME_SIMULATION_SEC;
int bechmarkRegenWriteFileSteps = REGEN_BLACK_HOLES_SEC;
benchmark BENCHS[MAX_BENCHMARKS];
char benchName[1024];
int benchNum;
int benchConfig;
int benchType;


//////////////////
// Error Handling
//////////////////

void checkCudaError(cudaError_t err) {
  if (err != cudaSuccess) {
    printf("Error: %s", cudaGetErrorString(err));
    exit(-1);
  }
}

/////////////////
// Miscellaneous
/////////////////

long currentTimeMillis() {
  struct timeval time;
  gettimeofday(&time, NULL);
  long s1 = (int64_t)(time.tv_sec) * 1000;
  long s2 = (time.tv_usec / 1000);
  return s1 + s2;
}

void initRandom() {
    srand(time(NULL));
}

float getRandom() {
    return (double) rand() / (double)RAND_MAX ;
}

float getRandonLimits(float min, float max) {
    float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
    return min + scale * ( max - min );      /* [min, max] */
}

int nsleep(long miliseconds) {
   struct timespec req, rem;

   if (miliseconds > 999) {   
        req.tv_sec = (int)(miliseconds / 1000);
        req.tv_nsec = (miliseconds - ((long)req.tv_sec * 1000)) * 1000000;
   } else {   
        req.tv_sec = 0;
        req.tv_nsec = miliseconds * 1000000;
   }   

   return nanosleep(&req , &rem);
}

/////////////
// File Dump
/////////////

void spaceTimeToFile(spacePoint* SPBox, int nRows, int nCols) {

    char filename[16];
    
    if (runMode != RUN_MODE_SIM) {
        strcpy(filename, SP_FILENAME_BENCH);
    } else {
        if (spCurrentBuff == 0) {
            strcpy(filename, SP_FILENAME_BUFF1);
            spCurrentBuff = 1;
        } else {
            strcpy(filename, SP_FILENAME_BUFF2);
            spCurrentBuff = 0;
        }
    }
    
    FILE *fp;
    fp = fopen(filename, "w+");
    
    fputs("{\"space\":[", fp);
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            int idx = i*nRows+j; 
            fprintf(fp, "[%0.2f,%0.2f,%0.2f]", SPBox[idx].x, SPBox[idx].y, SPBox[idx].g);
            
            if (i+1 < nRows || j+1 < nCols) {
                fprintf(fp, ",");
            }
        }
    }    
    fputs("]", fp);
    
    fputs(",\"blackHoles\":[", fp);
    for (int i = 0; i < nBlackHoles; i++) {
        fprintf(fp, "[%0.2f,%0.2f,%0.2f]", blackHoles[i].x, blackHoles[i].y, blackHoles[i].g);
        if (i+1 < nBlackHoles) {
            fprintf(fp, ",");
        }
    }
    fputs("]}", fp);
    fclose(fp);
    
    if (runMode == RUN_MODE_SIM) {
        if (access(SP_FILENAME, F_OK) != -1) {
            remove(SP_FILENAME);
        }
        symlink(filename, SP_FILENAME);
    }
}

void asteroidsToFile(asteroid* AST, int nAsteroids) {

    char filename[16];
    if (runMode != RUN_MODE_SIM) {
        strcpy(filename, AST_FILENAME_BENCH);
    } else {
        if (astCurrentBuff == 0) {
            strcpy(filename, AST_FILENAME_BUFF1);
            astCurrentBuff = 1;
        } else {
            strcpy(filename, AST_FILENAME_BUFF2);
            astCurrentBuff = 0;
        }
    }
    
    FILE *fp;
    fp = fopen(filename, "w+");
    
    fputs("{\"asteroids\":[", fp);
    for (int i = 0; i < nAsteroids; i++) { 
        fprintf(fp, "[%0.2f,%0.2f]", AST[i].x, AST[i].y);
            
        if (i+1 < nAsteroids) {
           fprintf(fp, ",");
        }
        
    }    
    fputs("]}", fp);
    fclose(fp);
    
    if (runMode == RUN_MODE_SIM) {
        if (access(AST_FILENAME, F_OK) != -1) {
            remove(AST_FILENAME);
        }
        symlink(filename, AST_FILENAME);
    }
}

void benchMarksToFile(benchmark* BENCHS, int nBenchMarks) {
    printf("Writing benchmark to File: %s\n", BENCH_FILENAME);
    fflush(stdout);
    
    FILE *fp;
    fp = fopen(BENCH_FILENAME, "w+");
    
    fputs("{\"benchmarks\":[", fp);
    for (int i = 0; i < nBenchMarks; i++) { 
        fputs("{", fp);
        fprintf(fp, "\"name\": \"%s\",", BENCHS[i].name);
        fprintf(fp, "\"number\": %d,",   BENCHS[i].number);
        fprintf(fp, "\"type\": %d,",     BENCHS[i].type);
        fprintf(fp, "\"config\": %d,",   BENCHS[i].config);
        fprintf(fp, "\"time\": %ld,",    BENCHS[i].time);
        fprintf(fp, "\"steps\": %d",     BENCHS[i].steps);
        fputs("}", fp);
        
        if (i+1 < nBenchMarks) {
           fprintf(fp, ",");
        }
    }    
    fputs("]}", fp);
    fclose(fp);
}

///////////////
// CPU Runtime
///////////////

void freeBlackHoles() {
    if (blackHoles != NULL) {
        cudaFree(blackHoles);
        blackHoles = NULL;
    }
}

void initBlackHoles() { 
    freeBlackHoles();
    
    bhSize = nBlackHoles*sizeof(blackHole);
    checkCudaError(cudaMallocManaged(&blackHoles, bhSize));
    
    for (int i = 0 ; i < nBlackHoles ; i++) {
        blackHoles[i].x = getRandonLimits(left, right);
        blackHoles[i].y = getRandonLimits(bottom, top);
        blackHoles[i].g = 500.0f;
    }
}

void generateBlackHoles() { 
    nBlackHoles = (int) getRandonLimits(1, MAX_BLACK_HOLES) % MAX_BLACK_HOLES;
    initBlackHoles();
}

void initSpaceTime() {
    rows = abs(right-left) / spStep;
    cols = abs(top-bottom) / spStep;
    nelems = cols*rows;

    spSize = sizeof(spacePoint) * nelems;
    checkCudaError(cudaMallocManaged(&SPBox, spSize));
    memset(SPBox, 0, spSize);
}

void freeSpaceTime() {
    if (SPBox != NULL) {
        cudaFree(SPBox);
        SPBox = NULL;
    }
}

void calculateSpaceTime(spacePoint* SPBox, int nRows, int nCols) { 
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            float x = (i / (float) nRows * (right-left)) + left;
            float y = (j / (float) nCols * (top-bottom)) + bottom;
            
            int idx = i*nRows+j;
            
            float g = 0.0f;
            for (int b = 0; b < nBlackHoles; b++){
                float dx = x - blackHoles[b].x;
                float dy = y - blackHoles[b].y;
                float distSqr = sqrt(dx*dx + dy*dy);
                if (distSqr == 0.0f) {
                    distSqr = 0.000000001;
                }  
                
                float invDist = 1 / (pow(distSqr, 0.05));
                g += (blackHoles[b].g * invDist);                                
            }
            SPBox[idx].x = x;
            SPBox[idx].y = y;
            SPBox[idx].g = g;
        }
    }
}

void freeAsteroids() {
    if (asteroids != NULL) {
        cudaFree(asteroids);
        asteroids = NULL;
    }
}

void initAsteroids() {
    freeAsteroids();
    astSize = nAsteroids*sizeof(asteroid);
    checkCudaError(cudaMallocManaged(&asteroids, astSize));
    memset(asteroids, 0, astSize);

    for (int i = 0; i < nAsteroids; i++) {
        asteroids[i].x = getRandonLimits(left, right);
        asteroids[i].y = getRandonLimits(bottom, top); 
    }
}

void calculateAsteroids(asteroid* AST, int nAsteroids, blackHole* BH, int nBlackHoles, float dt) {
    for (int i = 0; i < nAsteroids; i++) {
        float Fx = 0.0f; float Fy = 0.0f;

        for (int j = 0; j < nBlackHoles; j++) {
            float dx = AST[i].x - BH[j].x;
            float dy = AST[i].y - BH[j].y;
            
            if (dx == 0) break;
            if (dy == 0) break;
            
            float distSqr = dx*dx + dy*dy;
            //float invDist = rsqrtf(distSqr);
            float invDist = 1 / sqrt(distSqr*distSqr);

            Fx += dx * invDist; Fy += dy * invDist;
        }
        
        float modF = sqrt(Fx*Fx + Fy*Fy);
        Fx = Fx / modF;
        Fy = Fy / modF;
        
        AST[i].vx += dt*-Fx; 
        AST[i].vy += dt*-Fy;
        AST[i].x  += AST[i].vx*0.01;
        AST[i].y  += AST[i].vy*0.01;
    }
}

void runSimulationCPU() {
    printf("Running Simulation CPU ...\n");
    printf("SpaceTime Params: rows: %d, cols: %d, size: %d\n", rows, cols, nelems);
    
    long initSimulation = currentTimeMillis();
    int elapsedSimulation = 0;
    long lastBlackHoleRegen = 0;
    
    initSpaceTime();
    initAsteroids();
    
    while (elapsedSimulation < MAX_TIME_SIMULATION_SEC) {   
        
        long current = currentTimeMillis();
        
        if (blackHoles == NULL || (current - lastBlackHoleRegen) / 1000 > REGEN_BLACK_HOLES_SEC) {
            generateBlackHoles();
            calculateSpaceTime(SPBox, rows, cols);
            spaceTimeToFile(SPBox, rows, cols);
            lastBlackHoleRegen = current;
        }
    
        calculateAsteroids(asteroids, nAsteroids, blackHoles, nBlackHoles, DT);
        asteroidsToFile(asteroids, nAsteroids);
        
        nsleep(250);
        elapsedSimulation = (currentTimeMillis() - initSimulation) / 1000;
    }
    
    freeAsteroids();
}

////////////////////
// GPU CUDA Runtime
////////////////////

__global__ void calculateSpaceTimeK(spacePoint* SPBox, int nRows, int nCols, float left, float right, float top, float bottom, blackHole* BH, int nBlackHoles) { 
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i < nRows && j < nCols) {
        float x = (i / (float) nRows * (right-left)) + left;
        float y = (j / (float) nCols * (top-bottom)) + bottom;
            
        int idx = i*nRows+j;
            
        float g = 0.0f;
        for (int b = 0; b < nBlackHoles; b++){
            float dx = x - BH[b].x;
            float dy = y - BH[b].y;
            float distSqr = sqrt(dx*dx + dy*dy);
            if (distSqr == 0.0f) {
                distSqr = 0.000000001;
            }  
                
            float invDist = 1 / (pow((float)distSqr, (float)0.05));
            g += (BH[b].g * invDist);                                
        }
        
        SPBox[idx].x = x;
        SPBox[idx].y = y;
        SPBox[idx].g = g;
    }
}

__global__ void calculateSpaceTimeKSHM(spacePoint* SPBox, int nRows, int nCols, float left, float right, float top, float bottom, blackHole* BH, int nBlackHoles, int bhSize) { 

    extern __shared__ float s[];
    blackHole* bhCache = (blackHole*) s;

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i == 0 && j == 0)
        memcpy(bhCache, BH, bhSize);
    __syncthreads();
    
    if (i < nRows && j < nCols) {
        float x = (i / (float) nRows * (right-left)) + left;
        float y = (j / (float) nCols * (top-bottom)) + bottom;
            
        int idx = i*nRows+j;
            
        float g = 0.0f;
        for (int b = 0; b < nBlackHoles; b++){
            float dx = x - bhCache[b].x;
            float dy = y - bhCache[b].y;
            float distSqr = sqrt(dx*dx + dy*dy);
            if (distSqr == 0.0f) {
                distSqr = 0.000000001;
            }  
                
            float invDist = 1 / (pow((float)distSqr, (float)0.05));
            g += (bhCache[b].g * invDist);                                
        }
        
        SPBox[idx].x = x;
        SPBox[idx].y = y;
        SPBox[idx].g = g;
    }
}

__global__ void calculateAsteroidsK(asteroid* AST, int nAsteroids, blackHole* BH, int nBlackHoles, float dt) {

    int idx    = (blockIdx.x * blockDim.x) + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < nAsteroids; i += stride) {
        float Fx = 0.0f; float Fy = 0.0f;

        for (int j = 0; j < nBlackHoles; j++) {
            float dx = AST[i].x - BH[j].x;
            float dy = AST[i].y - BH[j].y;
            
            if (dx == 0) break;
            if (dy == 0) break;
            
            float distSqr = dx*dx + dy*dy; //+ SOFTENING;
            //float invDist = rsqrtf(distSqr);
            float invDist = 1 / sqrt(distSqr*distSqr);
            //float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist; Fy += dy * invDist;
        }
        
        float modF = sqrt(Fx*Fx + Fy*Fy);
        Fx = Fx / modF;
        Fy = Fy / modF;
        
        AST[i].vx += dt*-Fx; 
        AST[i].vy += dt*-Fy;
        AST[i].x  += AST[i].vx*0.01;
        AST[i].y  += AST[i].vy*0.01;
    }
}

__global__ void calculateAsteroidsKSHM(asteroid* AST, int nAsteroids, blackHole* BH, int nBlackHoles, float dt, int bhSize) {

    extern __shared__ float s[];
    blackHole* bhCache = (blackHole*) s;
    
    int idx    = (blockIdx.x * blockDim.x) + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    if (idx == 0) 
        memcpy(bhCache, BH, bhSize);
    __syncthreads();

    for (int i = idx; i < nAsteroids; i += stride) {
        float Fx = 0.0f; float Fy = 0.0f;

        for (int j = 0; j < nBlackHoles; j++) {
            float dx = AST[i].x - bhCache[j].x;
            float dy = AST[i].y - bhCache[j].y;
            
            if (dx == 0) break;
            if (dy == 0) break;
            
            float distSqr = dx*dx + dy*dy;
            float invDist = 1 / sqrt(distSqr*distSqr);

            Fx += dx * invDist; Fy += dy * invDist;
        }
        
        float modF = sqrt(Fx*Fx + Fy*Fy);
        Fx = Fx / modF;
        Fy = Fy / modF;
        
        AST[i].vx += dt*-Fx; 
        AST[i].vy += dt*-Fy;
        AST[i].x  += AST[i].vx*0.01;
        AST[i].y  += AST[i].vy*0.01;
    }
}

void runSimulationGPU() {
    printf("Running Simulation GPU ...\n");
    
    long initSimulation = currentTimeMillis();
    int elapsedSimulation = 0;
    long lastBlackHoleRegen = 0;
    
    int multiProcessorCount = 32;
    initSpaceTime();
    initAsteroids();
    
    printf("SpaceTime Params: rows: %d, cols: %d, size: %d\n", rows, cols, nelems);
    
    dim3 threads_per_block_K1(32, 32, 1);
    dim3 total_blocks_K1((rows / threads_per_block_K1.x) + 1, (cols / threads_per_block_K1.y) + 1, 1);
        
    int threads_per_block_K2 = 1024;
    int number_of_blocks_K2 = ceil(((nAsteroids / (float) threads_per_block_K2) + 1) / multiProcessorCount) * multiProcessorCount;
    
    
    while (elapsedSimulation < MAX_TIME_SIMULATION_SEC) {   
        
        long current = currentTimeMillis();
        
        if (blackHoles == NULL || (current - lastBlackHoleRegen) / 1000 > REGEN_BLACK_HOLES_SEC) {
            generateBlackHoles();
            
            calculateSpaceTimeK<<<total_blocks_K1, threads_per_block_K1>>>(SPBox, rows, cols, left, right, top, bottom, blackHoles, nBlackHoles);
            checkCudaError(cudaGetLastError());
            calculateAsteroidsK<<<number_of_blocks_K2, threads_per_block_K2>>>(asteroids, nAsteroids, blackHoles, nBlackHoles, DT);
            checkCudaError(cudaGetLastError());
            checkCudaError(cudaDeviceSynchronize());
            
            spaceTimeToFile(SPBox, rows, cols);
            
            lastBlackHoleRegen = current;
        }
    
        calculateAsteroidsK<<<number_of_blocks_K2, threads_per_block_K2>>>(asteroids, nAsteroids, blackHoles, nBlackHoles, DT);
        checkCudaError(cudaGetLastError());
        
        
        checkCudaError(cudaDeviceSynchronize());
        asteroidsToFile(asteroids, nAsteroids);
        
        nsleep(250);
        elapsedSimulation = (currentTimeMillis() - initSimulation) / 1000;
    }
    
    freeAsteroids();
}


//////////////
// Benchmarks
//////////////

void recordBenchmark(int i, char* name, int num, int type, int config, long time, int steps) {
    if (i < MAX_BENCHMARKS) {
        strcpy(BENCHS[i].name, name);
        BENCHS[i].number = num;
        BENCHS[i].type = type;
        BENCHS[i].config = config;
        BENCHS[i].time = time;
        BENCHS[i].steps = steps;
    }
}

void accessSPBenchCPU(spacePoint* SPBox, int nRows, int nCols) {
    float x, y, g;

    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            int idx = i*nRows+j; 
            x = SPBox[idx].x; y = SPBox[idx].y; g = SPBox[idx].g;
            x++; y++; g++;
        }
    }
}

void accessBHBenchCPU(){
    float x, y, g;
    for (int i = 0; i < nBlackHoles; i++) {
        x = blackHoles[i].x; y = blackHoles[i].y; g = blackHoles[i].g;
        x++; y++; g++;
    }
}

void accessAstBenchCPU(asteroid* AST, int nAsteroids){
    float x, y;
    for (int i = 0; i < nAsteroids; i++) { 
        x = AST[i].x; y = AST[i].y;
        x++; y++; 
    }    
}

void runBenchMarkCPU() {
    printf("\nBenchmark CPU %d_%d - Config %d\n", benchNum, benchConfig, benchConfig);
    printf("  top: %0.2f, bottom: %0.2f, left: %0.2f, right: %0.2f\n", top, bottom, left, right);
    printf("  spStep : %f, nBlackHoles :%d, nAsteroids: %d\n", spStep, nBlackHoles, nAsteroids);
    printf("  Matrix Space Resolution [%d, %d]\n", (int) (abs(right-left) / spStep), (int) (abs(top-bottom) / spStep));
    fflush(stdout);
    
    
    initSpaceTime();
    initAsteroids();
    initBlackHoles();
    
    long startTime = currentTimeMillis();
    int step = 0;    
    while (((currentTimeMillis() - startTime) / 1000) < BENCH_TIME_SECS) {
        if (step % bechmarkRegenBHSteps == 0) {
            initBlackHoles();
        }
        
        if (step % bechmarkRegenWriteFileSteps == 0) {
            accessSPBenchCPU(SPBox, rows, cols);
            accessAstBenchCPU(asteroids, nAsteroids);
        }
        
        calculateSpaceTime(SPBox, rows, cols);
        calculateAsteroids(asteroids, nAsteroids, blackHoles, nBlackHoles, DT);
        step++;
    }
    
    accessSPBenchCPU(SPBox, rows, cols);
    accessAstBenchCPU(asteroids, nAsteroids);
    
    long elapsedTime = currentTimeMillis() - startTime;
    
    freeSpaceTime();
    freeAsteroids();
    freeBlackHoles();
    
    sprintf(benchName, "Benchmark %d_%d", benchNum, benchConfig);
    recordBenchmark(nBenchmark, benchName, benchNum, benchType, benchConfig, elapsedTime, step);
    printf("\n  Benchmark %d - Time: %ld millis, steps : %d\n", nBenchmark, elapsedTime, step);
    printf("    Steps Per Second : %ld\n\n", step / (elapsedTime / 1000));
}

void runBenchMarkGPU() {

    printf("\n Benchmark GPU %d_%d - Config %d\n", benchNum, benchConfig, benchConfig);
    printf("  top: %0.2f, bottom: %0.2f, left: %0.2f, right: %0.2f\n", top, bottom, left, right);
    printf("  spStep: %f, nBlackHoles: %d, nAsteroids: %d\n", spStep, nBlackHoles, nAsteroids);
    printf("  Matrix Space Resolution [%d, %d]\n", (int) (abs(right-left) / spStep), (int) (abs(top-bottom) / spStep));
    fflush(stdout);
    
    int deviceId;
    int multiProcessorCount = 32;
    int computePreemptionSupported;
    cudaDeviceProp deviceProp;
    
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&deviceProp, deviceId);
    
    multiProcessorCount = deviceProp.multiProcessorCount;
    computePreemptionSupported = deviceProp.computePreemptionSupported;
    printf("  Device supports Compute Preemption: %s\n", computePreemptionSupported ? "Yes" : "No");

    initSpaceTime();
    initAsteroids();
    initBlackHoles();
    

    dim3 threads_per_block_K1(32, 32, 1);
    dim3 total_blocks_K1((rows / threads_per_block_K1.x) + 1, (cols / threads_per_block_K1.y) + 1, 1);
    //printf("    K1 -> Threads Per Block: [%d,%d], Total BLocks: [%d, %d]\n", threads_per_block_K1.x, threads_per_block_K1.y, total_blocks_K1.x, total_blocks_K1.y);
        
    //int threads_per_block_K2 = 64;
    int threads_per_block_K2 = 1024;
    int number_of_blocks_K2 = ceil(((nAsteroids / (float) threads_per_block_K2) + 1) / multiProcessorCount) * multiProcessorCount;
    //printf("    K2 -> Threads Per Block: [%d], Total BLocks: [%d]\n", threads_per_block_K2, number_of_blocks_K2);

    
    long startTime = currentTimeMillis();
    int step = 0;    
    while (((currentTimeMillis() - startTime) / 1000) < BENCH_TIME_SECS) {
        if (step % bechmarkRegenBHSteps == 0) {
            checkCudaError(cudaDeviceSynchronize());
            
            if (cudaOptLevel >= CUDA_OPT_LEVEL_2 && computePreemptionSupported)
                cudaMemPrefetchAsync(blackHoles, bhSize, cudaCpuDeviceId);
            
            initBlackHoles();
            
            if (cudaOptLevel >= CUDA_OPT_LEVEL_2 && computePreemptionSupported)
                cudaMemPrefetchAsync(blackHoles, bhSize, deviceId);
        }
        
        if (step % bechmarkRegenWriteFileSteps == 0) {
            if (cudaOptLevel >= CUDA_OPT_LEVEL_2 && computePreemptionSupported) {
                cudaMemPrefetchAsync(SPBox, spSize, cudaCpuDeviceId);
                cudaMemPrefetchAsync(asteroids, astSize, cudaCpuDeviceId);
            }
        
            //nsleep(150L); // CPU Ocupancy
            checkCudaError(cudaDeviceSynchronize());
            accessSPBenchCPU(SPBox, rows, cols);
            accessAstBenchCPU(asteroids, nAsteroids);
            
            if (cudaOptLevel >= CUDA_OPT_LEVEL_2 && computePreemptionSupported) {
                cudaMemPrefetchAsync(SPBox, spSize, deviceId);
                cudaMemPrefetchAsync(asteroids, astSize, deviceId);
            }
        } 
       
        if (cudaOptLevel >= CUDA_OPT_LEVEL_1) {
            cudaStream_t stream1, stream2;
            checkCudaError(cudaStreamCreate(&stream1));
            checkCudaError(cudaStreamCreate(&stream2));
            
            if (cudaOptLevel >= CUDA_OPT_LEVEL_3) {
                calculateSpaceTimeKSHM<<<total_blocks_K1, threads_per_block_K1, bhSize, stream1>>>(SPBox, rows, cols, left, right, top, bottom, blackHoles, nBlackHoles, bhSize);
                calculateAsteroidsKSHM<<<number_of_blocks_K2, threads_per_block_K2, bhSize, stream2>>>(asteroids, nAsteroids, blackHoles, nBlackHoles, DT, bhSize);
            } else {
                calculateSpaceTimeK<<<total_blocks_K1, threads_per_block_K1, 0, stream1>>>(SPBox, rows, cols, left, right, top, bottom, blackHoles, nBlackHoles);
                calculateAsteroidsK<<<number_of_blocks_K2, threads_per_block_K2, 0, stream2>>>(asteroids, nAsteroids, blackHoles, nBlackHoles, DT);
            }
            
            checkCudaError(cudaGetLastError());
            cudaStreamDestroy(stream1);
            cudaStreamDestroy(stream2);
        } else {
            calculateSpaceTimeK<<<total_blocks_K1, threads_per_block_K1>>>(SPBox, rows, cols, left, right, top, bottom, blackHoles, nBlackHoles);
            checkCudaError(cudaGetLastError());
            calculateAsteroidsK<<<number_of_blocks_K2, threads_per_block_K2>>>(asteroids, nAsteroids, blackHoles, nBlackHoles, DT);
            checkCudaError(cudaGetLastError());
        }
        
        step++;
    }
    
    if (cudaOptLevel >= CUDA_OPT_LEVEL_2 && computePreemptionSupported) {
        cudaMemPrefetchAsync(SPBox, spSize, cudaCpuDeviceId);
        cudaMemPrefetchAsync(asteroids, astSize, cudaCpuDeviceId);
    }
    
    checkCudaError(cudaDeviceSynchronize());
    accessSPBenchCPU(SPBox, rows, cols);
    accessAstBenchCPU(asteroids, nAsteroids);

    long elapsedTime = currentTimeMillis() - startTime + 1;
  
    freeSpaceTime();
    freeAsteroids();
    freeBlackHoles();

    sprintf(benchName, "Benchmark %d_%d", benchNum, benchConfig);
    recordBenchmark(nBenchmark, benchName, benchNum, benchType, benchConfig, elapsedTime, step);
    printf("\n  Benchmark %d - Time: %ld millis, steps : %d\n", nBenchmark, elapsedTime, step);
    printf("    Steps Per Second : %ld\n\n", step / (elapsedTime / 1000));
}

void runBenchmarks() {
    benchType = BENCH_GPU;
    
    for (int opLevel =  0; opLevel < CUDA_OPT_NLEVELS; opLevel++) {
        // Start on BENCH_CPU for include CPU execution. 
        for (int cp = BENCH_GPU ; cp <= BENCH_GPU; cp++) {
            benchType = cp;
            benchConfig = cudaOptLevel = opLevel;

            // Benchmark 0
            benchNum = 0;
            top = 2, left = -2, bottom = -2, right = 2; spStep = 0.01;
            nBlackHoles = 100; nAsteroids = 100;
            if(cp == BENCH_CPU) runBenchMarkCPU();
            else runBenchMarkGPU();
            nBenchmark++;

            // Benchmark 1
            benchNum = 1;
            top = 2, left = -2, bottom = -2, right = 2; spStep = 0.01;
            nBlackHoles = 500; nAsteroids = 500;
            if(cp == BENCH_CPU) runBenchMarkCPU();
            else runBenchMarkGPU();  
            nBenchmark++;
            
            // Benchmark 2
            benchNum = 2;
            top = 2, left = -2, bottom = -2, right = 2; spStep = 0.01;
            nBlackHoles = 1000; nAsteroids = 1000;
            if(cp == BENCH_CPU) runBenchMarkCPU();
            else runBenchMarkGPU();  
            nBenchmark++;
            
            // Benchmark 3
            benchNum = 3;
            top = 2, left = -2, bottom = -2, right = 2; spStep = 0.01;
            nBlackHoles = 1000; nAsteroids = 5000;
            if(cp == BENCH_CPU) runBenchMarkCPU();
            else runBenchMarkGPU();  
            nBenchmark++;

            // Benchmark 4
            benchNum = 4;
            top = 2, left = -2, bottom = -2, right = 2; spStep = 0.001;
            nBlackHoles = 100; nAsteroids = 100;
            if(cp == BENCH_CPU) runBenchMarkCPU();
            else runBenchMarkGPU();
            nBenchmark++;

            // Benchmark 5
            benchNum = 5;
            top = 2, left = -2, bottom = -2, right = 2; spStep = 0.001;
            nBlackHoles = 500; nAsteroids = 500;
            if(cp == BENCH_CPU) runBenchMarkCPU();
            else runBenchMarkGPU();
            nBenchmark++;
            
            // Benchmark 6
            benchNum = 6;
            top = 2, left = -2, bottom = -2, right = 2; spStep = 0.001;
            nBlackHoles = 1000; nAsteroids = 1000;
            if(cp == BENCH_CPU) runBenchMarkCPU();
            else runBenchMarkGPU();
            nBenchmark++;
            
            // Benchmark 7
            /*
            benchNum = 7;
            top = 2, left = -2, bottom = -2, right = 2; spStep = 0.001;
            nBlackHoles = 5000; nAsteroids = 5000;
            if(cp == BENCH_CPU) runBenchMarkCPU();
            else runBenchMarkGPU();
            nBenchmark++;
            */
            
            // Benchmark 8
            benchNum = 8;
            top = 2, left = -2, bottom = -2, right = 2; spStep = 0.0005;
            nBlackHoles = 100; nAsteroids = 100;
            if(cp == BENCH_CPU) runBenchMarkCPU();
            else runBenchMarkGPU();
            nBenchmark++;
            
            /*
            // Benchmark 9
            benchNum = 9;
            top = 2, left = -2, bottom = -2, right = 2; spStep = 0.0005;
            nBlackHoles = 500; nAsteroids = 500;
            if(cp == BENCH_CPU) runBenchMarkCPU();
            else runBenchMarkGPU();
            nBenchmark++;
            */
            
            /*
            // Benchmark 10
            benchNum = 10;
            top = 2, left = -2, bottom = -2, right = 2; spStep = 0.0005;
            nBlackHoles = 1000; nAsteroids = 1000;
            if(cp == BENCH_CPU) runBenchMarkCPU();
            else runBenchMarkGPU();
            nBenchmark++;
            
            // Benchmark 11
            benchNum = 11;
            top = 2, left = -2, bottom = -2, right = 2; spStep = 0.0005;
            nBlackHoles = 1000; nAsteroids = 1000;
            if(cp == BENCH_CPU) runBenchMarkCPU();
            else runBenchMarkGPU();
            nBenchmark++;
            */
        }
    }
    fflush(stdout);
}

void runAllBechmarks() {
    printf("Running All Benchmarks ...\n");
    long startTime = currentTimeMillis();
    nBenchmark = 0;
    runBenchmarks();
    benchMarksToFile(BENCHS, nBenchmark);
    printf("Benchmarks in : %ld seconds\n", (currentTimeMillis() - startTime) / 1000);
}

// Device Query code extracted from https://github.com/NVIDIA/cuda-samples
inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}


void printfGPUDeviceInfo(){
    int deviceId;
    cudaDeviceProp deviceProp;
    int driverVersion = 0, runtimeVersion = 0;
    char msg[256];
    
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&deviceProp, deviceId);
    
    printf("GPU Info\n");
    printf(" Device: \"%s\"\n", deviceProp.name);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
           deviceProp.major, deviceProp.minor);
    snprintf(msg, sizeof(msg),
             "  Total amount of global memory:                 %.0f MBytes "
             "(%llu bytes)\n",
             static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
             (unsigned long long)deviceProp.totalGlobalMem);
    printf("%s", msg);

    printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
           deviceProp.multiProcessorCount,
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
               deviceProp.multiProcessorCount);
    printf(
        "  GPU Max Clock rate:                            %.0f MHz (%0.2f "
        "GHz)\n",
        deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
        printf(
        "  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
        "%d), 3D=(%d, %d, %d)\n",
        deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
        deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
        deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
    printf(
        "  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
        deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
    printf(
        "  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
        "layers\n",
        deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
        deviceProp.maxTexture2DLayered[2]);

    printf("  Total amount of constant memory:               %zu bytes\n",
           deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block:       %zu bytes\n",
           deviceProp.sharedMemPerBlock);
    printf("  Total number of registers available per block: %d\n",
           deviceProp.regsPerBlock);
    printf("  Warp size:                                     %d\n",
           deviceProp.warpSize);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n",
           deviceProp.maxThreadsPerBlock);
    printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    printf("  Maximum memory pitch:                          %zu bytes\n",
           deviceProp.memPitch);
    printf("  Texture alignment:                             %zu bytes\n",
           deviceProp.textureAlignment);
    printf(
        "  Concurrent copy and kernel execution:          %s with %d copy "
        "engine(s)\n",
        (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
    printf("  Run time limit on kernels:                     %s\n",
           deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
    printf("  Integrated GPU sharing Host Memory:            %s\n",
           deviceProp.integrated ? "Yes" : "No");
    printf("  Support host page-locked memory mapping:       %s\n",
           deviceProp.canMapHostMemory ? "Yes" : "No");
    printf("  Alignment requirement for Surfaces:            %s\n",
           deviceProp.surfaceAlignment ? "Yes" : "No");
    printf("  Device has ECC support:                        %s\n",
           deviceProp.ECCEnabled ? "Enabled" : "Disabled");
    printf("  Device supports Unified Addressing (UVA):      %s\n",
           deviceProp.unifiedAddressing ? "Yes" : "No");
    printf("  Device supports Compute Preemption:            %s\n",
           deviceProp.computePreemptionSupported ? "Yes" : "No");
    printf("  Supports Cooperative Kernel Launch:            %s\n",
           deviceProp.cooperativeLaunch ? "Yes" : "No");
    printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
           deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
    printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
           deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

}

int main(const int argc, const char** argv) {
    runMode = RUN_MODE_SIM; // RUN_MODE_SIM | RUN_MODE_BENCH

    printf("Cuda Space Time Simulator - 2020\n\n");
    printf("  Author: Oscar Noel Amaya Garcia. email: dbanshee@gmail.com\n");
    printf("  nVidia DLI - Volunteer practice\n");
    printf("\n\n");
    
    printfGPUDeviceInfo();
    printf("\n\n");
    
    if (argc > 2) {
        printf("Invalid args.\n");
        printf("cudaspsim [benchmark]\n");
        exit(-1);
    } else if(argc == 2 && strcmp(argv[1], "benchmark") == 0) {
        runMode = RUN_MODE_BENCH;
    }
    
    if (runMode == RUN_MODE_SIM) {
        runSimulationGPU();
    } else {
        runAllBechmarks();
        printf("\n");
    }
}

