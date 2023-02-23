// #include <bang.h>
#include "cnrt.h"
#include <iostream>
#include <math.h>
#define EPS 1e-7
#define LEN 1024
void hostKernel(float* dst, float* source1, float* source2,cnrtDim3_t dim, cnrtFunctionType_t type, cnrtQueue_t queue) ;
int main(void) {
    unsigned int count = 0;
    cnrtGetDeviceCount(&count);
    cnrtSetDevice(0);

    cnrtQueue_t queue;
    cnrtQueueCreate(&queue);

    cnrtDim3_t dim = {1, 1, 1};
    cnrtFunctionType_t ktype = CNRT_FUNC_TYPE_BLOCK;

    cnrtNotifier_t start, end;
    CNRT_CHECK(cnrtNotifierCreate(&start));
    CNRT_CHECK(cnrtNotifierCreate(&end));

    float* host_dst = (float*)malloc(LEN * sizeof(float));
    float* host_src1 = (float*)malloc(LEN * sizeof(float));
    float* host_src2 = (float*)malloc(LEN * sizeof(float));

    for (int i = 0; i < LEN; i++) {
        host_src1[i] = i;
        host_src2[i] = i;
    }

    float* mlu_dst;
    float* mlu_src1;
    float* mlu_src2;
    CNRT_CHECK(cnrtMalloc((void**)&mlu_dst, LEN * sizeof(float)));
    CNRT_CHECK(cnrtMalloc((void**)&mlu_src1, LEN * sizeof(float)));
    CNRT_CHECK(cnrtMalloc((void**)&mlu_src2, LEN * sizeof(float)));

    CNRT_CHECK(cnrtMemcpy(mlu_src1, host_src1, LEN * sizeof(float), CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_CHECK(cnrtMemcpy(mlu_src2, host_src2, LEN * sizeof(float), CNRT_MEM_TRANS_DIR_HOST2DEV));

    CNRT_CHECK(cnrtPlaceNotifier(start, queue));
    hostKernel(mlu_dst, mlu_src1, mlu_src2, dim, ktype, queue);
    // Kernel<<<dim, ktype, queue>>>(mlu_dst, mlu_src1, mlu_src2);

    CNRT_CHECK(cnrtPlaceNotifier(end, queue));

    cnrtQueueSync(queue);
    CNRT_CHECK(cnrtMemcpy(host_dst, mlu_dst, LEN * sizeof(float), CNRT_MEM_TRANS_DIR_DEV2HOST));
    
    for (int i = 0; i < LEN; i++) {
        printf("i:%d\n",i);
        if (fabsf(host_dst[i] - 2 * i) > EPS) {
            printf("%f expected, but %f got!\n", (float)(2 * i), host_dst[i]);
        } 
    }
    float timeTotal;
    CNRT_CHECK(cnrtNotifierDuration(start, end, &timeTotal));
    printf("Total Time: %.3f ms\n", timeTotal / 1000.0);

    CNRT_CHECK(cnrtQueueDestroy(queue));
    cnrtFree(mlu_dst);
    cnrtFree(mlu_src1);
    cnrtFree(mlu_src2);
    free(host_dst);
    free(host_src1);
    free(host_src2);
    
    return 0; 
}

