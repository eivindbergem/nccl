/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "socket.h"
#include "net.h"
#include "topo.h"
#include "utils.h"
#include "param.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <sys/types.h>
#include <unistd.h>

#include <sisci_api.h>
#include <sisci_error.h>

#define NO_FLAGS          0
#define NO_OFFSET         0
#define NO_CALLBACK       NULL
#define NO_ARG            NULL
#define MAX_SCI_DEVS      1
#define COMM_SEGMENT_SIZE 4

#define SCI_ERROR sci_error
#define SCI(fn)                                 \
    ({                                          \
    sci_error_t  SCI_ERROR = SCI_ERR_OK;        \
    fn;                                         \
    if (handle_sisci_error(__FILE__, __LINE__, SCI_ERROR)) { \
      return ncclInternalError;                 \
    }                                           \
    })

#define SCI_RETURN(fn, ret_type)                \
    ({                                          \
    sci_error_t SCI_ERROR = SCI_ERR_OK;         \
    ret_type    ret       = fn;                 \
    if (handle_sisci_error(__FILE__, __LINE__, SCI_ERROR)) { \
      return ncclInternalError;                 \
    }                                           \
    ret;                                        \
    })

#define SCI_WAIT(fn)                            \
  do {						\
    sci_error_t SCI_ERROR;			\
    do {					\
      fn;					\
    } while (SCI_ERROR != SCI_ERR_OK);		\
  } while (0)

NCCL_PARAM(SciDisable, "SCI_DISABLE", 0);

struct ncclSisciDev {
    unsigned int adapter_no;
    unsigned int node_id;
};

struct ncclSisciDev ncclSisciDevs[MAX_SCI_DEVS];

pthread_mutex_t ncclSisciLock = PTHREAD_MUTEX_INITIALIZER;

int handle_sisci_error(const char *filename, int lineno, sci_error_t error) {
  if (error != SCI_ERR_OK) {
    printf("SCI error at %s:%d: %s\n", filename, lineno, SCIGetErrorString(error));
    return 1;
  }

  return 0;
}

// Initialize the network.
ncclResult_t ncclSisciInit(ncclDebugLogger_t logFunction) {
    SCI(SCIInitialize(NO_FLAGS, &SCI_ERROR));

    struct ncclSisciDev dev = ncclSisciDevs[0];

    dev.adapter_no = 0;

    SCI(SCIGetLocalNodeId(dev.adapter_no, &dev.node_id, NO_FLAGS, &SCI_ERROR));

    INFO(NCCL_INIT|NCCL_NET, "NET/SCI : adapter %u, node id %u",
         dev.adapter_no, dev.node_id);

    return ncclSuccess;
}

// Return the number of adapters.
ncclResult_t ncclSisciDevices(int* ndev) {
    // return 1;
    // return (ncclSisciDevs[0] != NULL ? 1 : 0);
    *ndev = 1;
    return ncclSuccess;
}

// Return the device path in /sys. NCCL will call free on this path.
ncclResult_t ncclSisciPciPath(int dev, char** path) {
    strcpy(*path, "/sys/class/pci_bus/0000:08/device");
    return ncclSuccess;
}

// Return whether this device supports host pointers and/or CUDA pointers
// as data from the current GPU. Supported types should be composed with
// NCCL_PTR_HOST and NCCL_PTR_CUDA.
ncclResult_t ncclSisciPtrSupport(int dev, int* supportedTypes) {
    *supportedTypes = NCCL_PTR_HOST | NCCL_PTR_CUDA;

    return ncclSuccess;
}

struct ncclSisciRequest {
    int size;
    int done;
};

struct ncclSisciListenComm {
    struct ncclSisciDev *dev;
    sci_desc_t sd;
    sci_local_data_interrupt_t ir;
};

struct ncclSisciSendComm {
    struct ncclSisciDev *dev;
    sci_desc_t sd;
    sci_remote_segment_t segment;
    sci_map_t map;
    volatile void *addr;
    sci_dma_queue_t dq;
};

struct ncclSisciRecvComm {
    struct ncclSisciDev *dev;
    sci_desc_t sd;
    sci_local_segment_t segment;
    sci_map_t map;
    volatile void *addr;
};

struct ncclSisciHandle {
    unsigned int node_id;
    unsigned int irno;
};

// Create a receiving object and provide a handle to connect to it. The
// handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
// between ranks to create a connection.
ncclResult_t ncclSisciListen(int dev, void* opaqueHandle, void** listenComm) {
    struct ncclSisciListenComm *comm;

    NCCLCHECK(ncclCalloc(&comm, 1));
    comm->dev = &ncclSisciDevs[dev];
    SCI(SCIOpen(&comm->sd, NO_FLAGS, &SCI_ERROR));
    *listenComm = comm;

    struct ncclSisciHandle* handle = (struct ncclSisciHandle*) opaqueHandle;
    static_assert(sizeof(struct ncclSisciHandle) < NCCL_NET_HANDLE_MAXSIZE,
                  "ncclSisciHandle size too large");
    handle->node_id = comm->dev->node_id;

    SCI(SCICreateDataInterrupt(comm->sd, &comm->ir, comm->dev->adapter_no, &handle->irno,
                               NO_CALLBACK, NO_ARG, NO_FLAGS, &SCI_ERROR));

    return ncclSuccess;
}

// Connect to a handle and return a sending comm object for that peer.
ncclResult_t ncclSisciConnect(int dev, void* opaqueHandle, void** sendComm) {
    struct ncclSisciSendComm *comm;
    struct ncclSisciHandle* handle = (struct ncclSisciHandle*) opaqueHandle;

    NCCLCHECK(ncclCalloc(&comm, 1));
    comm->dev = &ncclSisciDevs[dev];

    sci_desc_t sd;
    sci_remote_data_interrupt_t ir;
    uint32_t data = htons(comm->dev->node_id);

    SCI(SCIOpen(&sd, NO_FLAGS, &SCI_ERROR));
    SCI(SCIConnectDataInterrupt(sd, &ir, handle->node_id,
                                comm->dev->adapter_no, handle->irno,
                                SCI_INFINITE_TIMEOUT, NO_FLAGS,
                                &SCI_ERROR));
    SCI(SCITriggerDataInterrupt(ir, &data, sizeof(data), NO_FLAGS, &SCI_ERROR));

    SCI(SCIOpen(&comm->sd, NO_FLAGS, &SCI_ERROR));

    SCI_WAIT(SCIConnectSegment(comm->sd, &comm->segment, handle->node_id,
                               comm->dev->node_id, comm->dev->adapter_no,
                               NO_CALLBACK, NO_ARG, SCI_INFINITE_TIMEOUT,
                               NO_FLAGS, &SCI_ERROR));

    comm->addr = SCI_RETURN(SCIMapRemoteSegment(comm->segment, &comm->map, NO_OFFSET,
                                                COMM_SEGMENT_SIZE, NULL, NO_FLAGS,
                                                &SCI_ERROR),
                            volatile void*);

    *sendComm = comm;

    return ncclSuccess;
}

// Finalize connection establishment after remote peer has called connectHandel
ncclResult_t ncclSisciAccept(void* listenComm, void** recvComm) {
    struct ncclSisciListenComm *lcomm = (struct ncclSisciListenComm*)listenComm;
    struct ncclSisciRecvComm *rcomm;
    uint32_t data;
    unsigned int size = sizeof(data);
    unsigned int segment_id;

    SCI(SCIWaitForDataInterrupt(lcomm->ir, &data, &size, SCI_INFINITE_TIMEOUT,
                                NO_FLAGS, &SCI_ERROR));
    segment_id = ntohs(data);

    NCCLCHECK(ncclCalloc(&rcomm, 1));
    rcomm->dev = lcomm->dev;
    SCI(SCICreateSegment(rcomm->sd, &rcomm->segment, segment_id,
                         COMM_SEGMENT_SIZE, NO_CALLBACK, NO_ARG,
                         NO_FLAGS, &SCI_ERROR));

    SCI(SCIPrepareSegment(rcomm->segment, rcomm->dev->adapter_no,
                          NO_FLAGS, &SCI_ERROR));

    SCI(SCISetSegmentAvailable(rcomm->segment, rcomm->dev->adapter_no,
                               NO_FLAGS, &SCI_ERROR));

    rcomm->addr = SCI_RETURN(SCIMapLocalSegment(rcomm->segment,
                                                &rcomm->map,
                                                NO_OFFSET,
                                                COMM_SEGMENT_SIZE,
                                                NULL,
                                                NO_FLAGS,
                                                &SCI_ERROR),
                             volatile void*);

    return ncclSuccess;
}

// Register/Deregister memory. Comm can be either a sendComm or a recvComm.
// Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
ncclResult_t ncclSisciRegMr(void* comm, void* data, int size, int type, void** mhandle) {
    return ncclInternalError;
}
ncclResult_t ncclSisciDeregMr(void* comm, void* mhandle) {
    return ncclInternalError;
}

// Asynchronous send to a peer.
// May return request == NULL if the call cannot be performed (or would block)
ncclResult_t ncclSisciIsend(void* sendComm, void* data, int size, void* mhandle, void** request) {
    return ncclInternalError;
}

// Asynchronous recv from a peer.
// May return request == NULL if the call cannot be performed (or would block)
ncclResult_t ncclSisciIrecv(void* recvComm, void* data, int size, void* mhandle, void** request) {
    return ncclInternalError;
}

// Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
// visible to the GPU
ncclResult_t ncclSisciFlush(void* recvComm, void* data, int size, void* mhandle) {
    return ncclInternalError;
}

// Test whether a request is complete. If size is not NULL, it returns the
// number of bytes sent/received.
ncclResult_t ncclSisciTest(void* request, int* done, int* size) {
    return ncclInternalError;
}

// Close and free send/recv comm objects
ncclResult_t ncclSisciCloseSend(void* sendComm) {
    return ncclInternalError;
}
ncclResult_t ncclSisciCloseRecv(void* recvComm) {
    return ncclInternalError;
}
ncclResult_t ncclSisciCloseListen(void* listenComm) {
    return ncclInternalError;
}

ncclNet_t ncclNetSisci = {
  "SCI",
  ncclSisciInit,
  ncclSisciDevices,
  ncclSisciPciPath,
  ncclSisciPtrSupport,
  ncclSisciListen,
  ncclSisciConnect,
  ncclSisciAccept,
  ncclSisciRegMr,
  ncclSisciDeregMr,
  ncclSisciIsend,
  ncclSisciIrecv,
  ncclSisciFlush,
  ncclSisciTest,
  ncclSisciCloseSend,
  ncclSisciCloseRecv,
  ncclSisciCloseListen
};
