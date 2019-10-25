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
#include "sisciwrap.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <sys/types.h>
#include <unistd.h>

#include <sisci_error.h>
#include <sisci_error.h>
#include <sisci_types.h>

#define NO_FLAGS          0
#define NO_OFFSET         0
#define NO_CALLBACK       NULL
#define NO_ARG            NULL
#define MAX_SCI_DEVS      1
#define COMM_SEGMENT_SIZE 4
#define INFINITE_TIMEOUT 0xffffffff

NCCL_PARAM(SciDisable, "SCI_DISABLE", 0);

struct ncclSisciDev {
    unsigned int adapter_no;
    unsigned int node_id;
};

struct ncclSisciDev ncclSisciDevs[MAX_SCI_DEVS];

pthread_mutex_t ncclSisciLock = PTHREAD_MUTEX_INITIALIZER;

// Initialize the network.
ncclResult_t ncclSisciInit(ncclDebugLogger_t logFunction) {
    INFO(NCCL_NET|NCCL_INIT, "Trying to load SISCI");

    if (load_sisci() != ncclSuccess) {
        return ncclInternalError;
    }

    NCCLCHECK(WrapSisciInitialize(NO_FLAGS));

    struct ncclSisciDev dev = ncclSisciDevs[0];

    dev.adapter_no = 0;

    NCCLCHECK(WrapSisciGetLocalNodeId(dev.adapter_no, &dev.node_id, NO_FLAGS));

    INFO(NCCL_INIT|NCCL_NET, "NET/SISCI : adapter %u, node id %u",
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
    NCCLCHECK(WrapSisciOpen(&comm->sd, NO_FLAGS));
    *listenComm = comm;

    struct ncclSisciHandle* handle = (struct ncclSisciHandle*) opaqueHandle;
    static_assert(sizeof(struct ncclSisciHandle) < NCCL_NET_HANDLE_MAXSIZE,
                  "ncclSisciHandle size too large");
    handle->node_id = comm->dev->node_id;

    NCCLCHECK(WrapSisciCreateDataInterrupt(comm->sd, &comm->ir, comm->dev->adapter_no, &handle->irno,
                               NO_CALLBACK, NO_ARG, NO_FLAGS));

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

    NCCLCHECK(WrapSisciOpen(&sd, NO_FLAGS));
    NCCLCHECK(WrapSisciConnectDataInterrupt(sd, &ir, handle->node_id,
                                comm->dev->adapter_no, handle->irno,
                                INFINITE_TIMEOUT, NO_FLAGS));
    NCCLCHECK(WrapSisciTriggerDataInterrupt(ir, &data, sizeof(data), NO_FLAGS));

    NCCLCHECK(WrapSisciOpen(&comm->sd, NO_FLAGS));

    while (WrapSisciConnectSegment(comm->sd, &comm->segment, handle->node_id,
                             comm->dev->node_id, comm->dev->adapter_no,
                             NO_CALLBACK, NO_ARG, INFINITE_TIMEOUT,
                             NO_FLAGS) != ncclSuccess) {
        sleep(1);
    }

    NCCLCHECK(WrapSisciMapRemoteSegment(comm->segment, &comm->map, NO_OFFSET,
                                  COMM_SEGMENT_SIZE, &comm->addr, NO_FLAGS));

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

    NCCLCHECK(WrapSisciWaitForDataInterrupt(lcomm->ir, &data, &size, INFINITE_TIMEOUT,
                                NO_FLAGS));
    segment_id = ntohs(data);

    NCCLCHECK(ncclCalloc(&rcomm, 1));
    rcomm->dev = lcomm->dev;
    NCCLCHECK(WrapSisciCreateSegment(rcomm->sd, &rcomm->segment, segment_id,
                         COMM_SEGMENT_SIZE, NO_CALLBACK, NO_ARG,
                         NO_FLAGS));

    NCCLCHECK(WrapSisciPrepareSegment(rcomm->segment, rcomm->dev->adapter_no,
                          NO_FLAGS));

    NCCLCHECK(WrapSisciSetSegmentAvailable(rcomm->segment, rcomm->dev->adapter_no,
                               NO_FLAGS));

    NCCLCHECK(WrapSisciMapLocalSegment(rcomm->segment,
                                 &rcomm->map,
                                 NO_OFFSET,
                                 COMM_SEGMENT_SIZE,
                                 &rcomm->addr,
                                 NO_FLAGS));

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
  "Sisci",
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