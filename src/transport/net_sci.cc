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

#define NO_FLAGS              0
#define NO_OFFSET             0
#define NO_CALLBACK           NULL
#define NO_ARG                NULL
#define MAX_SCI_DEVS          4
#define MAILBOX_SEGMENT_SIZE  2
#define INFINITE_TIMEOUT      0xffffffff
#define MAX_NODES             60
#define SEGMENT_PREFIX        0xbaba0000
#define MAILBOX_SEGMENT_ID    SEGMENT_PREFIX | 1
#define MEMORY_SEGMENTS       2
#define MEMORY_SEGMENT_PREFIX SEGMENT_PREFIX | 0x0000ba00
#define REQUEST_BUFFER_SIZE   32

#define COMM_FLAG_EMPTY 0
#define COMM_FLAG_NOTIFY 1
#define COMM_FLAG_ACK 2

NCCL_PARAM(SciDisable, "SCI_DISABLE", 0);

struct ncclSisciDev {
    unsigned int available;
    unsigned int adapter_no;
    unsigned int node_id;
    unsigned int node_offset;
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

    for (int i = 0; i < MAX_SCI_DEVS; i++) {
        struct ncclSisciDev *dev = &ncclSisciDevs[i];

        dev->adapter_no = i;

        if (WrapSisciGetLocalNodeId(dev->adapter_no, &dev->node_id, NO_FLAGS) ==
            ncclSuccess) {
            INFO(NCCL_INIT|NCCL_NET, "NET/SISCI : adapter %u, node id %u",
                 dev->adapter_no, dev->node_id);

            dev->node_offset = (dev->node_id >> 2) - 1;

            dev->available = 1;
        }
        else {
            break;
        }
    }

    return ncclSuccess;
}

// Return the number of adapters.
ncclResult_t ncclSisciDevices(int* ndev) {
    // return 1;
    // return (ncclSisciDevs[0] != NULL ? 1 : 0);
    for (int i = 0; i < MAX_SCI_DEVS; i++) {
        if (ncclSisciDevs[i].available == 0) {
            *ndev = i;
            break;
        }
    }

    return ncclSuccess;
}

// Return the device path in /sys. NCCL will call free on this path.
ncclResult_t ncclSisciPciPath(int dev, char** path) {
    char devicepath[PATH_MAX];
    strcpy(devicepath, "/sys/class/pci_bus/0000:08/device");
    *path = realpath(devicepath, NULL);

    return ncclSuccess;
}

// Return whether this device supports host pointers and/or CUDA pointers
// as data from the current GPU. Supported types should be composed with
// NCCL_PTR_HOST and NCCL_PTR_CUDA.
ncclResult_t ncclSisciPtrSupport(int dev, int* supportedTypes) {
    *supportedTypes = NCCL_PTR_HOST | NCCL_PTR_CUDA;

    return ncclSuccess;
}

struct ncclSisciHandle {
    unsigned int node_id;
    unsigned int irno;
};

struct ncclSisciListenComm {
    struct ncclSisciDev *dev;
    sci_desc_t sd;
    sci_local_data_interrupt_t ir;
    sci_local_segment_t segment;
    sci_map_t map;
    volatile void *addr;
};

struct ncclSisciMailbox {
    sci_desc_t sd;
    sci_local_segment_t local_segment;
    sci_remote_segment_t remote_segment;
    sci_map_t local_map;
    sci_map_t remote_map;
    volatile void *local_addr;
    volatile void *remote_addr;
};

enum ncclSisciCommType { SISCI_RECV,
                         SISCI_SEND };

enum ncclSisciCommState {COMM_READY,
                         SEND_POSTED,
                         SEND_WAIT_ACK,
                         RECV_WAITING};

struct ncclSisciComm {
    sci_desc_t sd;
    enum ncclSisciCommType type;
    unsigned int mem_handle_cnt;
    unsigned int remote_node_offset;
    unsigned int remote_node_id;
    struct ncclSisciDev *dev;
    struct ncclSisciMailbox *mailbox;
    sci_dma_queue_t dq;
    unsigned int request_cnt;
    enum ncclSisciCommState state[REQUEST_BUFFER_SIZE];
};

#define ncclSisciRecvComm ncclSisciComm
#define ncclSisciSendComm ncclSisciComm

/* struct ncclSisciRecvComm { */
/*     struct ncclSisciComm; */
/*     /\* enum ncclSisciCommType type; *\/ */
/*     /\* unsigned int mem_handle_cnt; *\/ */
/*     /\* unsigned int remote_node_offset; *\/ */
/*     /\* struct ncclSisciDev *dev; *\/ */
/*     /\* struct ncclSisciMailbox *mailbox; *\/ */
/*     /\* unsigned int request_cnt; *\/ */
/* }; */

/* struct ncclSisciSendComm { */
/*     struct ncclSisciComm; */
/*     /\* enum ncclSisciCommType type; *\/ */
/*     /\* unsigned int mem_handle_cnt; *\/ */
/*     /\* unsigned int remote_node_offset; *\/ */
/*     /\* struct ncclSisciDev *dev; *\/ */

/*     /\* sci_dma_queue_t dq; *\/ */
/*     /\* unsigned int request_cnt; *\/ */
/* }; */


struct ncclSisciMemHandle {
    sci_desc_t sd;
    sci_local_segment_t local_segment;
    sci_remote_segment_t remote_segment;
    unsigned int segment_id;
    unsigned int remote_segment_id;
    unsigned int memory_id;
    sci_map_t map;
    volatile void *segment_addr;
    void *addr;
    unsigned int busy;
};

struct ncclSisciRequest {
    enum ncclSisciCommType type;
    void *comm;
    unsigned int memory_id;
    unsigned int id;
    void *data;
    unsigned int size;
    unsigned int offset;
    struct ncclSisciMemHandle *memhandle;
    volatile uint32_t *local_flag;
    volatile uint32_t *remote_flag;
    volatile uint32_t *local_size;
    volatile uint32_t *remote_size;
    enum ncclSisciCommState *state;
};

static unsigned int memory_segment_id(unsigned int node_offset,
                                      enum ncclSisciCommType type,
                                      unsigned int i) {
    return MEMORY_SEGMENT_PREFIX | (type << 8) | (node_offset << 1) | i;
}

static unsigned int get_mailbox_id(enum ncclSisciCommType type,
                                   unsigned int i) {
    return SEGMENT_PREFIX | (type << 8) | i;
}

void print_mailbox(volatile void* addr) {
    for (int i = 0; i < MAILBOX_SEGMENT_SIZE*MAX_NODES; i++) {
        printf("%d ", ((uint32_t*)addr)[i]);
    }
    printf("\n");
}

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

    /* NCCLCHECK(WrapSisciCreateSegment(comm->sd, &comm->segment, MAILBOX_SEGMENT_ID, */
    /*                                  MAILBOX_SEGMENT_SIZE*MAX_NODES*sizeof(uint32_t), */
    /*                                  NO_CALLBACK, NO_ARG, NO_FLAGS)); */

    /* NCCLCHECK(WrapSisciPrepareSegment(comm->segment, comm->dev->adapter_no, */
    /*                       NO_FLAGS)); */

    /* NCCLCHECK(WrapSisciSetSegmentAvailable(comm->segment, comm->dev->adapter_no, */
    /*                            NO_FLAGS)); */

    /* NCCLCHECK(WrapSisciMapLocalSegment(comm->segment, */
    /*                                    &comm->map, */
    /*                                    0, */
    /*                                    MAILBOX_SEGMENT_SIZE*MAX_NODES*sizeof(uint32_t), */
    /*                                    &comm->addr, */
    /*                                    NO_FLAGS)); */

    // for (int i = 0; i < MAILBOX_SEGMENT_SIZE*MAX_NODES; i++) {
    //     ((uint32_t*)comm->addr)[i] = 1;
    // }

    // NCCLCHECK(WrapSisciMapLocalSegment(comm->segment,
    //                              &comm->map,
    //                              NO_OFFSET,
    //                              COMM_SEGMENT_SIZE,
    //                              &comm->addr,
    //                              NO_FLAGS));


    return ncclSuccess;
}

static ncclResult_t ncclSisciCreateMailbox(struct ncclSisciDev *dev,
                                           struct ncclSisciMailbox **mailbox,
                                           unsigned int segment_id) {
    struct ncclSisciMailbox *mbox;

    NCCLCHECK(ncclCalloc(&mbox, 1));
    NCCLCHECK(WrapSisciOpen(&mbox->sd, NO_FLAGS));
    NCCLCHECK(WrapSisciCreateSegment(mbox->sd, &mbox->local_segment,
                                     segment_id,
                                     MAILBOX_SEGMENT_SIZE*REQUEST_BUFFER_SIZE*sizeof(uint32_t),
                                     NO_CALLBACK, NO_ARG, NO_FLAGS));

    NCCLCHECK(WrapSisciPrepareSegment(mbox->local_segment, dev->adapter_no,
                                      NO_FLAGS));

    NCCLCHECK(WrapSisciSetSegmentAvailable(mbox->local_segment, dev->adapter_no,
                                           NO_FLAGS));

    NCCLCHECK(WrapSisciMapLocalSegment(mbox->local_segment,
                                       &mbox->local_map,
                                       0,
                                       MAILBOX_SEGMENT_SIZE*MAX_NODES*sizeof(uint32_t),
                                       &mbox->local_addr,
                                       NO_FLAGS));

    *mailbox = mbox;

    return ncclSuccess;
}

static ncclResult_t ncclSisciConnectMailbox(struct ncclSisciDev *dev,
                                            struct ncclSisciMailbox *mailbox,
                                            unsigned int segment_id,
                                            unsigned int remote_node) {
    while (WrapSisciConnectSegment(mailbox->sd, &mailbox->remote_segment, remote_node,
                                   segment_id, dev->adapter_no,
                                   NO_CALLBACK, NO_ARG, INFINITE_TIMEOUT,
                                   NO_FLAGS) != ncclSuccess) {
        sleep(1);
    }

    NCCLCHECK(WrapSisciMapRemoteSegment(mailbox->remote_segment,
                                        &mailbox->remote_map,
                                        NO_OFFSET,
                                        MAILBOX_SEGMENT_SIZE*MAX_NODES*sizeof(uint32_t),
                                        &mailbox->remote_addr, NO_FLAGS));

    return ncclSuccess;
}

// Connect to a handle and return a sending comm object for that peer.
ncclResult_t ncclSisciConnect(int dev, void* opaqueHandle, void** sendComm) {
    struct ncclSisciSendComm *comm;
    struct ncclSisciHandle* handle = (struct ncclSisciHandle*) opaqueHandle;

    NCCLCHECK(ncclCalloc(&comm, 1));
    comm->dev = &ncclSisciDevs[dev];
    comm->remote_node_id = handle->node_id;
    comm->remote_node_offset = (comm->remote_node_id >> 2) - 1;
    comm->type = SISCI_SEND;

    sci_remote_data_interrupt_t ir;
    uint32_t data = htons(comm->dev->node_offset);

    NCCLCHECK(WrapSisciOpen(&comm->sd, NO_FLAGS));
    NCCLCHECK(WrapSisciConnectDataInterrupt(comm->sd, &ir, handle->node_id,
                                comm->dev->adapter_no, handle->irno,
                                INFINITE_TIMEOUT, NO_FLAGS));
    NCCLCHECK(WrapSisciTriggerDataInterrupt(ir, &data, sizeof(data), NO_FLAGS));
    /* TODO: Disconnect interrupt */

    NCCLCHECK(ncclSisciCreateMailbox(comm->dev, &comm->mailbox,
                                     get_mailbox_id(SISCI_SEND,
                                                    comm->remote_node_offset)));

    /* NCCLCHECK(WrapSisciOpen(&comm->sd, NO_FLAGS)); */

    /* NCCLCHECK(WrapSisciCreateSegment(comm->sd, &comm->segment, */
    /*                                  get_mailbox_id(SISCI_SEND, remote_node_offset), */
    /*                                  MAILBOX_SEGMENT_SIZE*REQUEST_BUFFER_SIZE*sizeof(uint32_t), */
    /*                                  NO_CALLBACK, NO_ARG, NO_FLAGS)); */

    /* NCCLCHECK(WrapSisciPrepareSegment(comm->segment, comm->dev->adapter_no, */
    /*                                   NO_FLAGS)); */

    /* NCCLCHECK(WrapSisciSetSegmentAvailable(comm->segment, comm->dev->adapter_no, */
    /*                                        NO_FLAGS)); */

    /* NCCLCHECK(WrapSisciMapLocalSegment(comm->segment, */
    /*                                    &comm->map, */
    /*                                    0, */
    /*                                    MAILBOX_SEGMENT_SIZE*MAX_NODES*sizeof(uint32_t), */
    /*                                    &comm->mailbox, */
    /*                                    NO_FLAGS)); */

    /* while (WrapSisciConnectSegment(comm->sd, &comm->mailbox, handle->node_id, */
    /*                                MAILBOX_SEGMENT_ID, comm->dev->adapter_no, */
    /*                                NO_CALLBACK, NO_ARG, INFINITE_TIMEOUT, */
    /*                                NO_FLAGS) != ncclSuccess) { */
    /*     sleep(1); */
    /* } */

    /* NCCLCHECK(WrapSisciMapRemoteSegment(comm->mailbox, &comm->map, */
    /*                                     MAILBOX_SEGMENT_SIZE*comm->dev->node_offset*sizeof(uint32_t), */
    /*                                     MAILBOX_SEGMENT_SIZE*sizeof(uint32_t), &comm->addr, NO_FLAGS)) */;
        // NCCLCHECK(WrapSisciMapRemoteSegment(comm->mailbox, &comm->map,
        //                                     0,
        //                                     MAILBOX_SEGMENT_SIZE*MAX_NODES,
        //                                     &comm->addr, NO_FLAGS));


    NCCLCHECK(WrapSisciCreateDMAQueue(comm->sd, &comm->dq, comm->dev->adapter_no,
                                      1, NO_FLAGS));

    *sendComm = comm;

    return ncclSuccess;
}

// Finalize connection establishment after remote peer has called connectHandel
ncclResult_t ncclSisciAccept(void* listenComm, void** recvComm) {
    struct ncclSisciListenComm *lcomm = (struct ncclSisciListenComm*)listenComm;

    struct ncclSisciRecvComm *rcomm;

    uint32_t data;
    unsigned int size = sizeof(data);

    NCCLCHECK(ncclCalloc(&rcomm, 1));
    rcomm->dev = lcomm->dev;
    rcomm->type = SISCI_RECV;

    NCCLCHECK(WrapSisciWaitForDataInterrupt(lcomm->ir, &data, &size, INFINITE_TIMEOUT,
                                            NO_FLAGS));
    rcomm->remote_node_offset = ntohs(data);
    rcomm->remote_node_id = (rcomm->remote_node_offset+1)*4;
    /* rcomm->addr = (uint32_t*)lcomm->addr + MAILBOX_SEGMENT_SIZE*rcomm->remote_node_offset; */
    /* rcomm->mailbox = lcomm->addr; */

    NCCLCHECK(ncclSisciCreateMailbox(rcomm->dev, &rcomm->mailbox,
                                     get_mailbox_id(SISCI_RECV,
                                                    rcomm->remote_node_offset)));

    /* NCCLCHECK(WrapSisciCreateSegment(rcomm->sd, &rcomm->segment, */
    /*                                  get_mailbox_id(SISCI_RECV, remote_node_offset), */
    /*                                  MAILBOX_SEGMENT_SIZE*REQUEST_BUFFER_SIZE*sizeof(uint32_t), */
    /*                                  NO_CALLBACK, NO_ARG, NO_FLAGS)); */

    /* NCCLCHECK(WrapSisciPrepareSegment(rcomm->segment, rcomm->dev->adapter_no, */
    /*                                   NO_FLAGS)); */

    /* NCCLCHECK(WrapSisciSetSegmentAvailable(rcomm->segment, rcomm->dev->adapter_no, */
    /*                                        NO_FLAGS)); */

    /* NCCLCHECK(WrapSisciMapLocalSegment(rcomm->segment, */
    /*                                    &rcomm->map, */
    /*                                    0, */
    /*                                    MAILBOX_SEGMENT_SIZE*MAX_NODES*sizeof(uint32_t), */
    /*                                    &rcomm->mailbox, */
    /*                                    NO_FLAGS)); */

    *recvComm = rcomm;
    // remote_node = ntohs(data);

    // NCCLCHECK(ncclCalloc(&rcomm, 1));
    // rcomm->dev = lcomm->dev;
    // rcomm->mailbox = lcomm->segment;

    // NCCLCHECK(WrapSisciMapLocalSegment(rcomm->mailbox,
    //                                    &rcomm->map,
    //                                    MAILBOX_SEGMENT_SIZE*remote_node,
    //                                    MAILBOX_SEGMENT_SIZE,
    //                                    &rcomm->addr,
    //                                    NO_FLAGS));

    return ncclSuccess;
}

void* devptr(void* ptr)
{
    cudaPointerAttributes attrs;

    cudaError_t err; // = cudaSetDevice(gpu);
    // if (err != cudaSuccess)
    // {
        // log_error("Failed to set GPU: %s", cudaGetErrorString(err));
    //     return NULL;
    // }

    err = cudaPointerGetAttributes(&attrs, ptr);
    if (err != cudaSuccess)
    {
        WARN("Failed to get pointer attributes: %s", cudaGetErrorString(err));
        return NULL;
    }

    INFO(NCCL_NET, "CUDA device buffer %p has device ptr %p", ptr, attrs.devicePointer);
    return attrs.hostPointer;
}

// Register/Deregister memory. Comm can be either a sendComm or a recvComm.
// Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
ncclResult_t ncclSisciRegMr(void* comm, void* data, int size, int type, void** mhandle) {
    struct ncclSisciComm *gcomm = (struct ncclSisciComm*)comm;

    struct ncclSisciMemHandle *memhandle;
    NCCLCHECK(ncclCalloc(&memhandle, 1));
    memhandle->memory_id = gcomm->mem_handle_cnt++;
    memhandle->segment_id = memory_segment_id(gcomm->remote_node_offset,
                                              gcomm->type,
                                              memhandle->memory_id);
    memhandle->remote_segment_id = memory_segment_id(gcomm->dev->node_offset,
                                                     (gcomm->type == SISCI_SEND ? SISCI_RECV : SISCI_SEND),
                                                     memhandle->memory_id);
    // memhandle->addr = devptr(data);
    memhandle->addr = data;

    NCCLCHECK(WrapSisciOpen(&memhandle->sd, NO_FLAGS));

    NCCLCHECK(WrapSisciCreateSegment(memhandle->sd, &memhandle->local_segment,
                                     memhandle->segment_id, size,
                                     NO_CALLBACK, NO_ARG, SCI_FLAG_EMPTY));

    if (type == NCCL_PTR_CUDA) {
        NCCLCHECK(WrapSisciAttachPhysicalMemory((sci_ioaddr_t)memhandle->addr, NULL, 0, size,
                                                memhandle->local_segment,
                                                SCI_FLAG_CUDA_BUFFER));
    } else {
        NCCLCHECK(WrapSisciRegisterSegmentMemory(memhandle->addr, size,
                                                 memhandle->local_segment,
                                                 NO_FLAGS));
    }

    NCCLCHECK(WrapSisciPrepareSegment(memhandle->local_segment, gcomm->dev->adapter_no,
                                      NO_FLAGS));
    NCCLCHECK(WrapSisciSetSegmentAvailable(memhandle->local_segment, gcomm->dev->adapter_no,
                                           NO_FLAGS));


    // if (gcomm->type == SISCI_RECV_COMM) {
    //     struct ncclSisciRecvComm *rcomm = (struct ncclSisicRecvComm*)comm;
    // } else {
    //     struct ncclSisciSendComm *scomm = (struct ncclSisicSendComm*)comm;
    // }

    *mhandle = memhandle;

    return ncclSuccess;
    // struct ncclSisicMemHandle memhandle;

    // NCCLCHECK(ncclCalloc(&memhandle, 1));
    // NCCLCHECK(WrapSisciOpen(&memhandle->sd, NO_FLAGS));
    // NCCLCHECK(WrapSisciCreateSegment(memhandle->sd, &memhandle->segment,));
    // NCCLCHECK(WrapSisciAttachPhysicalMemory());
    // return ncclSuccess;
    // return ncclInternalError;
}
ncclResult_t ncclSisciDeregMr(void* comm, void* mhandle) {
    /* return ncclInternalError; */
    return ncclSuccess;
}



uint16_t fletcher16( uint8_t *data, int count )
{
   uint16_t sum1 = 0;
   uint16_t sum2 = 0;
   int index;

   for ( index = 0; index < count; ++index )
   {
      sum1 = (sum1 + data[index]) % 255;
      sum2 = (sum2 + sum1) % 255;
   }

   return (sum2 << 8) | sum1;
}

// Asynchronous send to a peer.
// May return request == NULL if the call cannot be performed (or would block)
ncclResult_t ncclSisciIsend(void* sendComm, void* data, int size, void* mhandle, void** request) {
    struct ncclSisciSendComm *comm = (struct ncclSisciSendComm*)sendComm;
    struct ncclSisciMemHandle *memhandle = (struct ncclSisciMemHandle*)mhandle;
    struct ncclSisciRequest *req;
    size_t offset = (uint8_t*)data - (uint8_t*)memhandle->addr;
    enum ncclSisciCommState state = comm->state[comm->request_cnt % REQUEST_BUFFER_SIZE];

    printf("Try send: state=%u, memory_id=%d, data=%u\n",
           state,
           memhandle->memory_id,
           *(uint32_t*)data);

    if (state != COMM_READY) {
        *request = NULL;
        return ncclSuccess;
    }

    NCCLCHECK(ncclCalloc(&req, 1));

    if (memhandle->remote_segment == NULL) {
        while (WrapSisciConnectSegment(memhandle->sd, &memhandle->remote_segment,
                                       comm->remote_node_id, memhandle->remote_segment_id,
                                       comm->dev->adapter_no, NO_CALLBACK, NO_ARG,
                                       SCI_INFINITE_TIMEOUT, NO_FLAGS) != ncclSuccess) {
            sleep(1);
        }
    }

    if (comm->mailbox->remote_addr == NULL) {
        NCCLCHECK(ncclSisciConnectMailbox(comm->dev, comm->mailbox,
                                          get_mailbox_id(SISCI_RECV, comm->dev->node_offset),
                                          comm->remote_node_id));
    }

    if (size > 0) {
        NCCLCHECK(WrapSisciStartDmaTransfer(comm->dq, memhandle->local_segment,
                                            memhandle->remote_segment, offset, size,
                                            offset, NO_CALLBACK, NO_ARG, NO_FLAGS));
    }

    req->type = SISCI_SEND;
    req->comm = sendComm;
    req->memory_id = memhandle->memory_id;
    req->id = comm->request_cnt++;
    req->data = data;
    req->size = size;
    req->memhandle = memhandle;
    req->offset = offset;
    req->local_flag = ((volatile uint32_t*)comm->mailbox->local_addr +
                       (req->id % REQUEST_BUFFER_SIZE)*2);
    req->local_size = req->local_flag + 1;
    req->remote_flag = ((volatile uint32_t*)comm->mailbox->remote_addr +
                        (req->id % REQUEST_BUFFER_SIZE)*2);
    req->remote_size = req->remote_flag + 1;
    req->state = &comm->state[req->id % REQUEST_BUFFER_SIZE];
    *req->state = SEND_POSTED;

    printf("Sending request %d: size=%d, offset=%lu, local_segment=%x, remote_segment=%x, checksum=%04x\n",
           req->id, size, offset, memhandle->segment_id,
           memhandle->remote_segment_id, fletcher16((uint8_t*)data, size));

    *request = req;

    // *((uint8_t*)comm->addr+req->memory_id) = 1;

    return ncclSuccess;

    // return ncclInternalError;
}

// Asynchronous recv from a peer.
// May return request == NULL if the call cannot be performed (or would block)
ncclResult_t ncclSisciIrecv(void* recvComm, void* data, int size, void* mhandle, void** request) {
    struct ncclSisciRequest *req;
    struct ncclSisciMemHandle *memhandle = (struct ncclSisciMemHandle*)mhandle;
    struct ncclSisciRecvComm *comm = (struct ncclSisciRecvComm*)recvComm;
    size_t offset = (uint8_t*)data - (uint8_t*)memhandle->addr;
    enum ncclSisciCommState state = comm->state[comm->request_cnt % REQUEST_BUFFER_SIZE];

    printf("Try recv: state=%u, memory_id=%d, data=%u\n",
           state,
           memhandle->memory_id,
           *(uint32_t*)data);

    if (state != COMM_READY) {
        *request = NULL;
        return ncclSuccess;
    }

    NCCLCHECK(ncclCalloc(&req, 1));

    if (comm->mailbox->remote_addr == NULL) {
        NCCLCHECK(ncclSisciConnectMailbox(comm->dev, comm->mailbox,
                                          get_mailbox_id(SISCI_SEND, comm->dev->node_offset),
                                          comm->remote_node_id));
    }

    req->type = SISCI_RECV;
    req->comm = recvComm;
    req->memory_id = memhandle->memory_id;
    req->id = comm->request_cnt++;
    req->data = data;
    req->size = size;
    req->memhandle = memhandle;
    req->offset = offset;
    req->local_flag = ((volatile uint32_t*)comm->mailbox->local_addr +
                       (req->id % REQUEST_BUFFER_SIZE)*2);
    req->local_size = req->local_flag + 1;
    req->remote_flag = ((volatile uint32_t*)comm->mailbox->remote_addr +
                        (req->id % REQUEST_BUFFER_SIZE)*2);
    req->remote_size = req->remote_flag + 1;
    req->state = &comm->state[req->id % REQUEST_BUFFER_SIZE];
    *req->state = RECV_WAITING;

    printf("Receiving request %d: size=%d, offset=%lu, segment=%x\n",
           req->id, size, offset, memhandle->segment_id);

    *request = req;

    return ncclSuccess;
}

// Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
// visible to the GPU
ncclResult_t ncclSisciFlush(void* recvComm, void* data, int size, void* mhandle) {
    // return ncclSuccess;
    return ncclInternalError;
}

// Test whether a request is complete. If size is not NULL, it returns the
// number of bytes sent/received.
ncclResult_t ncclSisciTest(void* request, int* done, int* size) {
    struct ncclSisciRequest *req = (struct ncclSisciRequest*)request;
    *done = 0;

    if (req->type == SISCI_SEND) {
        struct ncclSisciSendComm *comm = (struct ncclSisciSendComm*)req->comm;

        /* printf("Test send: req->id=%d, state=%u, flag=%u\n", req->id, */
        /*        *req->state, *req->local_flag); */

        if (*req->state == SEND_POSTED) {
            sci_dma_queue_state_t state;
            NCCLCHECK(WrapSisciDMAQueueState(comm->dq, &state));
            if (state == SCI_DMAQUEUE_IDLE || state == SCI_DMAQUEUE_DONE) {
                printf("req->id=%d, SEND_POSTED->SEND_WAIT_ACK\n",
                       req->id);

                *req->remote_size = req->size;
                *req->remote_flag = COMM_FLAG_NOTIFY;
                *req->state = SEND_WAIT_ACK;
            }
        }

        if (*req->state == SEND_WAIT_ACK && *req->local_flag == COMM_FLAG_ACK) {
            printf("req->id=%d, SEND_WAIT_ACK->COMM_READY\n",
                   req->id);

            *req->local_flag = COMM_FLAG_EMPTY;
            *req->state = COMM_READY;
            *done = 1;
        }
    }
    else {
        /* struct ncclSisciRecvComm *comm = (struct ncclSisciRecvComm*)req->comm; */

        /* printf("Test recv: req->id=%d, state=%u, flag=%u\n", req->id, */
        /*        *req->state, *req->local_flag); */

        if (*req->state == RECV_WAITING && *req->local_flag == COMM_FLAG_NOTIFY) {
            req->size = *req->local_size;

            *req->local_flag = COMM_FLAG_EMPTY;
            *req->state = COMM_READY;

            *req->remote_flag = COMM_FLAG_ACK;
            *done = 1;

            printf("req->id=%d, RECV_WAITING->COMM_READY\n",
                   req->id);

            printf("Received data: size=%u, offset=%u, checksum=%04x\n",
                   req->size, req->offset, fletcher16((uint8_t*)req->data, req->size));

            // for (int i = 0; i < req->memhandle->size; i++) {
            //     uint32_t *value = ((uint32_t*)req->memhandle->addr + i);
            //     if (*value != 0) {
            //         printf("%d ", i);
            //     }
            //     /* if (i % 32 == 0) { */
            //     /*     printf("\n"); */
            //     /* } */
            //     /* printf("%u ",  */
            // }
            // printf("\n");

            /* for (int i = 0; i < 16; i++) { */
            /*     printf("%u ", *((uint32_t*)req->memhandle->addr + i)); */
            /* } */
            /* printf("\n"); */

            /* for (int i = 0; i < 16; i++) { */
            /*     printf("%u ", *((uint32_t*)req->memhandle->addr + req->offset + i)); */
            /* } */
            /* printf("\n"); */

            /* for (int i = 0; i < 16; i++) { */
            /*     printf("%u ", *((uint32_t*)req->memhandle->addr + req->size + i)); */
            /* } */
            /* printf("\n"); */

            /* for (int i = 0; i < 16; i++) { */
            /*     printf("%u ", *((uint32_t*)req->memhandle->addr - req->size + i)); */
            /* } */
            /* printf("\n"); */



        }

        /* // print_mailbox(comm->mailbox); */
        /* printf("Test recv: req->id=%d, status=%d, memory_id=%d\n", req->id, *status, */
        /*        req->memory_id); */

        /* if (*status == 2) { */
        /*     memcpy(req->data, (void*)req->memhandle->segment_addr, req->size); */

        /*     *status = 3; */
        /*     if (size) *size = req->size; */
        /* } */

        /* if (*status == 4) { */
        /*     *done = 1; */
        /*     *status = 0; */
        /*     comm->unhandled_requests = 0; */
        /* } */

        // *done = 0;
    }

    if (size) *size = req->size;

    return ncclSuccess;
}

// Close and free send/recv comm objects
ncclResult_t ncclSisciCloseSend(void* sendComm) {
    /* return ncclInternalError; */
    return ncclSuccess;
}
ncclResult_t ncclSisciCloseRecv(void* recvComm) {
    return ncclSuccess;
    /* return ncclInternalError; */
}
ncclResult_t ncclSisciCloseListen(void* listenComm) {
    /* struct ncclSisciListenComm *comm = (struct ncclSisciListenComm*)listenComm; */

    return ncclSuccess;
}

// ncclResult_t ncclSisciHostAlloc(void *comm, void** ptr, void** devPtr,
//                                 size_t size, void** mhandle) {
//     struct ncclSisciComm *gcomm = (struct ncclSisciComm*)comm;

//     struct ncclSisciMemHandle *memhandle;
//     NCCLCHECK(ncclCalloc(&memhandle, 1));
//     memhandle->segment_id = memory_segment_id(gcomm->remote_node_offset,
//                                               size == NCCL_LL_BUFF_SIZE ? 1 : 0);
//     memhandle->remote_segment_id = memory_segment_id(gcomm->dev->node_offset,
//                                                      size == NCCL_LL_BUFF_SIZE ? 1 : 0);

//     NCCLCHECK(WrapSisciOpen(&memhandle->sd, NO_FLAGS));
//     NCCLCHECK(WrapSisciCreateSegment(memhandle->sd, &memhandle->local_segment,
//                                      memhandle->segment_id, size,
//                                      NO_CALLBACK, NO_ARG, NO_FLAGS));

//     NCCLCHECK(WrapSisciPrepareSegment(memhandle->local_segment, gcomm->dev->adapter_no,
//                                       NO_FLAGS));
//     NCCLCHECK(WrapSisciSetSegmentAvailable(memhandle->local_segment, gcomm->dev->adapter_no,
//                                            NO_FLAGS));

//     NCCLCHECK(WrapSisciMapRemoteSegment(memhandle->mailbox, &memhandle->map,
//                                         NO_OFFSET, size, ptr, NO_FLAGS));
//     *devPtr = *ptr;

//     memhandle->addr = *ptr;

//     return ncclSuccess;

// }

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
