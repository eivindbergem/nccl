#ifndef NCCL_IBVWRAP_H_
#define NCCL_IBVWRAP_H_

#include <sisci_error.h>
#include <sisci_types.h>

#include "core.h"

#define SCI_FLAG_EMPTY       0x00000002
#define SCI_INFINITE_TIMEOUT 0xffffffff

#define SCI_DMA_TYPE_DONTCARE    0x00000001
#define SCI_DMA_TYPE_ADAPTER     0x00000002
#define SCI_DMA_TYPE_GLOBAL      0x00000003
#define SCI_DMA_TYPE_SYSTEM      0x00000004
#define SCI_DMA_TYPE_TRANSPARENT 0x00000005

#define SCI_DMA_CHANNEL_ID_DONTCARE 0xFFFFFFFF

/* FLAG values used in both user and kernel part of API */
#define SCI_FLAG_USE_CALLBACK       0x00000001
#define SCI_FLAG_EMPTY                             0x00000002
#define SCI_FLAG_PRIVATE            0x00000004
#define SCI_FLAG_DMA_SOURCE_ONLY    0x00000008
#define SCI_FLAG_PHYSICAL           0x00001000
#define SCI_FLAG_AUTO_ID            0x20000000

#define SCI_FLAG_CUDA_BUFFER                       0x00002000
#define SCI_FLAG_SCIF_BUFFER                       0x00004000

#define SCI_FLAG_DMA_WAIT                          0x00000004
#define SCI_FLAG_DMA_READ                          0x00000010
#define SCI_FLAGSEG                      0x00000010
#define SCI_FLAG_DMA_PHDMA                         0x00000020
#define SCI_FLAG_FORCE_REMOVE                      0x00000020
#define SCI_FLAG_DMA_GLOBAL                        0x00000040
#define SCI_FLAG_ALLOW_UNICAST                     0x00000080
#define SCI_FLAG_DMA_SYSDMA                        0x00000100
#define SCI_FLAG_DMA_USER_MEM                      0x00000200
#define SCI_FLAG_DMA_ADAPTER                       0x00000800

#define SCI_FLAG_DMA_CHANNEL_SHARED                0x00000001

#define SCI_FLAG_LOCK_USER_MEM                     0x00000400

#define SCI_FLAG_FIXED_INTNO                       0x00000002
#define SCI_FLAG_SHARED_INT                        0x00000004
#define SCI_FLAG_COUNTING_INT                      0x00000020
#define SCI_FLAG_SHARE_EXTERNAL                    0x00000040


#define SCI_FLAG_CONDITIONAL_INTERRUPT             0x00000008

#define SCI_FLAG_FIXED_MAP_ADDR                    0x00000001
#define SCI_FLAG_READONLY_MAP                      0x00000002
#define SCI_FLAG_LOCK_OPERATION                    0x00000004
#define SCI_FLAG_CONDITIONAL_INTERRUPT_MAP         0x00000008
#define SCI_FLAG_UNCONDITIONAL_DATA_INTERRUPT_MAP  0x00000010
#define SCI_FLAG_IO_MAP_IOSPACE                    0x00000080
#define SCI_FLAG_NO_MEMORY_LOOPBACK                0x00000800
#define SCI_FLAG_SHARED_MAP                        0x00002000
#define SCI_FLAG_BROADCAST                         0x00040000

#define SCI_FLAG_PCIE_REQUESTER_GLOBAL             0x00000001

#define SCI_FLAG_CACHE_FLUSH                       0x00100000
#define SCI_FLAG_CACHE_INVALIDATE                  0x00200000


#define SCI_FLAG_EXCLUSIVE                         0x01000000
#define SCI_FLAG_DEVICE_SIDE_ONLY                  0x02001000
#define SCI_FLAG_LOCAL_ONLY                        0x02002000
#define SCI_FLAG_DEVICE_PREFER_BANDWIDTH           0x04000001
#define SCI_FLAG_HOST_PREFER_BANDWIDTH             0x04000002
#define SCI_FLAG_SPECIFY_PATH                      0x10000000
#define SCI_FLAG_SPECIFY_SLOT                      0x20000000

#define SCI_MEMTYPE_BAR                            0x00000001
#define SCI_MEMTYPE_PRIVATE                        0x00000002
#define SCI_MEMTYPE_SHARED                         0x00000004

#define SCI_MEMACCESS_DEVICE_READ                  0x00000001
#define SCI_MEMACCESS_DEVICE_WRITE                 0x00000002
#define SCI_MEMACCESS_HOST_READ                    0x00000004
#define SCI_MEMACCESS_HOST_WRITE                   0x00000008
#define SCI_MEMACCESS_MULTIHOST_READ               (0x00000010 | SCI_MEMACCESS_HOST_READ)
#define SCI_MEMACCESS_MULTIHOST_WRITE              (0x00000010 | SCI_MEMACCESS_HOST_WRITE)
#define SCI_MEMACCESS_P2P_READ                     (0x00000020 | SCI_MEMACCESS_DEVICE_READ)
#define SCI_MEMACCESS_P2P_WRITE                    (0x00000020 | SCI_MEMACCESS_DEVICE_WRITE)

#define SCI_FLAG_WRITE_BACK_CACHE_MAP              0x08000000
#define DIS_BROADCAST_NODEID_GROUP_ALL             252


ncclResult_t load_sisci(void);

ncclResult_t WrapSisciInitialize(unsigned int flags);
ncclResult_t WrapSisciGetLocalNodeId(unsigned int adapterNo,
                                     unsigned int *nodeId,
                                     unsigned int flags);
ncclResult_t WrapSisciOpen(sci_desc_t   *sd,
                           unsigned int flags);
ncclResult_t WrapSisciCreateDataInterrupt(sci_desc_t            sd,
                                          sci_local_data_interrupt_t *interrupt,
                                          unsigned int          localAdapterNo,
                                          unsigned int          *interruptNo,
                                          sci_cb_data_interrupt_t    callback,
                                          void                  *callbackArg,
                                          unsigned int          flags);
ncclResult_t WrapSisciConnectDataInterrupt(sci_desc_t                   sd,
                                           sci_remote_data_interrupt_t  *interrupt,
                                           unsigned int                 nodeId,
                                           unsigned int                 localAdapterNo,
                                           unsigned int                 interruptNo,
                                           unsigned int                 timeout,
                                           unsigned int                 flags);
ncclResult_t WrapSisciTriggerDataInterrupt(sci_remote_data_interrupt_t interrupt,
                                           void                        *data,
                                           unsigned int                length,
                                           unsigned int                flags);
ncclResult_t WrapSisciWaitForDataInterrupt(
                                           sci_local_data_interrupt_t interrupt,
                                           void                       *data,
                                           unsigned int               *length,
                                           unsigned int               timeout,
                                           unsigned int               flags);
ncclResult_t WrapSisciConnectSegment(sci_desc_t              sd,
                                     sci_remote_segment_t    *segment,
                                     unsigned int            nodeId,
                                     unsigned int            segmentId,
                                     unsigned int            localAdapterNo,
                                     sci_cb_remote_segment_t callback, 
                                     void                    *callbackArg,
                                     unsigned int            timeout,
                                     unsigned int            flags);
ncclResult_t WrapSisciMapRemoteSegment(sci_remote_segment_t segment,
                                       sci_map_t            *map,
                                       size_t               offset,
                                       size_t               size,
                                       volatile void        **addr,
                                       unsigned int         flags);
ncclResult_t WrapSisciCreateSegment(sci_desc_t             sd,
                                    sci_local_segment_t    *segment,
                                    unsigned int           segmentId,
                                    size_t                 size,
                                    sci_cb_local_segment_t callback,
                                    void                   *callbackArg, 
                                    unsigned int           flags);
ncclResult_t WrapSisciPrepareSegment(sci_local_segment_t segment,
                                     unsigned int        localAdapterNo,
                                     unsigned int        flags);
ncclResult_t WrapSisciSetSegmentAvailable(sci_local_segment_t segment,
                                          unsigned int        localAdapterNo,
                                          unsigned int        flags);
ncclResult_t WrapSisciMapLocalSegment(sci_local_segment_t segment,
                                      sci_map_t           *map,
                                      size_t              offset,
                                      size_t              size,
                                      volatile void                **addr,
                                      unsigned int        flags);
ncclResult_t WrapSisciAttachPhysicalMemory(sci_ioaddr_t         ioaddress,
                                           void                *address,
                                           unsigned int         busNo,
                                           size_t               size,
                                           sci_local_segment_t  segment,
                                           unsigned int         flags);

ncclResult_t WrapSisciRegisterSegmentMemory(void                *address,
                                            size_t              size,
                                            sci_local_segment_t segment,
                                            unsigned int        flags);

ncclResult_t WrapSisciTerminate(void);
ncclResult_t WrapSisciClose(sci_desc_t sd,
                            unsigned int flags);
ncclResult_t WrapSisciDisconnectSegment(sci_remote_segment_t segment,
                                        unsigned int         flags);
ncclResult_t WrapSisciRemoveSegment(sci_local_segment_t segment,
                                    unsigned int        flags);
ncclResult_t WrapSisciQuery(unsigned int command,
                            void         *data,
                            unsigned int flags);
ncclResult_t WrapSisciCreateDMAQueue(sci_desc_t      sd,
                                     sci_dma_queue_t *dq,
                                     unsigned int    localAdapterNo,
                                     unsigned int    maxEntries,
                                     unsigned int    flags);
ncclResult_t WrapSisciRemoveDMAQueue(sci_dma_queue_t dq,
                                     unsigned int    flags);
ncclResult_t WrapSisciDMAQueueState(sci_dma_queue_t dq,
                                    sci_dma_queue_state_t *state);
ncclResult_t WrapSisciRemoveDataInterrupt(sci_local_data_interrupt_t interrupt,
                                          unsigned int          flags);
ncclResult_t WrapSisciStartDmaTransfer(sci_dma_queue_t      dq,
                                       sci_local_segment_t  localSegment,
                                       sci_remote_segment_t remoteSegment,
                                       size_t               localOffset,
                                       size_t               size,
                                       size_t               remoteOffset,
                                       sci_cb_dma_t         callback,
                                       void                 *callbackArg,
                                       unsigned int         flags);

#endif //End include guard
