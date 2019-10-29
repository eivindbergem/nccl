#include "sisciwrap.h"
#include <sys/types.h>
#include <unistd.h>

#include <dlfcn.h>
#include "core.h"

#define SCI_ERROR sci_error

void (*SCIInternalInitialize)(unsigned int flags,
                              sci_error_t* error);
const char* (*SCIInternalGetErrorString)(sci_error_t error);
void (*SCIInternalGetLocalNodeId)(unsigned int adapterNo,
                                  unsigned int *nodeId,
                                  unsigned int flags,
                                  sci_error_t  *error);
void (*SCIInternalOpen)(sci_desc_t   *sd,
                        unsigned int flags,
                        sci_error_t  *error);
void (*SCIInternalCreateDataInterrupt)(sci_desc_t            sd,
                                       sci_local_data_interrupt_t *interrupt,
                                       unsigned int          localAdapterNo,
                                       unsigned int          *interruptNo,
                                       sci_cb_data_interrupt_t    callback,
                                       void                  *callbackArg,
                                       unsigned int          flags,
                                       sci_error_t           *error);
void (*SCIInternalConnectDataInterrupt)(sci_desc_t                   sd,
                                        sci_remote_data_interrupt_t  *interrupt,
                                        unsigned int                 nodeId,
                                        unsigned int                 localAdapterNo,
                                        unsigned int                 interruptNo,
                                        unsigned int                 timeout,
                                        unsigned int                 flags,
                                        sci_error_t                  *error);
void (*SCIInternalTriggerDataInterrupt)(sci_remote_data_interrupt_t interrupt,
                                        void                        *data,
                                        unsigned int                length,
                                        unsigned int                flags,
                                        sci_error_t                 *error);
void (*SCIInternalWaitForDataInterrupt)(
                                        sci_local_data_interrupt_t interrupt,
                                        void                       *data,
                                        unsigned int               *length,
                                        unsigned int               timeout,
                                        unsigned int               flags,
                                        sci_error_t                *error);
void (*SCIInternalConnectSegment)(sci_desc_t              sd,
                                  sci_remote_segment_t    *segment,
                                  unsigned int            nodeId,
                                  unsigned int            segmentId,
                                  unsigned int            localAdapterNo,
                                  sci_cb_remote_segment_t callback,
                                  void                    *callbackArg,
                                  unsigned int            timeout,
                                  unsigned int            flags,
                                  sci_error_t             *error);
volatile void* (*SCIInternalMapRemoteSegment)(sci_remote_segment_t segment,
                                              sci_map_t            *map,
                                              size_t               offset,
                                              size_t               size,
                                              void                 *addr,
                                              unsigned int         flags,
                                              sci_error_t          *error);
void (*SCIInternalCreateSegment)(sci_desc_t             sd,
                                 sci_local_segment_t    *segment,
                                 unsigned int           segmentId,
                                 size_t                 size,
                                 sci_cb_local_segment_t callback,
                                 void                   *callbackArg,
                                 unsigned int           flags,
                                 sci_error_t            *error);
void (*SCIInternalPrepareSegment)(sci_local_segment_t segment,
                                  unsigned int        localAdapterNo,
                                  unsigned int        flags,
                                  sci_error_t         *error);
void (*SCIInternalSetSegmentAvailable)(sci_local_segment_t segment,
                                       unsigned int        localAdapterNo,
                                       unsigned int        flags,
                                       sci_error_t         *error);
void* (*SCIInternalMapLocalSegment)(sci_local_segment_t segment,
                                    sci_map_t           *map,
                                    size_t              offset,
                                    size_t              size,
                                    void                *addr,
                                    unsigned int        flags,
                                    sci_error_t         *error);
void (*SCIInternalAttachPhysicalMemory)(sci_ioaddr_t         ioaddress,
                                        void                *address,
                                        unsigned int         busNo,
                                        size_t               size,
                                        sci_local_segment_t  segment,
                                        unsigned int         flags,
                                        sci_error_t         *error);
void (*SCIInternalRegisterSegmentMemory)(void                *address,
                                         size_t              size,
                                         sci_local_segment_t segment,
                                         unsigned int        flags,
                                         sci_error_t         *error);
void (*SCIInternalTerminate)(void);
void (*SCIInternalClose)(sci_desc_t sd,
                         unsigned int flags,
                         sci_error_t *error);
void (*SCIInternalDisconnectSegment)(sci_remote_segment_t segment,
                                     unsigned int         flags,
                                     sci_error_t          *error);
void (*SCIInternalRemoveSegment)(sci_local_segment_t segment,
                                 unsigned int        flags,
                                 sci_error_t         *error);
void (*SCIInternalQuery)(unsigned int command,
                         void         *data,
                         unsigned int flags,
                         sci_error_t  *error);
void (*SCIInternalCreateDMAQueue)(sci_desc_t      sd,
                                  sci_dma_queue_t *dq,
                                  unsigned int    localAdapterNo,
                                  unsigned int    maxEntries,
                                  unsigned int    flags,
                                  sci_error_t     *error);
void (*SCIInternalRemoveDMAQueue)(sci_dma_queue_t dq,
                                  unsigned int    flags,
                                  sci_error_t     *error);
sci_dma_queue_state_t (*SCIInternalDMAQueueState)(sci_dma_queue_t dq);
void (*SCIInternalRemoveDataInterrupt)(sci_local_data_interrupt_t interrupt,
                                       unsigned int          flags,
                                       sci_error_t           *error);
void (*SCIInternalStartDmaTransfer)(sci_dma_queue_t      dq,
                                    sci_local_segment_t  localSegment,
                                    sci_remote_segment_t remoteSegment,
                                    size_t               localOffset,
                                    size_t               size,
                                    size_t               remoteOffset,
                                    sci_cb_dma_t         callback,
                                    void                 *callbackArg,
                                    unsigned int         flags,
                                    sci_error_t          *error);


ncclResult_t load_sisci(void) {
    static void* handle = NULL;
    void* tmp;
    void** cast;

    handle = dlopen("libsisci.so", RTLD_NOW);
    if (!handle) {
        WARN("Failed to open libsisci.so");
        goto teardown;
    }

#define LOAD_SYM(handle, symbol, funcptr) do {           \
    cast = (void**)&funcptr;                             \
    tmp = dlsym(handle, symbol);                         \
    if (tmp == NULL) {                                   \
      WARN("dlvsym failed on %s - %s", symbol, dlerror());  \
      goto teardown;                                     \
    }                                                    \
    *cast = tmp;                                         \
  } while (0)

    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCIInitialize", SCIInternalInitialize);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCIGetErrorString", SCIInternalGetErrorString);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCIGetLocalNodeId", SCIInternalGetLocalNodeId);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCIOpen", SCIInternalOpen);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCICreateDataInterrupt", SCIInternalCreateDataInterrupt);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCIConnectDataInterrupt", SCIInternalConnectDataInterrupt);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCITriggerDataInterrupt", SCIInternalTriggerDataInterrupt);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCIWaitForDataInterrupt", SCIInternalWaitForDataInterrupt);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCIConnectSegment", SCIInternalConnectSegment);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCIMapRemoteSegment", SCIInternalMapRemoteSegment);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCICreateSegment", SCIInternalCreateSegment);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCIPrepareSegment", SCIInternalPrepareSegment);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCISetSegmentAvailable", SCIInternalSetSegmentAvailable);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCIMapLocalSegment", SCIInternalMapLocalSegment);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCIAttachPhysicalMemory", SCIInternalAttachPhysicalMemory);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCIRegisterSegmentMemory", SCIInternalRegisterSegmentMemory);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCITerminate", SCIInternalTerminate);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCIClose", SCIInternalClose);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCIDisconnectSegment", SCIInternalDisconnectSegment);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCIRemoveSegment", SCIInternalRemoveSegment);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCIQuery", SCIInternalQuery);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCICreateDMAQueue", SCIInternalCreateDMAQueue);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCIRemoveDMAQueue", SCIInternalRemoveDMAQueue);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCIDMAQueueState", SCIInternalDMAQueueState);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCIRemoveDataInterrupt", SCIInternalRemoveDataInterrupt);
    LOAD_SYM(handle, "_SISCI_PUBLIC_FUNC_ST_SCIStartDmaTransfer", SCIInternalStartDmaTransfer);

    return ncclSuccess;

 teardown:

    return ncclSystemError;
}

#define SCI(fn)                                 \
    ({                                          \
    sci_error_t  SCI_ERROR = SCI_ERR_OK;        \
    fn;                                         \
    handle_sisci_error(__FILE__, __LINE__, SCI_ERROR); \
    })

ncclResult_t handle_sisci_error(const char *filename, int lineno, sci_error_t error) {
  if (error != SCI_ERR_OK) {
      WARN("SISCI error at %s:%d: %s", filename, lineno,
           SCIInternalGetErrorString(error));
    return ncclInternalError;
  }

  return ncclSuccess;
}


ncclResult_t WrapSisciInitialize(unsigned int flags) {
    return SCI(SCIInternalInitialize(flags, &SCI_ERROR));
}

ncclResult_t WrapSisciGetLocalNodeId(unsigned int adapterNo,
                                     unsigned int *nodeId,
                                     unsigned int flags) {
    return SCI(SCIInternalGetLocalNodeId(adapterNo, nodeId, flags, &SCI_ERROR));
}

ncclResult_t WrapSisciOpen(sci_desc_t   *sd,
                           unsigned int flags) {
    return SCI(SCIInternalOpen(sd, flags, &SCI_ERROR));
}

ncclResult_t WrapSisciCreateDataInterrupt(sci_desc_t            sd,
                                          sci_local_data_interrupt_t *interrupt,
                                          unsigned int          localAdapterNo,
                                          unsigned int          *interruptNo,
                                          sci_cb_data_interrupt_t    callback,
                                          void                  *callbackArg,
                                          unsigned int          flags) {
    return SCI(SCIInternalCreateDataInterrupt(sd, interrupt, localAdapterNo,
                                              interruptNo, callback, callbackArg,
                                              flags, &SCI_ERROR));
}

ncclResult_t WrapSisciConnectDataInterrupt(sci_desc_t                   sd,
                                           sci_remote_data_interrupt_t  *interrupt,
                                           unsigned int                 nodeId,
                                           unsigned int                 localAdapterNo,
                                           unsigned int                 interruptNo,
                                           unsigned int                 timeout,
                                           unsigned int                 flags) {
    return SCI(SCIInternalConnectDataInterrupt(sd, interrupt, nodeId, localAdapterNo,
                                               interruptNo, timeout, flags, &SCI_ERROR));
}

ncclResult_t WrapSisciTriggerDataInterrupt(sci_remote_data_interrupt_t interrupt,
                                           void                        *data,
                                           unsigned int                length,
                                           unsigned int                flags) {
    return SCI(SCIInternalTriggerDataInterrupt(interrupt, data, length, flags,
                                               &SCI_ERROR));
}

ncclResult_t WrapSisciWaitForDataInterrupt(sci_local_data_interrupt_t interrupt,
                                           void                       *data,
                                           unsigned int               *length,
                                           unsigned int               timeout,
                                           unsigned int               flags) {
    return SCI(SCIInternalWaitForDataInterrupt(interrupt, data, length, timeout,
                                               flags, &SCI_ERROR));
}

ncclResult_t WrapSisciConnectSegment(sci_desc_t              sd,
                                     sci_remote_segment_t    *segment,
                                     unsigned int            nodeId,
                                     unsigned int            segmentId,
                                     unsigned int            localAdapterNo,
                                     sci_cb_remote_segment_t callback,
                                     void                    *callbackArg,
                                     unsigned int            timeout,
                                     unsigned int            flags) {
    return SCI(SCIInternalConnectSegment(sd, segment, nodeId, segmentId,
                                         localAdapterNo, callback, callbackArg,
                                         timeout, flags, &SCI_ERROR));
}

ncclResult_t WrapSisciMapRemoteSegment(sci_remote_segment_t segment,
                                       sci_map_t            *map,
                                       size_t               offset,
                                       size_t               size,
                                       volatile void        **addr,
                                       unsigned int         flags) {
    sci_error_t error;
    *addr = SCIInternalMapRemoteSegment(segment, map, offset, size,
                                        NULL, flags, &error);
    return handle_sisci_error(__FILE__, __LINE__, error);
}

ncclResult_t WrapSisciCreateSegment(sci_desc_t             sd,
                                    sci_local_segment_t    *segment,
                                    unsigned int           segmentId,
                                    size_t                 size,
                                    sci_cb_local_segment_t callback,
                                    void                   *callbackArg,
                                    unsigned int           flags) {
    return SCI(SCIInternalCreateSegment(sd, segment, segmentId, size,
                                        callback, callbackArg, flags,
                                        &SCI_ERROR));
}

ncclResult_t WrapSisciPrepareSegment(sci_local_segment_t segment,
                                     unsigned int        localAdapterNo,
                                     unsigned int        flags) {
    return SCI(SCIInternalPrepareSegment(segment, localAdapterNo, flags,
                                         &SCI_ERROR));
}

ncclResult_t WrapSisciSetSegmentAvailable(sci_local_segment_t segment,
                                          unsigned int        localAdapterNo,
                                          unsigned int        flags) {
    return SCI(SCIInternalSetSegmentAvailable(segment, localAdapterNo, flags,
                                              &SCI_ERROR));
}

ncclResult_t WrapSisciMapLocalSegment(sci_local_segment_t   segment,
                                      sci_map_t            *map,
                                      size_t                offset,
                                      size_t                size,
                                      volatile void       **addr,
                                      unsigned int        flags) {
    sci_error_t error;
    *addr = SCIInternalMapLocalSegment(segment, map, offset, size,
                                       NULL, flags, &error);
    return handle_sisci_error(__FILE__, __LINE__, error);
}

ncclResult_t WrapSisciAttachPhysicalMemory(sci_ioaddr_t         ioaddress,
                                           void                *address,
                                           unsigned int         busNo,
                                           size_t               size,
                                           sci_local_segment_t  segment,
                                           unsigned int         flags) {
    return SCI(SCIInternalAttachPhysicalMemory(ioaddress, address,
                                               busNo, size, segment,
                                               flags, &SCI_ERROR));
}

ncclResult_t WrapSisciRegisterSegmentMemory(void                *address,
                                            size_t              size,
                                            sci_local_segment_t segment,
                                            unsigned int        flags) {
    return SCI(SCIInternalRegisterSegmentMemory(address, size,
                                                segment, flags,
                                                &SCI_ERROR));
}

ncclResult_t WrapSisciTerminate(void) {
    return SCI(SCIInternalTerminate());
}

ncclResult_t WrapSisciClose(sci_desc_t sd,
                            unsigned int flags) {
    return SCI(SCIInternalClose(sd, flags, &SCI_ERROR));
}

ncclResult_t WrapSisciDisconnectSegment(sci_remote_segment_t segment,
                                        unsigned int         flags) {
    return SCI(SCIInternalDisconnectSegment(segment, flags, &SCI_ERROR));
}

ncclResult_t WrapSisciRemoveSegment(sci_local_segment_t segment,
                                    unsigned int        flags) {
    return SCI(SCIInternalRemoveSegment(segment, flags, &SCI_ERROR));
}

ncclResult_t WrapSisciQuery(unsigned int command,
                            void         *data,
                            unsigned int flags) {
    return SCI(SCIInternalQuery(command, data, flags, &SCI_ERROR));
}

ncclResult_t WrapSisciCreateDMAQueue(sci_desc_t      sd,
                                     sci_dma_queue_t *dq,
                                     unsigned int    localAdapterNo,
                                     unsigned int    maxEntries,
                                     unsigned int    flags) {
    return SCI(SCIInternalCreateDMAQueue(sd, dq, localAdapterNo,
                                         maxEntries, flags, &SCI_ERROR));
}

ncclResult_t WrapSisciRemoveDMAQueue(sci_dma_queue_t dq,
                                     unsigned int    flags) {
    return SCI(SCIInternalRemoveDMAQueue(dq, flags, &SCI_ERROR));
}

ncclResult_t WrapSisciDMAQueueState(sci_dma_queue_t dq,
                                    sci_dma_queue_state_t *state) {
    *state = SCIInternalDMAQueueState(dq);

    return ncclSuccess;
}

ncclResult_t WrapSisciRemoveDataInterrupt(sci_local_data_interrupt_t interrupt,
                                          unsigned int          flags) {
    return SCI(SCIInternalRemoveDataInterrupt(interrupt, flags, &SCI_ERROR));
}

ncclResult_t WrapSisciStartDmaTransfer(sci_dma_queue_t      dq,
                                       sci_local_segment_t  localSegment,
                                       sci_remote_segment_t remoteSegment,
                                       size_t               localOffset,
                                       size_t               size,
                                       size_t               remoteOffset,
                                       sci_cb_dma_t         callback,
                                       void                 *callbackArg,
                                       unsigned int         flags) {
    return SCI(SCIInternalStartDmaTransfer(dq, localSegment, remoteSegment,
                                            localOffset, size, remoteOffset,
                                            callback, callbackArg, flags, &SCI_ERROR));
}
