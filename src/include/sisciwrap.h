#ifndef NCCL_IBVWRAP_H_
#define NCCL_IBVWRAP_H_

#include <sisci_error.h>
#include <sisci_types.h>

#include "core.h"

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

#endif //End include guard
