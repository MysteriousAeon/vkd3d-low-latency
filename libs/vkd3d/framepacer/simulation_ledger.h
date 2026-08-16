#pragma once

#include "util/util_time.h"
#include "util/thread.h"
#include "telemetry.h"

#include <atomic>
#include <cstdint>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

namespace pacer {

    struct PresentationRecord {
        uint64_t simulationId = 0;
        void* swapchain = nullptr;
        uint64_t sequence = 0;
    };

    struct PresentAttemptToken {
        uint64_t accountingEpoch = 0;
        uint64_t simulationId = 0;
        uint64_t attemptGeneration = 0;
        uint32_t threadId = 0;

        explicit operator bool() const {
            return accountingEpoch && attemptGeneration;
        }
    };

    enum class CaptureState : uint32_t {
        Open,
        Closing,
        Failed,
        Retired,
    };

    enum class CaptureFailureReason : uint32_t {
        None,
        InvalidStart,
        DuplicateStart,
        InvalidEnd,
        NoOpenCapture,
        AmbiguousOpenCaptures,
        PresentWithoutCapture,
        PresentCancelled,
        TokenMismatch,
        SubmitSlotUnavailable,
        CommitToRetired,
        EpochTransition,
    };

    struct CaptureToken {
        uint64_t accountingEpoch = 0;
        uint64_t captureGeneration = 0;
        uint64_t simulationId = 0;
        uint64_t publicationLeaseId = 0;
        uint64_t externalReflexId = 0;

        explicit operator bool() const {
            return accountingEpoch && captureGeneration && publicationLeaseId;
        }
    };

    enum class CaptureAcquireResult : uint32_t {
        NotReflex,
        Acquired,
        NoOpenCapture,
        AmbiguousOpenCaptures,
        StaleEpoch,
    };

    enum class CaptureRetireReason : uint32_t {
        BenignAbort,
    };

    struct CaptureLeaseAcquisition {
        CaptureAcquireResult result = CaptureAcquireResult::NotReflex;
        CaptureToken token;
    };

    struct CaptureRecord {
        uint64_t accountingEpoch = 0;
        uint64_t captureGeneration = 0;
        uint64_t simulationId = 0;
        uint64_t externalReflexId = 0;
        uint32_t originatingThreadId = 0;
        CaptureState state = CaptureState::Open;
        CaptureFailureReason failureReason = CaptureFailureReason::None;
        uint32_t inFlightPublications = 0;
        bool submissionSealRequested = false;
        bool cpuSealRequested = false;
    };

    struct SubmitRecord {
        void* commandQueue = nullptr;
        void* vulkanQueue = nullptr;
        uint64_t commandGeneration = 0;
        uint64_t vulkanGeneration = 0;
        uint64_t simulationId = 0;
        uint64_t accountingEpoch = 0;
        uint64_t captureGeneration = 0;
        uint64_t gpuTimestamp = 0;
        /* Captured at Vulkan-submit publication and retained with the exact
         * ledger generation until its accepted completion is accounted. */
        uint64_t publicationCpuTimestampNs = 0;
        uint8_t telemetryQueueRole = 0;
        bool telemetryPublicationAvailable = false;
        bool published = false;
        bool completionAccounted = true;
    };

    struct SubmitTelemetryMetadata {
        uint64_t accountingEpoch = 0;
        uint64_t simulationId = 0;
        uint64_t captureGeneration = 0;
        uint64_t publicationCpuTimestampNs = 0;
        uint64_t gpuExecutionStart = 0;
        uint8_t queueRole = 0;
        bool publicationAvailable = false;
        bool gpuExecutionStartAvailable = false;
    };

    enum class SubmitPublicationResult : uint32_t {
        NotCaptured,
        Published,
        Rejected,
    };

    enum class SubmitCompletionResult : uint32_t {
        NotCaptured,
        Accepted,
        Duplicate,
    };

    struct SimulationRecord {
        using time_point = dxvk::high_resolution_clock::time_point;

        uint64_t simulationId = 0;
        uint64_t accountingEpoch = 0;
        uint64_t externalReflexId = 0;
        time_point start = {};
        time_point cpuFinished = {};
        int32_t renderStart = 0;
        int32_t renderEnd = 0;
        uint64_t gpuTimestamp = 0;
        uint32_t publishedSubmits = 0;
        uint32_t accountedSubmits = 0;
        uint32_t completedSubmits = 0;
        bool cpuSealed = false;
        bool submissionsSealed = false;
        bool trackingFailed = false;
        bool renderCaptureOpened = false;
        std::vector<PresentationRecord> presentations;
    };

    /* Keep the render-START rejection classification independent from the
     * persistent-bypass path which consumes it. This also gives the
     * source-invariant-impossible states a narrow deterministic unit surface. */
    inline telemetry::InvalidRenderStartSubreason classifyInvalidRenderStart(
            bool mappingPresent, const SimulationRecord* simulation,
            uint64_t accountingEpoch) {
        if (!mappingPresent)
            return telemetry::InvalidRenderStartSubreason::MappingAbsent;
        if (!simulation)
            return telemetry::InvalidRenderStartSubreason::MappingTargetMissing;
        if (simulation->accountingEpoch != accountingEpoch)
            return telemetry::InvalidRenderStartSubreason::MappedWrongEpoch;
        if (simulation->submissionsSealed)
            return telemetry::InvalidRenderStartSubreason::MappedSubmissionsSealed;
        if (simulation->trackingFailed)
            return telemetry::InvalidRenderStartSubreason::MappedTrackingFailed;
        return telemetry::InvalidRenderStartSubreason::None;
    }

    struct ThreadPresentOwnership {
        uint32_t threadId = 0;
        uint64_t accountingEpoch = 0;
        uint64_t presentSimulationId = 0;
        uint64_t presentAttemptGeneration = 0;
        bool presentPending = false;
        bool presentAccepted = false;
    };

    struct SimulationProgress {
        uint64_t accountingEpoch = 0;

        struct CpuCompletion {
            uint64_t simulationId;
            uint64_t externalReflexId;
            dxvk::high_resolution_clock::time_point start;
            dxvk::high_resolution_clock::time_point cpuFinished;
            int32_t renderStart;
            int32_t renderEnd;
        };

        struct GpuCompletion {
            uint64_t simulationId;
            uint64_t gpuTimestamp;
            uint32_t publishedSubmits;
            uint32_t completedSubmits;
        };

        std::vector<CpuCompletion> cpu;
        std::vector<GpuCompletion> gpu;
    };

    /* Fixed-size first-failure context. The common telemetry Event keeps these
     * fields generic so ordinary tracking paths do not need bespoke snapshots. */
    struct FirstFailureContext {
        uint64_t simulationId = 0;
        uint64_t captureGeneration = 0;
        uint64_t externalReflexId = 0;
        uint64_t contextId = 0;
        uint64_t contextValue0 = 0;
        uint64_t contextValue1 = 0;
        uint64_t contextValue2 = 0;
        uint32_t contextCount0 = 0;
        uint32_t contextCount1 = 0;
        uint32_t flags = 0;
        telemetry::InvalidRenderStartSubreason invalidRenderStartSubreason =
                telemetry::InvalidRenderStartSubreason::None;
        /* Diagnostic-only identity of the exact RENDERSUBMIT_START invocation
         * that entered openRenderCapture(). Zero means unavailable. */
        uint64_t originatingMarkerSerializationSequence = 0;
    };

#ifdef VKD3D_ENABLE_TEST_HOOKS
    struct CaptureSnapshot {
        uint64_t accountingEpoch = 0;
        uint64_t simulationId = 0;
        uint64_t captureGeneration = 0;
        CaptureState state = CaptureState::Retired;
        CaptureFailureReason failureReason = CaptureFailureReason::None;
        uint32_t inFlightPublications = 0;
        bool submissionSealRequested = false;
        bool cpuSealRequested = false;
        bool submissionsSealed = false;
        bool cpuSealed = false;
        uint32_t publishedSubmits = 0;
        uint32_t accountedSubmits = 0;
        uint32_t completedSubmits = 0;
        bool trackingFailed = false;
        uint32_t openCaptureCount = 0;
        uint32_t activeLeaseCount = 0;
        uint32_t captureRecordCount = 0;
        uint32_t endAssociationCount = 0;
        uint64_t highestStartedExternalId = 0;
    };
#endif

    class SimulationLedger {
    public:
        using time_point = dxvk::high_resolution_clock::time_point;

        explicit SimulationLedger(uint64_t firstSimulationId);

        void activateEpoch(uint64_t accountingEpoch, uint64_t firstSimulationId);
        void deactivateEpoch(uint64_t accountingEpoch);

        uint64_t beginSimulation(uint64_t accountingEpoch,
                uint64_t externalReflexId, uint32_t threadId,
                time_point start);
        void openRenderCapture(uint64_t accountingEpoch,
                uint64_t externalReflexId, uint32_t threadId,
                int32_t renderStart,
                uint64_t originatingMarkerSerializationSequence = 0);
        SimulationProgress closeRenderCapture(uint64_t accountingEpoch,
                uint64_t externalReflexId, uint32_t threadId,
                int32_t renderEnd);
        SimulationProgress beginPresent(uint64_t accountingEpoch,
                uint64_t simulationId, uint32_t threadId,
                int32_t renderStart, int32_t renderEnd,
                PresentAttemptToken *attemptToken);
        SimulationProgress endPresent(uint64_t accountingEpoch,
                uint64_t simulationId, uint32_t threadId);
        SimulationProgress cancelPresent(const PresentAttemptToken& attemptToken);

        CaptureLeaseAcquisition acquireCaptureLease(uint64_t accountingEpoch);
        bool commitCapturedSubmit(const CaptureToken& token, SubmitRecord& submit,
                void* commandQueue, uint64_t commandGeneration);
        SimulationProgress retireCaptureLease(const CaptureToken& token,
                CaptureRetireReason reason);
        SubmitPublicationResult publishVulkanSubmit(SubmitRecord& submit, void* commandQueue,
                uint64_t commandGeneration, void* vulkanQueue,
                uint64_t vulkanGeneration, uint8_t telemetryQueueRole,
                uint64_t publicationCpuTimestampNs, bool publicationAvailable,
                void** rollbackVulkanQueue,
                uint64_t* rollbackVulkanGeneration,
                SubmitTelemetryMetadata* telemetryMetadata);
        bool ownsSubmit(const SubmitRecord& submit, void* commandQueue,
                uint64_t commandGeneration) const;
        SimulationProgress accountCompletion(SubmitRecord& submit,
                void* commandQueue, uint64_t commandGeneration,
                void* vulkanQueue, uint64_t vulkanGeneration,
                uint64_t gpuTimestamp, uint64_t gpuExecutionStart,
                bool gpuExecutionStartAvailable, SubmitCompletionResult* result,
                SubmitTelemetryMetadata* telemetryMetadata);
        SimulationProgress abandonSubmit(SubmitRecord& submit,
                void* commandQueue, uint64_t commandGeneration,
                void* vulkanQueue, uint64_t vulkanGeneration);
        SimulationProgress recordPresent(const PresentAttemptToken& attemptToken,
                void* swapchain, uint64_t sequence);
        void unregisterSwapchain(void* swapchain);

        void setTelemetryDeviceId(uint64_t deviceId);

        bool shouldBypassPacing() const;
        /* This path serializes its context snapshot with normal ledger work,
         * then the CAS determines the sole first-failure owner. */
        bool forcePacingBypass(telemetry::FailureReason reason =
                telemetry::FailureReason::ExplicitForceOrOther,
                const FirstFailureContext& context = {});

#ifdef VKD3D_ENABLE_TEST_HOOKS
        bool testSubmitPending(const SubmitRecord& submit, void* commandQueue,
                uint64_t commandGeneration) const;
        bool testSubmitCompletionAccounted(const SubmitRecord& submit,
                void* commandQueue, uint64_t commandGeneration) const;
        uint64_t testSubmitSimulation(const SubmitRecord& submit, void* commandQueue,
                uint64_t commandGeneration) const;
        uint32_t testPresentationCount(uint64_t simulationId) const;
        CaptureSnapshot testCaptureSnapshot(uint64_t captureGeneration = 0) const;
#endif

    private:
        static constexpr size_t MaxRetainedTerminalCaptures = 64;
        enum class EndAssociationState : uint32_t { AwaitingEnd, Consumed };
        struct EndAssociation {
            uint64_t accountingEpoch = 0;
            uint64_t captureGeneration = 0;
            uint64_t simulationId = 0;
            EndAssociationState state = EndAssociationState::AwaitingEnd;
        };

        SimulationRecord* findSimulationLocked(uint64_t simulationId);
        CaptureRecord* findCaptureLocked(uint64_t captureGeneration);
        FirstFailureContext contextForSimulation(const SimulationRecord* simulation) const;
        FirstFailureContext contextForCapture(const CaptureRecord* capture) const;
        telemetry::FailureReason firstFailureReason(CaptureFailureReason reason) const;
        bool claimPacingBypassLocked(telemetry::FailureReason reason,
                const FirstFailureContext& context);
        void markTrackingFailureLocked(SimulationRecord* simulation,
                telemetry::FailureReason reason =
                        telemetry::FailureReason::ExplicitForceOrOther,
                const FirstFailureContext& context = {}, bool exactContext = false);
        void markUntrustedFrontierLocked(CaptureFailureReason reason,
                const FirstFailureContext& context = {});
        void failCaptureLocked(CaptureRecord& capture, CaptureFailureReason reason);
        void failCaptureLocked(CaptureRecord& capture, CaptureFailureReason reason,
                const FirstFailureContext& context);
        void cancelPresentLocked(ThreadPresentOwnership& ownership);
        void requestSubmissionSealLocked(SimulationRecord& simulation,
                CaptureRecord* capture);
        void finalizeSubmissionSealLocked(SimulationRecord& simulation,
                CaptureRecord* capture);
        void closeOpenCapturesForPresentLocked(SimulationRecord& simulation,
                bool failed, CaptureFailureReason reason);
        bool consumeLeaseLocked(const CaptureToken& token,
                CaptureRetireReason reason, bool commit,
                SubmitRecord* submit, void* commandQueue,
                uint64_t commandGeneration);
        SimulationProgress collectProgressLocked();
        void cleanupLocked();

        mutable dxvk::mutex m_mutex;
        std::map<uint64_t, SimulationRecord> m_simulations;
        std::map<uint64_t, CaptureRecord> m_captures;
        std::set<uint64_t> m_openCaptures;
        std::unordered_map<uint64_t, CaptureToken> m_activePublicationLeases;
        std::unordered_map<uint64_t, EndAssociation> m_endAssociations;
        std::unordered_map<uint64_t, uint64_t> m_externalMappings;
        std::unordered_map<uint32_t, ThreadPresentOwnership> m_threadOwnership;
        std::unordered_map<void*, uint64_t> m_swapchainSequences;
        uint64_t m_nextSimulationId;
        uint64_t m_cpuWatermark;
        uint64_t m_gpuWatermark;
        uint64_t m_nextCaptureGeneration = 1;
        uint64_t m_nextPublicationLeaseId = 1;
        uint64_t m_nextPresentAttemptGeneration = 1;
        uint64_t m_activeEpoch = 0;
        uint64_t m_highestStartedExternalId = 0;
        uint64_t m_telemetryDeviceId = 0;
        bool m_epochActive = false;
        std::atomic<bool> m_bypassPacing = { false };
    };

}
