#include "simulation_ledger.h"

#include "util/util_log.h"

#include <algorithm>

namespace pacer {

    SimulationLedger::SimulationLedger(uint64_t firstSimulationId)
    : m_nextSimulationId(firstSimulationId),
      m_cpuWatermark(firstSimulationId - 1),
      m_gpuWatermark(firstSimulationId - 1) {
    }

    void SimulationLedger::activateEpoch(uint64_t accountingEpoch,
            uint64_t firstSimulationId) {
        std::lock_guard<dxvk::mutex> lock(m_mutex);

        m_simulations.clear();
        m_captures.clear();
        m_openCaptures.clear();
        m_activePublicationLeases.clear();
        m_endAssociations.clear();
        m_externalMappings.clear();
        m_threadOwnership.clear();
        m_nextSimulationId = firstSimulationId;
        m_cpuWatermark = firstSimulationId - 1;
        m_gpuWatermark = firstSimulationId - 1;
        m_activeEpoch = accountingEpoch;
        m_highestStartedExternalId = 0;
        m_epochActive = true;
        m_bypassPacing.store(false, std::memory_order_release);
    }

    void SimulationLedger::deactivateEpoch(uint64_t accountingEpoch) {
        std::lock_guard<dxvk::mutex> lock(m_mutex);

        if (!m_epochActive || m_activeEpoch != accountingEpoch)
            return;

        for (auto& entry : m_captures) {
            CaptureRecord& capture = entry.second;
            if (capture.state != CaptureState::Retired) {
                capture.state = CaptureState::Failed;
                capture.failureReason = CaptureFailureReason::EpochTransition;
            }
        }
        m_simulations.clear();
        m_captures.clear();
        m_openCaptures.clear();
        m_activePublicationLeases.clear();
        m_endAssociations.clear();
        m_externalMappings.clear();
        m_threadOwnership.clear();
        m_highestStartedExternalId = 0;
        m_epochActive = false;
        m_bypassPacing.store(true, std::memory_order_release);
    }

    SimulationRecord* SimulationLedger::findSimulationLocked(uint64_t simulationId) {
        auto entry = m_simulations.find(simulationId);
        return entry != m_simulations.end() ? &entry->second : nullptr;
    }

    CaptureRecord* SimulationLedger::findCaptureLocked(uint64_t captureGeneration) {
        auto entry = m_captures.find(captureGeneration);
        return entry != m_captures.end() ? &entry->second : nullptr;
    }

    void SimulationLedger::markTrackingFailureLocked(SimulationRecord* simulation) {
        if (simulation)
            simulation->trackingFailed = true;
        m_bypassPacing.store(true, std::memory_order_release);
    }

    void SimulationLedger::failCaptureLocked(CaptureRecord& capture,
            CaptureFailureReason reason) {
        if (capture.state == CaptureState::Retired)
            return;
        capture.state = CaptureState::Failed;
        if (capture.failureReason == CaptureFailureReason::None)
            capture.failureReason = reason;
        capture.submissionSealRequested = true;
        m_openCaptures.erase(capture.captureGeneration);
        SimulationRecord* simulation = findSimulationLocked(capture.simulationId);
        markTrackingFailureLocked(simulation);
        if (simulation)
            finalizeSubmissionSealLocked(*simulation, &capture);
    }

    void SimulationLedger::markUntrustedFrontierLocked(CaptureFailureReason reason) {
        for (auto& entry : m_simulations) {
            SimulationRecord& simulation = entry.second;
            if (simulation.simulationId > m_gpuWatermark)
                simulation.trackingFailed = true;
        }
        while (!m_openCaptures.empty()) {
            uint64_t generation = *m_openCaptures.begin();
            CaptureRecord* capture = findCaptureLocked(generation);
            if (capture)
                failCaptureLocked(*capture, reason);
            else
                m_openCaptures.erase(generation);
        }
        if (m_epochActive)
            m_bypassPacing.store(true, std::memory_order_release);
    }

    uint64_t SimulationLedger::beginSimulation(uint64_t accountingEpoch,
            uint64_t externalReflexId, uint32_t, time_point start) {
        std::lock_guard<dxvk::mutex> lock(m_mutex);
        if (!m_epochActive || accountingEpoch != m_activeEpoch || !externalReflexId)
            return 0;
        auto mapping = m_externalMappings.find(externalReflexId);

        if (mapping != m_externalMappings.end()) {
            SimulationRecord* existing = findSimulationLocked(mapping->second);
            if (existing && !existing->cpuSealed)
                return mapping->second;
            m_externalMappings.erase(mapping);
        }

        const uint64_t simulationId = m_nextSimulationId++;
        SimulationRecord& simulation = m_simulations[simulationId];
        simulation.simulationId = simulationId;
        simulation.accountingEpoch = accountingEpoch;
        simulation.externalReflexId = externalReflexId;
        simulation.start = start;
        m_externalMappings[externalReflexId] = simulationId;
        return simulationId;
    }

    void SimulationLedger::openRenderCapture(uint64_t accountingEpoch,
            uint64_t externalReflexId, uint32_t threadId, int32_t renderStart) {
        std::lock_guard<dxvk::mutex> lock(m_mutex);
        if (!m_epochActive || accountingEpoch != m_activeEpoch)
            return;
        if (!externalReflexId) {
            markUntrustedFrontierLocked(CaptureFailureReason::InvalidStart);
            return;
        }

        /* Frame marker IDs are device-level monotonic and each render START ID
         * is single-use within an accounting epoch. The high-watermark is the
         * permanent replay guard; heavyweight associations are only needed
         * while exact capture ownership can still be affected. */
        if (externalReflexId <= m_highestStartedExternalId) {
            markUntrustedFrontierLocked(CaptureFailureReason::DuplicateStart);
            return;
        }
        auto mapping = m_externalMappings.find(externalReflexId);
        SimulationRecord* simulation = mapping != m_externalMappings.end()
                ? findSimulationLocked(mapping->second) : nullptr;
        if (!simulation || simulation->accountingEpoch != accountingEpoch ||
                simulation->submissionsSealed || simulation->trackingFailed) {
            markUntrustedFrontierLocked(CaptureFailureReason::InvalidStart);
            return;
        }

        const uint64_t generation = m_nextCaptureGeneration++;
        CaptureRecord& capture = m_captures[generation];
        capture.accountingEpoch = accountingEpoch;
        capture.captureGeneration = generation;
        capture.simulationId = simulation->simulationId;
        capture.externalReflexId = externalReflexId;
        capture.originatingThreadId = threadId;
        m_highestStartedExternalId = externalReflexId;
        m_openCaptures.insert(generation);
        m_endAssociations.emplace(externalReflexId, EndAssociation{
                accountingEpoch, generation, simulation->simulationId,
                EndAssociationState::AwaitingEnd});
        simulation->renderCaptureOpened = true;
        if (!simulation->renderStart)
            simulation->renderStart = renderStart;
    }

    SimulationProgress SimulationLedger::closeRenderCapture(
            uint64_t accountingEpoch, uint64_t externalReflexId,
            uint32_t, int32_t renderEnd) {
        std::lock_guard<dxvk::mutex> lock(m_mutex);
        if (!m_epochActive || accountingEpoch != m_activeEpoch)
            return {};
        auto association = m_endAssociations.find(externalReflexId);
        if (association == m_endAssociations.end() ||
                association->second.accountingEpoch != accountingEpoch) {
            if (externalReflexId &&
                    externalReflexId <= m_highestStartedExternalId)
                return collectProgressLocked();
            markUntrustedFrontierLocked(CaptureFailureReason::InvalidEnd);
            return collectProgressLocked();
        }
        if (association->second.state == EndAssociationState::Consumed)
            return collectProgressLocked();

        CaptureRecord* capture = findCaptureLocked(
                association->second.captureGeneration);
        SimulationRecord* simulation = findSimulationLocked(
                association->second.simulationId);
        if (capture &&
                (capture->simulationId != association->second.simulationId ||
                 capture->externalReflexId != externalReflexId)) {
            markUntrustedFrontierLocked(CaptureFailureReason::InvalidEnd);
            return collectProgressLocked();
        }

        /* Present can close the exact capture before its END marker. Preserve
         * the association while publication leases keep ownership live; after
         * terminal reclamation the high-watermark handles the delayed END. */
        if (!capture || capture->state == CaptureState::Closing ||
                capture->state == CaptureState::Retired) {
            association->second.state = EndAssociationState::Consumed;
            if (simulation && !simulation->renderEnd)
                simulation->renderEnd = renderEnd;
            return collectProgressLocked();
        }
        if (!simulation || capture->state != CaptureState::Open) {
            markUntrustedFrontierLocked(CaptureFailureReason::InvalidEnd);
            return collectProgressLocked();
        }

        association->second.state = EndAssociationState::Consumed;
        capture->state = CaptureState::Closing;
        capture->submissionSealRequested = true;
        m_openCaptures.erase(capture->captureGeneration);
        if (!simulation->renderEnd)
            simulation->renderEnd = renderEnd;
        requestSubmissionSealLocked(*simulation, capture);
        return collectProgressLocked();
    }

    CaptureLeaseAcquisition SimulationLedger::acquireCaptureLease(
            uint64_t accountingEpoch) {
        std::lock_guard<dxvk::mutex> lock(m_mutex);
        CaptureLeaseAcquisition result;
        if (!m_epochActive || accountingEpoch != m_activeEpoch) {
            result.result = CaptureAcquireResult::StaleEpoch;
            return result;
        }
        if (m_openCaptures.empty()) {
            markUntrustedFrontierLocked(CaptureFailureReason::NoOpenCapture);
            result.result = CaptureAcquireResult::NoOpenCapture;
            return result;
        }
        if (m_openCaptures.size() != 1) {
            markUntrustedFrontierLocked(CaptureFailureReason::AmbiguousOpenCaptures);
            result.result = CaptureAcquireResult::AmbiguousOpenCaptures;
            return result;
        }

        CaptureRecord* capture = findCaptureLocked(*m_openCaptures.begin());
        if (!capture || capture->state != CaptureState::Open) {
            markUntrustedFrontierLocked(CaptureFailureReason::NoOpenCapture);
            result.result = CaptureAcquireResult::NoOpenCapture;
            return result;
        }
        CaptureToken token;
        token.accountingEpoch = accountingEpoch;
        token.captureGeneration = capture->captureGeneration;
        token.simulationId = capture->simulationId;
        token.publicationLeaseId = m_nextPublicationLeaseId++;
        token.externalReflexId = capture->externalReflexId;
        m_activePublicationLeases.emplace(token.publicationLeaseId, token);
        capture->inFlightPublications++;
        result.result = CaptureAcquireResult::Acquired;
        result.token = token;
        return result;
    }

    bool SimulationLedger::consumeLeaseLocked(const CaptureToken& token,
            CaptureRetireReason reason, bool commit, SubmitRecord* submit,
            void* commandQueue, uint64_t commandGeneration) {
        if (!m_epochActive || token.accountingEpoch != m_activeEpoch)
            return false;

        auto lease = m_activePublicationLeases.find(token.publicationLeaseId);
        if (lease == m_activePublicationLeases.end()) {
            markUntrustedFrontierLocked(CaptureFailureReason::TokenMismatch);
            return false;
        }
        const CaptureToken actual = lease->second;
        CaptureRecord* capture = findCaptureLocked(actual.captureGeneration);
        SimulationRecord* simulation = findSimulationLocked(actual.simulationId);
        const bool identityMatches = actual.accountingEpoch == token.accountingEpoch &&
                actual.captureGeneration == token.captureGeneration &&
                actual.simulationId == token.simulationId &&
                actual.publicationLeaseId == token.publicationLeaseId;

        if (!identityMatches) {
            if (capture)
                failCaptureLocked(*capture, CaptureFailureReason::TokenMismatch);
            markUntrustedFrontierLocked(CaptureFailureReason::TokenMismatch);
            commit = false;
        }

        bool accepted = false;
        if (commit && capture && simulation && identityMatches) {
            if (capture->state == CaptureState::Open ||
                    capture->state == CaptureState::Closing) {
                if (submit && (!submit->published || submit->completionAccounted)) {
                    *submit = {};
                    submit->commandQueue = commandQueue;
                    submit->commandGeneration = commandGeneration;
                    submit->simulationId = simulation->simulationId;
                    submit->accountingEpoch = token.accountingEpoch;
                    submit->captureGeneration = token.captureGeneration;
                    submit->published = true;
                    submit->completionAccounted = false;
                    /* Published ownership must exist before the final lease can
                     * make the deferred submission seal effective. */
                    simulation->publishedSubmits++;
                    accepted = true;
                } else {
                    failCaptureLocked(*capture,
                            CaptureFailureReason::SubmitSlotUnavailable);
                }
            } else if (capture->state == CaptureState::Retired) {
                markUntrustedFrontierLocked(CaptureFailureReason::CommitToRetired);
            }
            /* Failed captures deliberately reject pre-failure tokens. */
        }

        m_activePublicationLeases.erase(lease);
        if (capture) {
            if (capture->inFlightPublications)
                capture->inFlightPublications--;
            else
                markUntrustedFrontierLocked(CaptureFailureReason::TokenMismatch);
            if (simulation)
                finalizeSubmissionSealLocked(*simulation, capture);
        }
        cleanupLocked();
        return accepted;
    }

    bool SimulationLedger::commitCapturedSubmit(const CaptureToken& token,
            SubmitRecord& submit, void* commandQueue,
            uint64_t commandGeneration) {
        std::lock_guard<dxvk::mutex> lock(m_mutex);
        return consumeLeaseLocked(token, CaptureRetireReason::BenignAbort, true,
                &submit, commandQueue, commandGeneration);
    }

    SimulationProgress SimulationLedger::retireCaptureLease(
            const CaptureToken& token, CaptureRetireReason reason) {
        std::lock_guard<dxvk::mutex> lock(m_mutex);
        consumeLeaseLocked(token, reason, false, nullptr, nullptr, 0);
        return collectProgressLocked();
    }

    void SimulationLedger::requestSubmissionSealLocked(
            SimulationRecord& simulation, CaptureRecord* capture) {
        if (capture)
            capture->submissionSealRequested = true;
        finalizeSubmissionSealLocked(simulation, capture);
    }

    void SimulationLedger::finalizeSubmissionSealLocked(
            SimulationRecord& simulation, CaptureRecord* capture) {
        if (!capture || !capture->submissionSealRequested ||
                capture->inFlightPublications)
            return;
        if (!simulation.submissionsSealed)
            simulation.submissionsSealed = true;
        if (!simulation.publishedSubmits)
            markTrackingFailureLocked(&simulation);
        if (capture->state == CaptureState::Closing)
            capture->state = CaptureState::Retired;
    }

    void SimulationLedger::closeOpenCapturesForPresentLocked(
            SimulationRecord& simulation, bool failed,
            CaptureFailureReason reason) {
        bool found = false;
        std::vector<uint64_t> open(m_openCaptures.begin(), m_openCaptures.end());
        for (uint64_t generation : open) {
            CaptureRecord* capture = findCaptureLocked(generation);
            if (!capture || capture->simulationId != simulation.simulationId)
                continue;
            found = true;
            capture->cpuSealRequested = true;
            capture->submissionSealRequested = true;
            m_openCaptures.erase(generation);
            if (failed)
                failCaptureLocked(*capture, reason);
            else {
                capture->state = CaptureState::Closing;
                finalizeSubmissionSealLocked(simulation, capture);
            }
        }
        if (!found) {
            for (auto& entry : m_captures) {
                CaptureRecord& capture = entry.second;
                if (capture.simulationId == simulation.simulationId) {
                    found = true;
                    capture.cpuSealRequested = true;
                    capture.submissionSealRequested = true;
                    if (failed) {
                        if (capture.state == CaptureState::Retired) {
                            capture.state = CaptureState::Failed;
                            capture.failureReason = reason;
                            markTrackingFailureLocked(&simulation);
                        } else
                            failCaptureLocked(capture, reason);
                    }
                    else
                        finalizeSubmissionSealLocked(simulation, &capture);
                }
            }
        }
        if (!found || !simulation.renderCaptureOpened) {
            simulation.submissionsSealed = true;
            markTrackingFailureLocked(&simulation);
        }
    }

    SimulationProgress SimulationLedger::beginPresent(uint64_t accountingEpoch,
            uint64_t simulationId, uint32_t threadId,
            int32_t renderStart, int32_t renderEnd,
            PresentAttemptToken *attemptToken) {
        std::lock_guard<dxvk::mutex> lock(m_mutex);
        if (attemptToken)
            *attemptToken = {};
        if (!m_epochActive || accountingEpoch != m_activeEpoch)
            return {};
        SimulationRecord* simulation = findSimulationLocked(simulationId);
        ThreadPresentOwnership& ownership = m_threadOwnership[threadId];

        ownership.threadId = threadId;
        ownership.accountingEpoch = accountingEpoch;
        if (ownership.presentPending)
            cancelPresentLocked(ownership);
        if (!simulation) {
            markTrackingFailureLocked(nullptr);
            ownership.presentSimulationId = 0;
            return collectProgressLocked();
        }

        ownership.presentSimulationId = simulationId;
        ownership.presentAttemptGeneration = m_nextPresentAttemptGeneration++;
        ownership.presentPending = true;
        ownership.presentAccepted = false;
        if (!simulation->renderStart)
            simulation->renderStart = renderStart;
        if (!simulation->renderEnd)
            simulation->renderEnd = renderEnd;
        if (attemptToken) {
            attemptToken->accountingEpoch = accountingEpoch;
            attemptToken->simulationId = simulationId;
            attemptToken->attemptGeneration = ownership.presentAttemptGeneration;
            attemptToken->threadId = threadId;
        }
        return collectProgressLocked();
    }

    void SimulationLedger::cancelPresentLocked(ThreadPresentOwnership& ownership) {
        SimulationRecord* simulation = findSimulationLocked(
                ownership.presentSimulationId);
        if (simulation) {
            if (!simulation->cpuSealed) {
                simulation->cpuSealed = true;
                simulation->cpuFinished = dxvk::high_resolution_clock::now();
            }
            closeOpenCapturesForPresentLocked(*simulation, true,
                    CaptureFailureReason::PresentCancelled);
            markTrackingFailureLocked(simulation);
        } else {
            markTrackingFailureLocked(nullptr);
        }
        ownership.presentPending = false;
        ownership.presentAccepted = false;
        ownership.presentAttemptGeneration = 0;
    }

    SimulationProgress SimulationLedger::endPresent(uint64_t accountingEpoch,
            uint64_t simulationId, uint32_t threadId) {
        std::lock_guard<dxvk::mutex> lock(m_mutex);
        bool matched = false;
        if (!m_epochActive || accountingEpoch != m_activeEpoch)
            return {};
        auto ownership = m_threadOwnership.find(threadId);
        if (ownership != m_threadOwnership.end() &&
                ownership->second.accountingEpoch == accountingEpoch &&
                ownership->second.presentSimulationId == simulationId) {
            if (ownership->second.presentPending)
                cancelPresentLocked(ownership->second);
            ownership->second.presentSimulationId = 0;
            ownership->second.presentAttemptGeneration = 0;
            ownership->second.presentAccepted = false;
            matched = true;
        }
        if (!matched) {
            for (auto& entry : m_threadOwnership) {
                ThreadPresentOwnership& candidate = entry.second;
                if (candidate.accountingEpoch == accountingEpoch &&
                        candidate.presentSimulationId == simulationId) {
                    if (candidate.presentPending)
                        cancelPresentLocked(candidate);
                    candidate.presentSimulationId = 0;
                    candidate.presentAttemptGeneration = 0;
                    candidate.presentAccepted = false;
                }
            }
        }
        return collectProgressLocked();
    }

    SimulationProgress SimulationLedger::cancelPresent(
            const PresentAttemptToken& attemptToken) {
        std::lock_guard<dxvk::mutex> lock(m_mutex);
        if (!m_epochActive || attemptToken.accountingEpoch != m_activeEpoch)
            return {};
        auto ownership = m_threadOwnership.find(attemptToken.threadId);
        if (ownership != m_threadOwnership.end() &&
                ownership->second.accountingEpoch == attemptToken.accountingEpoch &&
                ownership->second.presentSimulationId == attemptToken.simulationId &&
                ownership->second.presentAttemptGeneration == attemptToken.attemptGeneration &&
                ownership->second.presentPending)
            cancelPresentLocked(ownership->second);
        else
            markTrackingFailureLocked(nullptr);
        return collectProgressLocked();
    }

    SubmitPublicationResult SimulationLedger::publishVulkanSubmit(SubmitRecord& submit,
            void* commandQueue, uint64_t commandGeneration, void* vulkanQueue,
            uint64_t vulkanGeneration, uint8_t telemetryQueueRole,
            uint64_t publicationCpuTimestampNs, bool publicationAvailable,
            void** rollbackVulkanQueue,
            uint64_t* rollbackVulkanGeneration,
            SubmitTelemetryMetadata* telemetryMetadata) {
        std::lock_guard<dxvk::mutex> lock(m_mutex);
        *rollbackVulkanQueue = nullptr;
        *rollbackVulkanGeneration = 0;
        if (telemetryMetadata)
            *telemetryMetadata = {};
        if (!submit.published || submit.commandQueue != commandQueue ||
                submit.commandGeneration != commandGeneration)
            return SubmitPublicationResult::NotCaptured;
        *rollbackVulkanQueue = submit.vulkanQueue;
        *rollbackVulkanGeneration = submit.vulkanGeneration;
        if (submit.completionAccounted ||
                submit.vulkanGeneration) {
            markTrackingFailureLocked(findSimulationLocked(submit.simulationId));
            return SubmitPublicationResult::Rejected;
        }
        if (!m_epochActive || submit.accountingEpoch != m_activeEpoch)
            return SubmitPublicationResult::Rejected;
        SimulationRecord* simulation = findSimulationLocked(submit.simulationId);
        if (!simulation) {
            markTrackingFailureLocked(nullptr);
            return SubmitPublicationResult::Rejected;
        }
        submit.vulkanQueue = vulkanQueue;
        submit.vulkanGeneration = vulkanGeneration;
        submit.telemetryQueueRole = telemetryQueueRole;
        submit.publicationCpuTimestampNs = publicationCpuTimestampNs;
        submit.telemetryPublicationAvailable = publicationAvailable;
        *rollbackVulkanQueue = vulkanQueue;
        *rollbackVulkanGeneration = vulkanGeneration;
        if (telemetryMetadata) {
            telemetryMetadata->accountingEpoch = submit.accountingEpoch;
            telemetryMetadata->simulationId = submit.simulationId;
            telemetryMetadata->captureGeneration = submit.captureGeneration;
            telemetryMetadata->queueRole = submit.telemetryQueueRole;
            telemetryMetadata->publicationCpuTimestampNs = submit.publicationCpuTimestampNs;
            telemetryMetadata->publicationAvailable = submit.telemetryPublicationAvailable;
        }
        return SubmitPublicationResult::Published;
    }

    bool SimulationLedger::ownsSubmit(const SubmitRecord& submit,
            void* commandQueue, uint64_t commandGeneration) const {
        std::lock_guard<dxvk::mutex> lock(m_mutex);
        return submit.published && submit.commandQueue == commandQueue &&
                submit.commandGeneration == commandGeneration;
    }

    SimulationProgress SimulationLedger::accountCompletion(SubmitRecord& submit,
            void* commandQueue, uint64_t commandGeneration, void* vulkanQueue,
            uint64_t vulkanGeneration, uint64_t gpuTimestamp,
            uint64_t gpuExecutionStart, bool gpuExecutionStartAvailable,
            SubmitCompletionResult* result,
            SubmitTelemetryMetadata* telemetryMetadata) {
        std::lock_guard<dxvk::mutex> lock(m_mutex);
        *result = SubmitCompletionResult::NotCaptured;
        if (telemetryMetadata)
            *telemetryMetadata = {};
        if (!submit.published || submit.commandQueue != commandQueue ||
                submit.commandGeneration != commandGeneration ||
                submit.vulkanQueue != vulkanQueue ||
                submit.vulkanGeneration != vulkanGeneration)
            return {};
        if (submit.completionAccounted) {
            *result = SubmitCompletionResult::Duplicate;
            return {};
        }
        *result = SubmitCompletionResult::Accepted;
        if (telemetryMetadata) {
            telemetryMetadata->accountingEpoch = submit.accountingEpoch;
            telemetryMetadata->simulationId = submit.simulationId;
            telemetryMetadata->captureGeneration = submit.captureGeneration;
            telemetryMetadata->queueRole = submit.telemetryQueueRole;
            telemetryMetadata->publicationCpuTimestampNs = submit.publicationCpuTimestampNs;
            telemetryMetadata->gpuExecutionStart = gpuExecutionStart;
            telemetryMetadata->publicationAvailable = submit.telemetryPublicationAvailable;
            telemetryMetadata->gpuExecutionStartAvailable = gpuExecutionStartAvailable;
        }
        submit.completionAccounted = true;
        submit.gpuTimestamp = gpuTimestamp;
        if (!m_epochActive || submit.accountingEpoch != m_activeEpoch)
            return {};
        SimulationRecord* simulation = findSimulationLocked(submit.simulationId);
        if (!simulation) {
            markTrackingFailureLocked(nullptr);
            return {};
        }
        simulation->accountedSubmits++;
        if (!gpuTimestamp)
            markTrackingFailureLocked(simulation);
        else {
            simulation->completedSubmits++;
            simulation->gpuTimestamp = std::max(simulation->gpuTimestamp, gpuTimestamp);
        }
        return collectProgressLocked();
    }

    SimulationProgress SimulationLedger::abandonSubmit(SubmitRecord& submit,
            void* commandQueue, uint64_t commandGeneration, void* vulkanQueue,
            uint64_t vulkanGeneration) {
        std::lock_guard<dxvk::mutex> lock(m_mutex);
        if (!submit.published || submit.completionAccounted ||
                submit.commandQueue != commandQueue ||
                submit.commandGeneration != commandGeneration)
            return {};
        if (submit.vulkanGeneration &&
                (submit.vulkanQueue != vulkanQueue ||
                submit.vulkanGeneration != vulkanGeneration))
            return {};
        submit.completionAccounted = true;
        if (!m_epochActive || submit.accountingEpoch != m_activeEpoch)
            return {};
        SimulationRecord* simulation = findSimulationLocked(submit.simulationId);
        if (simulation) {
            simulation->accountedSubmits++;
            markTrackingFailureLocked(simulation);
        } else
            markTrackingFailureLocked(nullptr);
        return collectProgressLocked();
    }

    SimulationProgress SimulationLedger::recordPresent(
            const PresentAttemptToken& attemptToken,
            void* swapchain, uint64_t sequence) {
        std::lock_guard<dxvk::mutex> lock(m_mutex);
        if (!m_epochActive || attemptToken.accountingEpoch != m_activeEpoch)
            return {};
        auto ownership = m_threadOwnership.find(attemptToken.threadId);
        SimulationRecord* simulation = ownership != m_threadOwnership.end() &&
                ownership->second.accountingEpoch == attemptToken.accountingEpoch &&
                ownership->second.presentSimulationId == attemptToken.simulationId &&
                ownership->second.presentAttemptGeneration == attemptToken.attemptGeneration &&
                ownership->second.presentPending
                ? findSimulationLocked(ownership->second.presentSimulationId) : nullptr;
        uint64_t& lastSequence = m_swapchainSequences[swapchain];
        if (!simulation || !sequence || sequence <= lastSequence) {
            if (simulation)
                cancelPresentLocked(ownership->second);
            else
                markTrackingFailureLocked(nullptr);
            return collectProgressLocked();
        }

        lastSequence = sequence;
        simulation->presentations.push_back({simulation->simulationId, swapchain, sequence});
        ownership->second.presentPending = false;
        ownership->second.presentAccepted = true;
        ownership->second.presentAttemptGeneration = 0;
        if (!simulation->cpuSealed) {
            simulation->cpuSealed = true;
            simulation->cpuFinished = dxvk::high_resolution_clock::now();
        }
        closeOpenCapturesForPresentLocked(*simulation, false,
                CaptureFailureReason::None);
        return collectProgressLocked();
    }

    void SimulationLedger::unregisterSwapchain(void* swapchain) {
        std::lock_guard<dxvk::mutex> lock(m_mutex);
        m_swapchainSequences.erase(swapchain);
    }

    SimulationProgress SimulationLedger::collectProgressLocked() {
        SimulationProgress progress;
        progress.accountingEpoch = m_activeEpoch;
        SimulationRecord* simulation;
        while ((simulation = findSimulationLocked(m_cpuWatermark + 1)) &&
                simulation->cpuSealed) {
            m_cpuWatermark++;
            progress.cpu.push_back({simulation->simulationId,
                    simulation->externalReflexId, simulation->start,
                    simulation->cpuFinished, simulation->renderStart,
                    simulation->renderEnd});
        }
        while ((simulation = findSimulationLocked(m_gpuWatermark + 1)) &&
                simulation->cpuSealed && simulation->submissionsSealed &&
                !simulation->trackingFailed && simulation->publishedSubmits &&
                simulation->completedSubmits == simulation->publishedSubmits) {
            m_gpuWatermark++;
            progress.gpu.push_back({simulation->simulationId,
                    simulation->gpuTimestamp, simulation->publishedSubmits,
                    simulation->completedSubmits});
        }
        cleanupLocked();
        return progress;
    }

    void SimulationLedger::cleanupLocked() {
        size_t retainedTerminalCaptures = 0;

        /* Once a capture is terminal and its final publication lease has
         * retired, a later exact or duplicate END cannot affect ownership.
         * The monotonic START high-watermark then classifies it as historical. */
        for (auto entry = m_endAssociations.begin();
                entry != m_endAssociations.end(); ) {
            CaptureRecord* capture = findCaptureLocked(
                    entry->second.captureGeneration);
            if (capture && !capture->inFlightPublications &&
                    (capture->state == CaptureState::Retired ||
                     capture->state == CaptureState::Failed))
                entry = m_endAssociations.erase(entry);
            else
                ++entry;
        }

        /* Submit completion is keyed entirely by the immutable SubmitRecord;
         * after the final publication lease retires, it no longer consults the
         * CaptureRecord. Keep only a small diagnostic tail of terminal records.
         * END replay protection lives separately in m_endAssociations. */
        for (auto entry = m_captures.rbegin(); entry != m_captures.rend(); ++entry) {
            const CaptureRecord& capture = entry->second;
            if (capture.state != CaptureState::Open &&
                    !capture.inFlightPublications)
                retainedTerminalCaptures++;
        }
        for (auto entry = m_captures.begin();
                entry != m_captures.end() &&
                retainedTerminalCaptures > MaxRetainedTerminalCaptures; ) {
            const CaptureRecord& capture = entry->second;
            if (capture.state == CaptureState::Open ||
                    capture.inFlightPublications) {
                ++entry;
                continue;
            }
            /* inFlightPublications is maintained with the authoritative lease
             * map, so zero also proves that no active lease refers here. */
            entry = m_captures.erase(entry);
            retainedTerminalCaptures--;
        }

        uint64_t retainFrom;
        if (m_bypassPacing.load(std::memory_order_relaxed))
            retainFrom = m_nextSimulationId > 64 ? m_nextSimulationId - 64 : 0;
        else {
            uint64_t completed = std::min(m_cpuWatermark, m_gpuWatermark);
            retainFrom = completed > 64 ? completed - 64 : 0;
        }
        for (auto entry = m_simulations.begin();
                entry != m_simulations.end() && entry->first < retainFrom; ) {
            SimulationRecord& simulation = entry->second;
            if (!simulation.cpuSealed || !simulation.submissionsSealed ||
                    simulation.accountedSubmits != simulation.publishedSubmits) {
                ++entry;
                continue;
            }
            auto mapping = m_externalMappings.find(simulation.externalReflexId);
            if (mapping != m_externalMappings.end() &&
                    mapping->second == simulation.simulationId)
                m_externalMappings.erase(mapping);
            entry = m_simulations.erase(entry);
        }
    }

    bool SimulationLedger::shouldBypassPacing() const {
        return m_bypassPacing.load(std::memory_order_acquire);
    }

    void SimulationLedger::forcePacingBypass() {
        m_bypassPacing.store(true, std::memory_order_release);
    }

#ifdef VKD3D_ENABLE_TEST_HOOKS
    bool SimulationLedger::testSubmitPending(const SubmitRecord& submit,
            void* commandQueue, uint64_t commandGeneration) const {
        std::lock_guard<dxvk::mutex> lock(m_mutex);
        return submit.published && !submit.completionAccounted &&
                submit.commandQueue == commandQueue &&
                submit.commandGeneration == commandGeneration;
    }

    bool SimulationLedger::testSubmitCompletionAccounted(
            const SubmitRecord& submit, void* commandQueue,
            uint64_t commandGeneration) const {
        std::lock_guard<dxvk::mutex> lock(m_mutex);
        return submit.published && submit.commandQueue == commandQueue &&
                submit.commandGeneration == commandGeneration &&
                submit.completionAccounted;
    }

    uint64_t SimulationLedger::testSubmitSimulation(const SubmitRecord& submit,
            void* commandQueue, uint64_t commandGeneration) const {
        std::lock_guard<dxvk::mutex> lock(m_mutex);
        return submit.published && submit.commandQueue == commandQueue &&
                submit.commandGeneration == commandGeneration
                ? submit.simulationId : 0;
    }

    uint32_t SimulationLedger::testPresentationCount(uint64_t simulationId) const {
        std::lock_guard<dxvk::mutex> lock(m_mutex);
        auto simulation = m_simulations.find(simulationId);
        return simulation != m_simulations.end()
                ? simulation->second.presentations.size() : 0;
    }

    CaptureSnapshot SimulationLedger::testCaptureSnapshot(
            uint64_t captureGeneration) const {
        std::lock_guard<dxvk::mutex> lock(m_mutex);
        CaptureSnapshot snapshot;
        auto capture = captureGeneration ? m_captures.find(captureGeneration)
                : (m_captures.empty() ? m_captures.end()
                                      : std::prev(m_captures.end()));
        snapshot.accountingEpoch = m_activeEpoch;
        snapshot.openCaptureCount = m_openCaptures.size();
        snapshot.activeLeaseCount = m_activePublicationLeases.size();
        snapshot.captureRecordCount = m_captures.size();
        snapshot.endAssociationCount = m_endAssociations.size();
        snapshot.highestStartedExternalId = m_highestStartedExternalId;
        if (capture == m_captures.end())
            return snapshot;
        const CaptureRecord& record = capture->second;
        snapshot.accountingEpoch = record.accountingEpoch;
        snapshot.simulationId = record.simulationId;
        snapshot.captureGeneration = record.captureGeneration;
        snapshot.state = record.state;
        snapshot.failureReason = record.failureReason;
        snapshot.inFlightPublications = record.inFlightPublications;
        snapshot.submissionSealRequested = record.submissionSealRequested;
        snapshot.cpuSealRequested = record.cpuSealRequested;
        auto simulation = m_simulations.find(record.simulationId);
        if (simulation != m_simulations.end()) {
            snapshot.submissionsSealed = simulation->second.submissionsSealed;
            snapshot.cpuSealed = simulation->second.cpuSealed;
            snapshot.publishedSubmits = simulation->second.publishedSubmits;
            snapshot.accountedSubmits = simulation->second.accountedSubmits;
            snapshot.completedSubmits = simulation->second.completedSubmits;
            snapshot.trackingFailed = simulation->second.trackingFailed;
        }
        return snapshot;
    }
#endif

}
