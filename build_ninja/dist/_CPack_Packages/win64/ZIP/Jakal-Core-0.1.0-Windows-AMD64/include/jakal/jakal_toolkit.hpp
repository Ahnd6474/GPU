#pragma once

#include "jakal/jakal_l0.hpp"

#include <optional>
#include <string>
#include <vector>

namespace jakal {

struct JakalToolkitVariant {
    JakalL0Binding binding;
    JakalL0LaunchTuning tuning;
    double toolkit_score = 0.0;
    bool executable = false;
    std::string rationale;
};

[[nodiscard]] bool jakal_variant_executes_directly(const JakalToolkitVariant& variant);

struct JakalToolkitIndexEntry {
    std::string device_uid;
    std::string graph_fingerprint;
    std::vector<JakalToolkitVariant> variants;
};

class JakalToolkit {
public:
    JakalToolkit();
    explicit JakalToolkit(std::vector<std::unique_ptr<IJakalL0Adapter>> adapters);

    [[nodiscard]] std::vector<JakalToolkitVariant> rank_variants(
        const HardwareGraph& graph,
        const JakalL0WorkloadTraits& workload = {}) const;

    [[nodiscard]] std::optional<JakalToolkitVariant> select_best(
        const HardwareGraph& graph,
        const JakalL0WorkloadTraits& workload = {}) const;

    [[nodiscard]] std::vector<JakalToolkitIndexEntry> build_index(
        const std::vector<HardwareGraph>& graphs,
        const JakalL0WorkloadTraits& workload = {}) const;

private:
    std::vector<std::unique_ptr<IJakalL0Adapter>> adapters_;
};

}  // namespace jakal

