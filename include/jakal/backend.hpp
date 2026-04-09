#pragma once

#include "jakal/device.hpp"

#include <memory>
#include <string>
#include <vector>

namespace jakal {

class IDeviceProbe {
public:
    virtual ~IDeviceProbe() = default;

    [[nodiscard]] virtual std::string name() const = 0;
    [[nodiscard]] virtual bool available() const = 0;
    virtual std::vector<HardwareGraph> discover_hardware() = 0;
};

std::unique_ptr<IDeviceProbe> make_host_probe();
std::unique_ptr<IDeviceProbe> make_opencl_probe();
std::unique_ptr<IDeviceProbe> make_level_zero_probe();
std::unique_ptr<IDeviceProbe> make_vulkan_probe();
std::unique_ptr<IDeviceProbe> make_cuda_probe();
std::unique_ptr<IDeviceProbe> make_rocm_probe();

}  // namespace jakal

