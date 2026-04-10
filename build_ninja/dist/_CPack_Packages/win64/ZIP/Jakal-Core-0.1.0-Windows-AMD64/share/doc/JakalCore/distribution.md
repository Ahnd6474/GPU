# Distribution Notes

Jakal-Core now supports local installs and package generation through CMake and CPack.

## Installed layout

- `bin/`: runtime executables such as `jakal_bootstrap`, `jakal_core_cli`, and `launch-jakal-hardware-setup.cmd`
- `lib/`: `jakal_core` and exported CMake package files
- `include/`: public headers
- `share/jakal-core/install/`: install helper scripts and prerequisite manifest
- `share/jakal-core/install/prereqs/`: optional bundled prerequisite installers
- `share/jakal-core/update/`: update helper scripts
- `share/jakal-core/remove/`: uninstall helper scripts
- `share/doc/JakalCore/`: README, license, and distribution notes

## Installer flow

The packaged runtime now includes an install helper that can:

- probe CPU, Intel/NVIDIA/AMD, Vulkan, Level Zero, and OpenCL availability
- recommend a backend preset such as `cpu-only`, `intel-level-zero`, `vulkan-runtime`, `opencl-fallback`, or `auto`
- select optional prerequisite actions for Vulkan, Level Zero, or OpenCL support
- run `jakal_core_cli doctor` after configuration
- launch from the NSIS finish page or Start Menu through `launch-jakal-hardware-setup.cmd`

Typical usage after unpacking a package:

```powershell
powershell -ExecutionPolicy Bypass -File .\share\jakal-core\install\install-jakal-core.ps1
```

Installed Windows packages also expose:

```powershell
.\bin\launch-jakal-hardware-setup.cmd
```

If vendor runtime installers are bundled, place them under:

- `share/jakal-core/install/prereqs/vulkan-support/`
- `share/jakal-core/install/prereqs/intel-level-zero-runtime/`
- `share/jakal-core/install/prereqs/opencl-runtime/`

For automation:

```powershell
.\bin\jakal_core_cli.exe doctor --json
```

For NSIS package generation with a portable `makensis.exe`:

```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\build-nsis-package.ps1
```

## Package generation

Typical flow:

```powershell
cmake -S . -B build -DJAKAL_CORE_BUILD_TESTS=OFF
cmake --build build --config Release --target package
```

On Windows the default package generator is ZIP, with NSIS enabled automatically when `makensis` is available.

## Code signing

Code signing is opt-in and currently targets Windows executables. Configure these cache variables:

- `JAKAL_CORE_ENABLE_CODE_SIGNING=ON`
- `JAKAL_CORE_SIGNTOOL_PATH=...`
- `JAKAL_CORE_CODESIGN_CERT_SHA1=...`
- `JAKAL_CORE_CODESIGN_TIMESTAMP_URL=...`

The build will then sign installable executables after each successful build. A valid certificate is still required; the repository does not include one.
