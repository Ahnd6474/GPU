# Distribution Notes

Jakal-Core now supports local installs and package generation through CMake and CPack.

## Installed layout

- `bin/`: runtime executables such as `jakal_bootstrap`
- `lib/`: `jakal_core` and exported CMake package files
- `include/`: public headers
- `share/jakal-core/update/`: update helper scripts
- `share/jakal-core/remove/`: uninstall helper scripts
- `share/doc/JakalCore/`: README, license, and distribution notes

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
