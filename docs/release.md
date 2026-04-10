# Release Checklist

This repository now includes a GitHub Actions workflow at `.github/workflows/release.yml` that builds Windows release assets when you push a tag like `v0.1.0`.

## Before tagging

1. Update `project(... VERSION ...)` in `CMakeLists.txt`.
2. Review release-facing docs such as `README.md` and `docs/distribution.md`.
3. Make sure the working tree does not contain local build output or scratch files.
4. Commit the release changes.

## Create the draft release

Push a version tag:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

The workflow will:

- build the ZIP package
- build the NSIS installer
- generate `.sha256` sidecars for both assets
- create a draft GitHub release with generated release notes

## Signed installer flow

The GitHub workflow intentionally publishes unsigned installers unless you replace them manually. If you want an Authenticode-signed NSIS installer:

```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\build-nsis-package.ps1 `
  -BuildDir build_ninja `
  -OutputDir .\build_ninja\dist-nsis `
  -CodeSignCertSha1 "<thumbprint>" `
  -SignToolPath "C:\Program Files (x86)\Windows Kits\10\App Certification Kit\signtool.exe"
```

That command signs the installer, verifies the signature, and writes the adjacent checksum file. Upload the signed `.exe` and `.exe.sha256` to the draft release before publishing it.
