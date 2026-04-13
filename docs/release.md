# Release Checklist

This repository now includes a GitHub Actions workflow at `.github/workflows/release.yml` that builds Windows release assets when you push a tag like `v0.1.0` or a prerelease tag like `v0.1.1-beta.1`.

## Before tagging

1. Update `project(... VERSION ...)` in `CMakeLists.txt` to the numeric core version you want to release.
2. If you are publishing a beta, keep `project(... VERSION ...)` on the numeric core version and put the prerelease suffix in the Git tag, for example `v0.1.1-beta.1`.
3. Review release-facing docs such as `README.md` and `docs/distribution.md`.
4. Make sure the working tree does not contain local build output or scratch files.
5. Commit the release changes.

## Create the draft release

Push a version tag:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

For a beta release:

```powershell
git tag v0.1.1-beta.1
git push origin v0.1.1-beta.1
```

You can also trigger the workflow manually from GitHub Actions with `workflow_dispatch` and pass `tag_name=v0.1.1-beta.1`.

The workflow will:

- build the ZIP package
- build the NSIS installer
- run a packaged-layout smoke test against the generated ZIP
- generate `.sha256` sidecars for both assets
- create a GitHub release with generated release notes
- mark the release as a prerelease when the tag contains a suffix such as `-beta.1`
- name the packaged assets with the same suffix, for example `Jakal-Core-0.1.1-beta.1-Windows-AMD64.zip`

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
