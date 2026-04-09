import ctypes
import os
from pathlib import Path


def _default_library_candidates() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    candidates = []
    env = os.environ.get("JAKAL_RUNTIME_DLL")
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            root / "build_ninja" / "jakal_runtime.dll",
            root / "build_ninja" / "Debug" / "jakal_runtime.dll",
            root / "build" / "Debug" / "jakal_runtime.dll",
            root / "build" / "jakal_runtime.dll",
            root / "build_ninja" / "libjakal_runtime.so",
            root / "build" / "libjakal_runtime.so",
        ]
    )
    return candidates


def load_runtime_library() -> ctypes.CDLL:
    for candidate in _default_library_candidates():
        if candidate.exists():
            return ctypes.CDLL(str(candidate))
    raise FileNotFoundError("Unable to locate jakal_runtime shared library. Set JAKAL_RUNTIME_DLL.")


class RuntimeOptions(ctypes.Structure):
    _fields_ = [
        ("enable_host_probe", ctypes.c_int),
        ("enable_opencl_probe", ctypes.c_int),
        ("enable_level_zero_probe", ctypes.c_int),
        ("enable_vulkan_status", ctypes.c_int),
        ("enable_cuda_probe", ctypes.c_int),
        ("enable_rocm_probe", ctypes.c_int),
        ("prefer_level_zero_over_opencl", ctypes.c_int),
        ("eager_hardware_refresh", ctypes.c_int),
        ("install_root", ctypes.c_char_p),
        ("cache_path", ctypes.c_char_p),
        ("execution_cache_path", ctypes.c_char_p),
        ("telemetry_path", ctypes.c_char_p),
    ]


class RuntimePaths(ctypes.Structure):
    _fields_ = [
        ("install_root", ctypes.c_char * 260),
        ("writable_root", ctypes.c_char * 260),
        ("config_dir", ctypes.c_char * 260),
        ("cache_dir", ctypes.c_char * 260),
        ("logs_dir", ctypes.c_char * 260),
        ("telemetry_path", ctypes.c_char * 260),
        ("planner_cache_path", ctypes.c_char * 260),
        ("execution_cache_path", ctypes.c_char * 260),
        ("python_dir", ctypes.c_char * 260),
    ]


class BackendStatus(ctypes.Structure):
    _fields_ = [
        ("backend_name", ctypes.c_char * 32),
        ("device_uid", ctypes.c_char * 128),
        ("code", ctypes.c_char * 32),
        ("detail", ctypes.c_char * 256),
        ("enabled", ctypes.c_int),
        ("available", ctypes.c_int),
        ("direct_execution", ctypes.c_int),
        ("modeled_fallback", ctypes.c_int),
    ]


class Runtime:
    def __init__(self, options: RuntimeOptions | None = None):
        self._lib = load_runtime_library()
        self._lib.jakal_core_runtime_create.restype = ctypes.c_void_p
        self._lib.jakal_core_runtime_create_with_options.restype = ctypes.c_void_p
        self._lib.jakal_core_runtime_backend_status_count.restype = ctypes.c_size_t
        handle = (
            self._lib.jakal_core_runtime_create()
            if options is None
            else self._lib.jakal_core_runtime_create_with_options(ctypes.byref(options))
        )
        if not handle:
            raise RuntimeError("Failed to create Jakal runtime")
        self._handle = ctypes.c_void_p(handle)

    def __del__(self):
        if getattr(self, "_handle", None):
            self._lib.jakal_core_runtime_destroy(self._handle)
            self._handle = None

    def paths(self) -> dict[str, str]:
        paths = RuntimePaths()
        if self._lib.jakal_core_runtime_get_install_paths(self._handle, ctypes.byref(paths)) != 0:
            raise RuntimeError("Failed to query install paths")
        return {name: bytes(getattr(paths, name)).split(b"\0", 1)[0].decode() for name, _ in paths._fields_}

    def backend_statuses(self) -> list[dict[str, str | int]]:
        count = self._lib.jakal_core_runtime_backend_status_count(self._handle)
        statuses = []
        for index in range(count):
            item = BackendStatus()
            if self._lib.jakal_core_runtime_get_backend_status(self._handle, index, ctypes.byref(item)) == 0:
                statuses.append(
                    {
                        "backend_name": bytes(item.backend_name).split(b"\0", 1)[0].decode(),
                        "device_uid": bytes(item.device_uid).split(b"\0", 1)[0].decode(),
                        "code": bytes(item.code).split(b"\0", 1)[0].decode(),
                        "detail": bytes(item.detail).split(b"\0", 1)[0].decode(),
                        "enabled": int(item.enabled),
                        "available": int(item.available),
                        "direct_execution": int(item.direct_execution),
                        "modeled_fallback": int(item.modeled_fallback),
                    }
                )
        return statuses


def default_host_only_options() -> RuntimeOptions:
    return RuntimeOptions(1, 0, 0, 0, 0, 0, 1, 1, None, None, None, None)
