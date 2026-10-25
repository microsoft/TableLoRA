import os
from pathlib import Path
import re
import stat


MAX_ADAPTERS = 32
MAX_WEIGHTS_BYTES = 512 * 1024 * 1024
WEIGHTS_NAME = "adapter_model.safetensors"


def _adapter_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", name):
        raise ValueError("Adapter names must be single ASCII identifiers")
    return name


def _subfolder(value):
    if value is None or value == "":
        return None
    if not isinstance(value, str) or len(value) > 512:
        raise ValueError("Invalid adapter subfolder")
    parts = value.split("/")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", part) for part in parts):
        raise ValueError("Adapter subfolders must be relative paths without traversal")
    return "/".join(parts)


def adapter_selection(adapter_name="default", subfolder=None, adapter_subfolders=None):
    _adapter_name(adapter_name)
    if adapter_subfolders is None:
        return {adapter_name: _subfolder(subfolder)}
    if subfolder is not None:
        raise ValueError("Use subfolder or adapter_subfolders, not both")
    if not isinstance(adapter_subfolders, dict) or not 1 <= len(adapter_subfolders) <= MAX_ADAPTERS:
        raise ValueError("adapter_subfolders must be a nonempty mapping of at most 32 adapters")
    if adapter_name not in adapter_subfolders:
        raise ValueError("adapter_subfolders must include the requested active adapter")
    ordered = {adapter_name: adapter_subfolders[adapter_name]}
    ordered.update(adapter_subfolders)
    return {_adapter_name(name): _subfolder(folder) for name, folder in ordered.items()}


def load_selection(names, subfolder=None, adapter_subfolders=None):
    single_name = isinstance(names, str)
    names = [names] if single_name else names
    if not isinstance(names, list) or not 1 <= len(names) <= MAX_ADAPTERS:
        raise ValueError("Explicit adapter names are required")
    names = [_adapter_name(name) for name in names]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate adapter names are not supported")
    if adapter_subfolders is not None:
        if not isinstance(adapter_subfolders, dict) or set(names) != set(adapter_subfolders):
            raise ValueError("adapter_subfolders must match the requested adapters exactly")
        return adapter_selection(names[0], subfolder, adapter_subfolders)
    if len(names) == 1 and (single_name or subfolder is not None):
        return adapter_selection(names[0], subfolder)
    if subfolder is not None:
        raise ValueError("Multiple adapters require explicit per-adapter subfolders")
    return adapter_selection(names[0], adapter_subfolders={name: None if name == "default" else name for name in names})


def require_safe_options(options):
    if options.get("use_safetensors", True) is not True:
        raise ValueError("Only safetensors adapter weights are supported")
    if options.get("trust_remote_code", False) is not False:
        raise ValueError("Remote adapter code is not supported")
    if options.get("adapters") is not None:
        raise ValueError("Nested adapter loading is not supported; use adapter_subfolders")


def _windows_handle_path(descriptor):
    import ctypes
    from ctypes import wintypes
    import msvcrt

    get_path = ctypes.WinDLL("kernel32", use_last_error=True).GetFinalPathNameByHandleW
    get_path.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    get_path.restype = wintypes.DWORD
    buffer = ctypes.create_unicode_buffer(512)
    handle = wintypes.HANDLE(msvcrt.get_osfhandle(descriptor))
    size = get_path(handle, buffer, len(buffer), 0)
    if size >= len(buffer):
        if size > 32768:
            raise ValueError("Adapter file path exceeds the platform limit")
        buffer = ctypes.create_unicode_buffer(size + 1)
        size = get_path(handle, buffer, len(buffer), 0)
    if not size or size >= len(buffer):
        raise ctypes.WinError(ctypes.get_last_error())
    path = buffer.value
    if path.startswith("\\\\?\\UNC\\"):
        path = "\\\\" + path[8:]
    elif path.startswith("\\\\?\\"):
        path = path[4:]
    return Path(path)


def _open_local_weights(root, candidate):
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    if os.name == "nt":
        descriptor = os.open(candidate, flags)
        try:
            _windows_handle_path(descriptor).relative_to(root)
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor
    if os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ValueError("Secure local adapter loading requires directory-descriptor or Windows handle-path support")
    relative = candidate.relative_to(root)
    directory = os.open(root.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in root.parts[1:] + relative.parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        return os.open(relative.name, flags | os.O_NOFOLLOW, dir_fd=directory)
    finally:
        os.close(directory)


def load_safe_weights(model_id, subfolder=None, device="cpu", **hub_options):
    subfolder = _subfolder(subfolder)
    require_safe_options(hub_options)
    local = Path(os.fspath(model_id)).expanduser()
    if local.exists():
        if not local.is_dir():
            raise ValueError("Local model_id must be a directory")
        root = local.resolve(strict=True)
        candidate = root / subfolder / WEIGHTS_NAME if subfolder else root / WEIGHTS_NAME
        try:
            candidate = candidate.resolve(strict=True)
            candidate.relative_to(root)
        except (OSError, ValueError) as error:
            raise ValueError("Adapter weights must exist inside model_id; pickle fallback is disabled") from error
        descriptor = _open_local_weights(root, candidate)
    else:
        if not isinstance(model_id, str) or model_id.startswith(("/", ".", "\\")) or "\\" in model_id or ":" in model_id:
            raise ValueError("Local adapter directory does not exist")
        from huggingface_hub import hf_hub_download
        allowed = {"revision", "cache_dir", "token", "local_files_only", "force_download", "proxies"}
        options = {key: value for key, value in hub_options.items() if key in allowed}
        if "token" not in options and hub_options.get("use_auth_token") is not None:
            options["token"] = hub_options["use_auth_token"]
        candidate = Path(hf_hub_download(model_id, filename=WEIGHTS_NAME, subfolder=subfolder, **options)).resolve(strict=True)
        descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as weights:
        metadata = os.fstat(weights.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_WEIGHTS_BYTES:
            raise ValueError("Adapter weights must be a regular safetensors file of at most 512 MiB")
        data = weights.read(MAX_WEIGHTS_BYTES + 1)
        if len(data) > MAX_WEIGHTS_BYTES:
            raise ValueError("Adapter weights exceed the size limit")
    from safetensors.torch import load
    tensors = load(data)
    if str(device) != "cpu":
        tensors = {name: tensor.to(device) for name, tensor in tensors.items()}
    return tensors