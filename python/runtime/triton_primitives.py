from __future__ import annotations


def ensure_triton_runtime_available() -> None:
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "primitive_runtime_backend=triton requires the primitive-triton CUDA env with a working Triton install"
        ) from exc

    raise NotImplementedError(
        "primitive_runtime_backend=triton is not implemented yet; use env_kind=primitive-triton with primitive_runtime_backend=torch until the first Triton scan kernel lands"
    )
