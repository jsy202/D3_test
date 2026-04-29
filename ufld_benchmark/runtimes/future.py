from __future__ import annotations


class FutureRuntimeBackend:
    def __init__(self, name: str) -> None:
        self.name = name

    def infer(self, *_args, **_kwargs):
        raise NotImplementedError(f"{self.name} is a future candidate and is not implemented in phase 1.")


class TnnOpenClBackend(FutureRuntimeBackend):
    def __init__(self) -> None:
        super().__init__("tnn_opencl")


class NcnnVulkanBackend(FutureRuntimeBackend):
    def __init__(self) -> None:
        super().__init__("ncnn_vulkan")


class MnnBackend(FutureRuntimeBackend):
    def __init__(self) -> None:
        super().__init__("mnn")
