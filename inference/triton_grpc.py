"""gRPC client wrapper for NVIDIA Triton Inference Server.

Requires tritonclient[grpc] package.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

try:
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import InferenceServerException
    _GRPC_AVAILABLE = True
except ImportError:
    _GRPC_AVAILABLE = False
    grpcclient = None  # type: ignore
    InferenceServerException = Exception  # type: ignore

from config import get_settings


class TritonGRPCClient:
    """Thin wrapper around the Triton gRPC client."""

    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        if not _GRPC_AVAILABLE:
            raise RuntimeError(
                "tritonclient[grpc] is not installed. "
                "Run: pip install tritonclient[grpc]"
            )
        settings = get_settings()
        url = f"{host or settings.triton_grpc_host}:{port or settings.triton_grpc_port}"
        self._client = grpcclient.InferenceServerClient(url=url, verbose=False)

    def health(self) -> bool:
        try:
            return self._client.is_server_ready()
        except Exception:
            return False

    def infer(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        output_names: List[str],
        model_version: str = "1",
    ) -> Dict[str, np.ndarray]:
        """Run inference.

        inputs: {input_name: numpy_array}
        returns: {output_name: numpy_array}
        """
        triton_inputs = []
        for name, array in inputs.items():
            inp = grpcclient.InferInput(name, list(array.shape), _numpy_to_triton_dtype(array.dtype))
            inp.set_data_from_numpy(array)
            triton_inputs.append(inp)

        triton_outputs = [grpcclient.InferRequestedOutput(n) for n in output_names]

        response = self._client.infer(
            model_name=model_name,
            inputs=triton_inputs,
            outputs=triton_outputs,
            model_version=model_version,
        )

        return {name: response.as_numpy(name) for name in output_names}


def _numpy_to_triton_dtype(dtype: np.dtype) -> str:
    mapping = {
        np.float32: "FP32",
        np.float16: "FP16",
        np.float64: "FP64",
        np.int32: "INT32",
        np.int64: "INT64",
        np.uint8: "UINT8",
        np.int8: "INT8",
    }
    for np_type, triton_type in mapping.items():
        if np.dtype(np_type) == dtype:
            return triton_type
    return "FP32"
