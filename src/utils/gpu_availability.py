"""Hardware acceleration detection for model training backends."""

import logging
import shutil


class GpuAvailabilityChecker:
    """Detects CUDA and cuML availability, caching results after first probe."""

    _cuda_detected: bool | None = None
    _cuml_detected: bool | None = None

    def __init__(self) -> None:
        """Initializes the checker with a module-level logger."""
        self._logger = logging.getLogger(__name__)

    def is_cuda_available(self) -> bool:
        """Checks whether an NVIDIA GPU with CUDA drivers is accessible."""
        if GpuAvailabilityChecker._cuda_detected is not None:
            return GpuAvailabilityChecker._cuda_detected

        GpuAvailabilityChecker._cuda_detected = self._probe_cuda()
        return GpuAvailabilityChecker._cuda_detected

    def is_cuml_available(self) -> bool:
        """Checks whether cuML is installed AND the CUDA driver is sufficient."""
        if GpuAvailabilityChecker._cuml_detected is not None:
            return GpuAvailabilityChecker._cuml_detected

        try:
            import cuml  # type: ignore
            import cupy  # type: ignore

            # Try to get device count to verify driver sufficiency
            cupy.cuda.runtime.getDeviceCount()

            GpuAvailabilityChecker._cuml_detected = True
            self._logger.info("cuML (RAPIDS) detected and functional.")
        except ImportError:
            GpuAvailabilityChecker._cuml_detected = False
            self._logger.info("cuML not found. Using scikit-learn (CPU).")
        except Exception as e:
            GpuAvailabilityChecker._cuml_detected = False
            self._logger.warning(
                "cuML installed but NOT functional (Driver issue?): %s. Falling back to CPU.", e
            )
        return GpuAvailabilityChecker._cuml_detected

    def _probe_cuda(self) -> bool:
        """Attempts CUDA detection via PyTorch first, then nvidia-smi presence."""
        if self._try_torch_cuda():
            return True
        return self._try_nvidia_smi()

    def _try_torch_cuda(self) -> bool:
        """Probes CUDA via PyTorch if installed."""
        try:
            import torch  # type: ignore

            available = torch.cuda.is_available()
            if available:
                device_label = torch.cuda.get_device_name(0)
                self._logger.info("CUDA detected via PyTorch: %s", device_label)
            return available
        except ImportError:
            return False

    def _try_nvidia_smi(self) -> bool:
        """Falls back to checking nvidia-smi binary in PATH."""
        found = shutil.which("nvidia-smi") is not None
        if found:
            self._logger.info("CUDA detected via nvidia-smi presence.")
        else:
            self._logger.info("No CUDA detected. GPU models will fall back to CPU.")
        return found
