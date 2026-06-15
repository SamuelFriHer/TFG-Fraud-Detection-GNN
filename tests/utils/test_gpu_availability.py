"""Unit tests for the GpuAvailabilityChecker class."""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from src.utils.gpu_availability import GpuAvailabilityChecker


@pytest.fixture(autouse=True)
def reset_gpu_cache() -> Generator[None, None, None]:
    """Resets the cached GPU availability before and after each test."""
    GpuAvailabilityChecker._cuda_detected = None
    GpuAvailabilityChecker._cuml_detected = None
    yield
    GpuAvailabilityChecker._cuda_detected = None
    GpuAvailabilityChecker._cuml_detected = None


@patch.object(GpuAvailabilityChecker, "_try_torch_cuda", return_value=False)
@patch("shutil.which", return_value="/usr/bin/nvidia-smi")
def test_cuda_detected_via_nvidia_smi(mock_which: MagicMock, mock_try_torch: MagicMock) -> None:
    """Verifies CUDA is detected via nvidia-smi if PyTorch detection fails."""
    checker: GpuAvailabilityChecker = GpuAvailabilityChecker()
    assert checker.is_cuda_available() is True
    mock_which.assert_called_once_with("nvidia-smi")


@patch.object(GpuAvailabilityChecker, "_try_torch_cuda", return_value=False)
@patch("shutil.which", return_value=None)
def test_cuda_not_detected_when_nvidia_smi_missing(
    mock_which: MagicMock, mock_try_torch: MagicMock
) -> None:
    """Verifies CUDA is not detected if PyTorch and nvidia-smi check fail."""
    checker: GpuAvailabilityChecker = GpuAvailabilityChecker()
    assert checker.is_cuda_available() is False
    mock_which.assert_called_once_with("nvidia-smi")


@patch.object(GpuAvailabilityChecker, "_try_torch_cuda", return_value=True)
@patch("shutil.which", return_value=None)
def test_cuda_detected_via_torch(mock_which: MagicMock, mock_try_torch: MagicMock) -> None:
    """Verifies PyTorch detection takes precedence and nvidia-smi is not checked."""
    checker: GpuAvailabilityChecker = GpuAvailabilityChecker()
    assert checker.is_cuda_available() is True
    mock_which.assert_not_called()


@patch.object(GpuAvailabilityChecker, "_try_torch_cuda", return_value=False)
@patch("shutil.which")
def test_cuda_detection_caching(mock_which: MagicMock, mock_try_torch: MagicMock) -> None:
    """Verifies the detection result is cached and subsequent calls do not recheck."""
    mock_which.return_value = "/usr/bin/nvidia-smi"
    checker: GpuAvailabilityChecker = GpuAvailabilityChecker()

    # First call performs the detection
    assert checker.is_cuda_available() is True
    assert mock_try_torch.call_count == 1
    assert mock_which.call_count == 1

    # Second call should return the cached result
    assert checker.is_cuda_available() is True
    assert mock_try_torch.call_count == 1
    assert mock_which.call_count == 1


@patch.object(GpuAvailabilityChecker, "_probe_cuda")
def test_cuda_available_caching(mock_probe_cuda: MagicMock) -> None:
    """Verifies that is_cuda_available queries the driver via _probe_cuda."""
    checker: GpuAvailabilityChecker = GpuAvailabilityChecker()
    mock_probe_cuda.return_value = True

    # First call: queries driver
    assert checker.is_cuda_available() is True
    mock_probe_cuda.assert_called_once()

    # Second call: uses cached value
    assert checker.is_cuda_available() is True
    assert mock_probe_cuda.call_count == 1


@patch.object(GpuAvailabilityChecker, "_probe_cuda")
def test_cuda_unavailable_caching(mock_probe_cuda: MagicMock) -> None:
    """Verifies that is_cuda_available queries the driver via _probe_cuda."""
    checker: GpuAvailabilityChecker = GpuAvailabilityChecker()
    mock_probe_cuda.return_value = False

    # First call: queries driver
    assert checker.is_cuda_available() is False
    mock_probe_cuda.assert_called_once()

    # Second call: uses cached value
    assert checker.is_cuda_available() is False
    assert mock_probe_cuda.call_count == 1
