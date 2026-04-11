from unittest.mock import MagicMock, patch

import pytest

from src.utils.data_manager import DataSyncManager


@patch("src.utils.data_manager.HfApi")
def test_data_manager_initialization(mock_hfapi):
    # Test initialization cleanly grabs token or accepts it
    manager = DataSyncManager(token="fake_token")
    assert manager.token == "fake_token"
    mock_hfapi.assert_called_once_with(token="fake_token")


@patch("src.utils.data_manager.hf_hub_download")
def test_download_dataset_file(mock_download):
    manager = DataSyncManager(token="fake_token")
    mock_download.return_value = "/fake/local/dir/dataset.csv"

    path = manager.download_dataset_file("repo/id", "dataset.csv", "/fake/local/dir")

    assert path == "/fake/local/dir/dataset.csv"
    mock_download.assert_called_once_with(
        repo_id="repo/id",
        filename="dataset.csv",
        repo_type="dataset",
        local_dir="/fake/local/dir",
        token="fake_token",
    )


@patch("os.path.exists")
@patch("src.utils.data_manager.HfApi")
def test_upload_artifact(mock_api_class, mock_exists):
    mock_exists.return_value = True

    # Mocking HfApi instance
    mock_api_instance = MagicMock()
    mock_api_class.return_value = mock_api_instance
    mock_api_instance.upload_file.return_value = "https://huggingface.co/fake/url"

    manager = DataSyncManager(token="fake_token")

    url = manager.upload_artifact("local_model.bin", "repo/id", "remote_model.bin")

    assert url == "https://huggingface.co/fake/url"
    mock_api_instance.upload_file.assert_called_once_with(
        path_or_fileobj="local_model.bin",
        path_in_repo="remote_model.bin",
        repo_id="repo/id",
        repo_type="model",
    )


@patch("os.path.exists")
def test_upload_missing_artifact(mock_exists):
    mock_exists.return_value = False
    manager = DataSyncManager(token="fake_token")

    with pytest.raises(FileNotFoundError):
        manager.upload_artifact("missing_model.bin", "repo/id", "remote_model.bin")
