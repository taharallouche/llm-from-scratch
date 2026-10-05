from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@patch("llm.utils.data_path")
def test_load_data(mock_data_path: MagicMock, tmp_path: Path) -> None:
    # Given
    from llm.utils import load_data

    mock_data_path.return_value = tmp_path
    dataset = "test_data"
    content = "This is a test"
    Path(tmp_path / "input").mkdir()
    file_path = Path(tmp_path / "input" / "test_data.txt")
    file_path.touch()
    file_path.write_text(content)

    expected_result = content

    # When
    data = load_data(dataset)

    # Then
    assert expected_result == data
