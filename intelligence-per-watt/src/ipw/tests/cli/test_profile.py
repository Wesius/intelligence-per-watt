"""Tests for profile CLI command."""

from __future__ import annotations

from unittest.mock import Mock, patch

from click.testing import CliRunner

from ipw.cli.profile import profile


class TestProfileCommand:
    """Test the profile CLI command."""

    @patch("ipw.execution.ProfilerRunner")
    @patch("ipw.datasets.ensure_registered")
    @patch("ipw.clients.ensure_registered")
    def test_passes_batch_size_to_runner(
        self,
        mock_clients_ensure: Mock,
        mock_datasets_ensure: Mock,
        mock_runner: Mock,
    ) -> None:
        mock_clients_ensure.return_value = None
        mock_datasets_ensure.return_value = None
        mock_runner.return_value.run.return_value = None

        runner = CliRunner()
        with patch.dict("ipw.clients.MISSING_CLIENTS", {}, clear=True):
            result = runner.invoke(
                profile,
                [
                    "--client",
                    "demo",
                    "--model",
                    "llama",
                    "--batch-size",
                    "4",
                ],
            )

        assert result.exit_code == 0
        config = mock_runner.call_args[0][0]
        assert config.batch_size == 4

    def test_rejects_invalid_batch_size(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            profile,
            [
                "--client",
                "demo",
                "--model",
                "llama",
                "--batch-size",
                "0",
            ],
        )

        assert result.exit_code != 0
        assert "Invalid value for '--batch-size'" in result.output
