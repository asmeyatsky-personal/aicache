import subprocess
from abc import ABC, abstractmethod


class CLIWrapper(ABC):
    @abstractmethod
    def get_cli_name(self) -> str:
        """Returns the name of the CLI this wrapper supports (e.g., 'gcloud', 'llm', 'openai')."""
        pass

    @abstractmethod
    def parse_arguments(self, args: list) -> tuple[str, dict]:
        """
        Parses the command-line arguments and extracts the prompt and context for caching.
        Returns a tuple: (prompt_content: str, context: dict).
        """
        pass

    @abstractmethod
    def execute_cli(self, args: list) -> tuple[str, int, str]:
        """
        Executes the real CLI command with the given arguments.
        Returns a tuple: (stdout: str, return_code: int, stderr: str).
        """
        pass

    def _run_cli_command(
        self, real_cli_path: str, args: list, input_data: str | None = None
    ) -> tuple[str, int, str]:
        """Helper method to run a CLI command."""
        try:
            if input_data:
                process = subprocess.run(
                    [real_cli_path, *args],
                    input=input_data.encode("utf-8"),
                    capture_output=True,
                    check=False,
                )
            else:
                process = subprocess.run([real_cli_path, *args], capture_output=True, check=False)
            return (
                process.stdout.decode("utf-8"),
                process.returncode,
                process.stderr.decode("utf-8"),
            )
        except FileNotFoundError:
            return "", 1, f"Error: {real_cli_path} executable not found."

    async def _run_cli_command_async(
        self, real_cli_path: str, args: list, input_data: str | None = None
    ) -> tuple[str, int, str]:
        """Async helper method to run a CLI command."""
        import asyncio

        try:
            if input_data:
                process = await asyncio.create_subprocess_exec(
                    real_cli_path,
                    *args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await process.communicate(input=input_data.encode("utf-8"))
            else:
                process = await asyncio.create_subprocess_exec(
                    real_cli_path,
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await process.communicate()
            return stdout.decode("utf-8"), process.returncode, stderr.decode("utf-8")
        except FileNotFoundError:
            return "", 1, f"Error: {real_cli_path} executable not found."
