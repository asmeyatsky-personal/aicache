import re
import shutil

from .base import CLIWrapper


class OpenAICLIWrapper(CLIWrapper):
    def get_cli_name(self) -> str:
        return "openai"

    def parse_arguments(self, args: list) -> tuple[str, dict]:
        model = None
        prompt_content = None

        i = 0
        while i < len(args):
            if args[i] == "--model":
                if i + 1 < len(args):
                    model = args[i + 1]
                    i += 2
                else:
                    i += 1
            elif args[i] == "--messages":
                if i + 1 < len(args):
                    messages_str = args[i + 1]
                    match = re.search(r"content='([^']+)'", messages_str)
                    if match:
                        prompt_content = match.group(1)
                    i += 2
                else:
                    i += 1
            else:
                i += 1

        context = {"model": model}
        return prompt_content, context

    def execute_cli(self, args: list) -> tuple[str, int, str]:
        real_openai_path = shutil.which("openai")
        if not real_openai_path:
            return "", 1, "Error: openai executable not found."

        return self._run_cli_command(real_openai_path, args)
