import inspect
import pkgutil

from .base import CLIWrapper

REGISTERED_PLUGINS = {}


def discover_plugins():
    """Discovers and registers CLIWrapper plugins."""
    for _finder, name, _ispkg in pkgutil.iter_modules(__path__):
        if name != "base":  # Avoid importing the base class itself
            module = __import__(f"{__name__}.{name}", fromlist=[name])
            for _member_name, member_obj in inspect.getmembers(module):
                if (
                    inspect.isclass(member_obj)
                    and issubclass(member_obj, CLIWrapper)
                    and member_obj is not CLIWrapper
                ):
                    plugin_instance = member_obj()
                    REGISTERED_PLUGINS[plugin_instance.get_cli_name()] = plugin_instance


discover_plugins()
