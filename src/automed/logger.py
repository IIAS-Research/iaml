from .meta_singleton import MetaSingleton
import rich.console
import rich.progress


class Logger(metaclass=MetaSingleton):
    def __init__(self, quiet=False) -> None:
        self.console = rich.console.Console(log_path=False)
        self.progress = rich.progress.Progress(console=self.console)
        self.quiet = quiet
        
    def set_quiet(self, value:bool) -> None:
        self.quiet = value

    def log(self, *text: list[str]):
        if not self.quiet:
            self.console.log(*text)

