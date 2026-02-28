class CommandDispatcher:
    def __init__(self):
        self._commands = {}

    def register(self, name, command):
        self._commands[name] = command

    def execute(self, name, sender=None, app_data=None, user_data=None):
        if name in self._commands:
            self._commands[name].execute(sender, app_data, user_data)