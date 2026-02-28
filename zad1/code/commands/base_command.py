from abc import ABC, abstractmethod

class Command(ABC):
    @abstractmethod
    def execute(self, sender=None, app_data=None, user_data=None):
        pass