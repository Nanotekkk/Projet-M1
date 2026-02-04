"""Observer Pattern for event-driven architecture"""

from abc import ABC, abstractmethod
from typing import List, Any


class Observer(ABC):
    """Abstract observer interface"""

    @abstractmethod
    def update(self, event_type: str, data: Any) -> None:
        """Handle event notification"""
        pass


class Observable:
    """Observable base class implementing the Observer pattern"""

    def __init__(self):
        self._observers: List[Observer] = []

    def attach(self, observer: Observer) -> None:
        """Attach an observer"""
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        """Detach an observer"""
        if observer in self._observers:
            self._observers.remove(observer)

    def notify(self, event_type: str, data: Any = None) -> None:
        """Notify all observers of an event"""
        for observer in self._observers:
            observer.update(event_type, data)
