"""Local reminder scheduling and persistence."""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable


@dataclass
class Reminder:
    id: str
    text: str
    trigger_at: str
    created_at: str
    fired: bool = False


def _now_local() -> datetime:
    return datetime.now().astimezone()


def _parse_dt(iso_text: str) -> datetime:
    return datetime.fromisoformat(iso_text)


class ReminderStore:
    """Thread-safe JSON reminder storage."""

    def __init__(self, file_path: str) -> None:
        self.file_path = Path(file_path)
        self._lock = threading.Lock()
        if not self.file_path.exists():
            self._write([])

    def _read(self) -> list[Reminder]:
        with self.file_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return [Reminder(**item) for item in raw]

    def _write(self, reminders: list[Reminder]) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("w", encoding="utf-8") as handle:
            json.dump([asdict(item) for item in reminders], handle, indent=2)

    def add(self, text: str, when_dt: datetime) -> Reminder:
        with self._lock:
            reminders = self._read()
            reminder = Reminder(
                id=str(uuid.uuid4()),
                text=text,
                trigger_at=when_dt.isoformat(),
                created_at=_now_local().isoformat(),
                fired=False,
            )
            reminders.append(reminder)
            self._write(reminders)
            return reminder

    def due(self, now_dt: datetime) -> list[Reminder]:
        with self._lock:
            reminders = self._read()
            due_items = [
                item for item in reminders if (not item.fired and _parse_dt(item.trigger_at) <= now_dt)
            ]
            if due_items:
                due_ids = {item.id for item in due_items}
                for item in reminders:
                    if item.id in due_ids:
                        item.fired = True
                self._write(reminders)
            return due_items


def _parse_at_time(raw: str, now_dt: datetime) -> datetime | None:
    text = raw.strip().lower()
    formats = [
        "%I %p",
        "%I:%M %p",
        "%H:%M",
        "%I%p",
    ]

    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt)
            candidate = now_dt.replace(
                hour=parsed.hour,
                minute=parsed.minute,
                second=0,
                microsecond=0,
            )
            if candidate <= now_dt:
                candidate += timedelta(days=1)
            return candidate
        except ValueError:
            continue

    try:
        # Optional full datetime format: YYYY-MM-DD HH:MM
        parsed_full = datetime.strptime(text, "%Y-%m-%d %H:%M")
        return parsed_full.replace(tzinfo=now_dt.tzinfo)
    except ValueError:
        return None


def parse_reminder_request(text: str, now_dt: datetime | None = None) -> tuple[str, datetime] | None:
    """Parse reminder requests from natural-ish commands."""
    now_dt = now_dt or _now_local()
    normalized = " ".join(text.strip().split())

    in_match = re.search(
        r"(?:remind me to|set a reminder to)\s+(.+?)\s+in\s+(\d+)\s*(minutes?|hours?)\b",
        normalized,
        re.IGNORECASE,
    )
    if in_match:
        task = in_match.group(1).strip()
        value = int(in_match.group(2))
        unit = in_match.group(3).lower()
        delta = timedelta(minutes=value) if "minute" in unit else timedelta(hours=value)
        return task, now_dt + delta

    at_match = re.search(
        r"(?:remind me to|set a reminder to)\s+(.+?)\s+at\s+(.+)$",
        normalized,
        re.IGNORECASE,
    )
    if at_match:
        task = at_match.group(1).strip()
        when_raw = at_match.group(2).strip()
        when_dt = _parse_at_time(when_raw, now_dt)
        if when_dt:
            return task, when_dt

    alarm_match = re.search(
        r"(?:set alarm(?: for)?|alarm(?: for)?)\s+(.+)$",
        normalized,
        re.IGNORECASE,
    )
    if alarm_match:
        when_raw = alarm_match.group(1).strip()
        when_dt = _parse_at_time(when_raw, now_dt)
        if when_dt:
            return "alarm", when_dt

    return None


def try_schedule_reminder(text: str, store: ReminderStore) -> str | None:
    parsed = parse_reminder_request(text)
    if not parsed:
        return None

    task, when_dt = parsed
    reminder = store.add(task, when_dt)
    local_time = _parse_dt(reminder.trigger_at).strftime("%Y-%m-%d %I:%M %p")
    if task == "alarm":
        return f"Okay, alarm set for {local_time}."
    return f"Okay, reminder set for {local_time}: {task}."


class ReminderScheduler(threading.Thread):
    """Background loop that triggers due reminders."""

    def __init__(
        self,
        store: ReminderStore,
        on_due: Callable[[Reminder], None],
        poll_seconds: float = 1.0,
    ) -> None:
        super().__init__(daemon=True)
        self.store = store
        self.on_due = on_due
        self.poll_seconds = poll_seconds
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            now_dt = _now_local()
            for reminder in self.store.due(now_dt):
                self.on_due(reminder)
            self._stop_event.wait(self.poll_seconds)
