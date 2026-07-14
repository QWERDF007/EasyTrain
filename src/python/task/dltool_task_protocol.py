import asyncio
import json
import threading
from concurrent.futures import Future, TimeoutError
from enum import Enum
from typing import Any, Optional


class ProtocolField(Enum):
    TASK_ID = "task_id"
    TYPE = "type"
    STATUS = "status"
    PROGRESS = "progress"
    ETA_SECONDS = "eta_seconds"
    MESSAGE = "message"
    COMMAND = "command"


class MessageType(Enum):
    EVENT = "event"
    STATUS = "status"
    PROGRESS = "progress"
    LOG = "log"
    COMMAND = "command"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    FINISHED = "finished"
    FAILED = "failed"
    ERROR = "error"


class TaskCommand(Enum):
    STOP = "stop"

    @classmethod
    def from_value(cls, value: Any) -> Optional["TaskCommand"]:
        normalized = str(value).strip().lower()
        for item in cls:
            if item.value == normalized:
                return item
        return None


def protocol_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


class AsyncTaskClient:
    def __init__(self, host: str, port: int):
        self._host = host
        self._port = int(port)
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._commands: "asyncio.Queue[dict[str, Any]]" = asyncio.Queue()
        self._write_lock = asyncio.Lock()
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._closed = False

    async def connect(self) -> None:
        self._reader, self._writer = await asyncio.open_connection(self._host, self._port)
        self._reader_task = asyncio.create_task(self._read_loop())

    async def close(self) -> None:
        self._closed = True
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None
        if self._writer is not None:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except OSError:
                pass
            self._writer = None

    async def send(self, task_id: int, msg_type: MessageType, status: Optional[TaskStatus],
                   progress: int, eta_seconds: int, message: str = "", **payload: Any) -> None:
        if self._closed or self._writer is None:
            return

        data: dict[str, Any] = {
            ProtocolField.TASK_ID.value: int(task_id),
            ProtocolField.TYPE.value: protocol_value(msg_type),
        }
        if status is not None:
            data[ProtocolField.STATUS.value] = protocol_value(status)
        if progress >= 0:
            data[ProtocolField.PROGRESS.value] = max(0, min(100, int(progress)))
        data[ProtocolField.ETA_SECONDS.value] = max(-1, int(eta_seconds))
        if message:
            data[ProtocolField.MESSAGE.value] = message
        data.update(payload)

        raw = json.dumps(data, ensure_ascii=False).encode("utf-8") + b"\n"
        async with self._write_lock:
            if not self._closed and self._writer is not None:
                self._writer.write(raw)
                await self._writer.drain()

    async def status(self, task_id: int, status: TaskStatus, progress: int, eta_seconds: int,
                     message: str = "", **payload: Any) -> None:
        await self.send(task_id, MessageType.STATUS, status, progress, eta_seconds, message, **payload)

    async def progress(self, task_id: int, progress: int, eta_seconds: int, message: str = "",
                       **payload: Any) -> None:
        await self.send(task_id, MessageType.PROGRESS, None, progress, eta_seconds, message, **payload)

    async def log(self, task_id: int, message: str) -> None:
        await self.send(task_id, MessageType.LOG, None, -1, -1, message)

    async def should_stop(self, *task_ids: int) -> bool:
        expected = {int(task_id) for task_id in task_ids}
        kept: list[dict[str, Any]] = []
        should_stop = False

        while True:
            try:
                command = self._commands.get_nowait()
            except asyncio.QueueEmpty:
                break

            try:
                command_task_id = int(command.get(ProtocolField.TASK_ID.value, -1))
            except (TypeError, ValueError):
                command_task_id = -1

            if expected and command_task_id not in expected:
                kept.append(command)
                continue

            if TaskCommand.from_value(command.get(ProtocolField.COMMAND.value)) == TaskCommand.STOP:
                should_stop = True
                continue
            kept.append(command)

        for command in kept:
            self._commands.put_nowait(command)
        return should_stop

    async def _read_loop(self) -> None:
        if self._reader is None:
            return

        try:
            while not self._closed:
                line = await self._reader.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                if protocol_value(data.get(ProtocolField.TYPE.value)) == MessageType.COMMAND.value:
                    self._commands.put_nowait(data)
        except asyncio.CancelledError:
            raise
        except OSError:
            pass
        finally:
            self._closed = True


class TaskClient:
    def __init__(self, host: str, port: int):
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)
        self._client = AsyncTaskClient(host, int(port))
        self._submit(self._client.connect()).result(timeout=10)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._submit(self._client.close()).result(timeout=5)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)

    def send(self, task_id: int, msg_type: MessageType, status: Optional[TaskStatus],
             progress: int, eta_seconds: int, message: str = "", **payload: Any) -> None:
        if self._closed:
            return
        self._submit(self._client.send(task_id, msg_type, status, progress, eta_seconds, message,
                                       **payload)).result(timeout=5)

    def status(self, task_id: int, status: TaskStatus, progress: int, eta_seconds: int, message: str = "",
               **payload: Any) -> None:
        if self._closed:
            return
        self._submit(self._client.status(task_id, status, progress, eta_seconds, message, **payload)).result(timeout=5)

    def progress(self, task_id: int, progress: int, eta_seconds: int, message: str = "", **payload: Any) -> None:
        if self._closed:
            return
        self._submit(self._client.progress(task_id, progress, eta_seconds, message, **payload)).result(timeout=5)

    def log(self, task_id: int, message: str) -> None:
        if self._closed:
            return
        self._submit(self._client.log(task_id, message)).result(timeout=5)

    def should_stop(self, *task_ids: int) -> bool:
        if self._closed:
            return False
        try:
            return self._submit(self._client.should_stop(*task_ids)).result(timeout=1)
        except TimeoutError:
            return False

    def _submit(self, coro: Any) -> Future[Any]:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self._loop.close()
