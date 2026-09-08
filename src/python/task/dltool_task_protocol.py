import asyncio
import json
import threading
from concurrent.futures import Future, TimeoutError
from enum import Enum
from typing import Any, Optional


class ProtocolField(Enum):
    PROJECT_ID = "project_id"
    TASK_ID = "task_id"
    RUN_ID = "run_id"
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


VALID_MESSAGE_TYPES = {t.value for t in MessageType}
VALID_TASK_STATUSES = {s.value for s in TaskStatus}


def validate_task_message(data: Any) -> tuple[bool, str]:
    """Strict validation of task protocol messages for both client and server."""
    if not isinstance(data, dict):
        return False, "消息必须为字典对象"

    # project_id
    if ProtocolField.PROJECT_ID.value not in data:
        return False, "缺少 project_id 字段"
    project_id = data[ProtocolField.PROJECT_ID.value]
    if not isinstance(project_id, str) or not project_id.strip():
        return False, "project_id 必须为非空字符串"

    # task_id
    if ProtocolField.TASK_ID.value not in data:
        return False, "缺少 task_id 字段"
    task_id = data[ProtocolField.TASK_ID.value]
    if type(task_id) is not int or task_id < 0:
        return False, "task_id 必须为非负整数"

    # run_id
    if ProtocolField.RUN_ID.value not in data:
        return False, "缺少 run_id 字段"
    run_id = data[ProtocolField.RUN_ID.value]
    if not isinstance(run_id, str) or not run_id.strip():
        return False, "run_id 必须为非空字符串"

    # type
    if ProtocolField.TYPE.value not in data:
        return False, "缺少 type 字段"
    msg_type = protocol_value(data[ProtocolField.TYPE.value])
    if not isinstance(msg_type, str) or msg_type not in VALID_MESSAGE_TYPES:
        return False, f"type 必须为有效消息类型: {msg_type}"

    # status
    if ProtocolField.STATUS.value in data:
        status_val = protocol_value(data[ProtocolField.STATUS.value])
        if not isinstance(status_val, str) or status_val not in VALID_TASK_STATUSES:
            return False, f"status 必须为有效任务状态: {status_val}"
    elif msg_type == MessageType.STATUS.value:
        return False, "status 类型的消息必须包含 status 字段"

    # progress
    if ProtocolField.PROGRESS.value in data:
        progress_val = data[ProtocolField.PROGRESS.value]
        if type(progress_val) is not int or ((progress_val < 0 and progress_val != -1) or progress_val > 100):
            return False, f"progress 必须为 0 到 100 的整数 (或 -1): {progress_val}"
    elif msg_type == MessageType.PROGRESS.value:
        return False, "progress 类型的消息必须包含 progress 字段"

    # eta_seconds
    if ProtocolField.ETA_SECONDS.value in data:
        eta_val = data[ProtocolField.ETA_SECONDS.value]
        if type(eta_val) is not int or eta_val < -1:
            return False, f"eta_seconds 必须为 >= -1 的整数: {eta_val}"

    # message
    if ProtocolField.MESSAGE.value in data:
        msg_val = data[ProtocolField.MESSAGE.value]
        if not isinstance(msg_val, str):
            return False, "message 必须为字符串"

    # command
    if ProtocolField.COMMAND.value in data:
        cmd_val = protocol_value(data[ProtocolField.COMMAND.value])
        if not isinstance(cmd_val, str) or TaskCommand.from_value(cmd_val) is None:
            return False, f"command 必须为有效命令: {cmd_val}"
    elif msg_type == MessageType.COMMAND.value:
        return False, "command 类型的消息必须包含 command 字段"

    return True, ""


class AsyncTaskClient:
    def __init__(self, host: str, port: int, task_id: int, run_id: str, project_id: str):
        self._task_id = int(task_id)
        self._run_id = str(run_id).strip()
        self._project_id = str(project_id).strip()
        if self._task_id < 0 or not self._run_id or not self._project_id:
            raise ValueError("task_id, run_id and project_id are required for task communication")
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
        if type(task_id) is not int or int(task_id) != self._task_id:
            raise ValueError("task message identity does not match the connected task")

        data: dict[str, Any] = dict(payload)
        data[ProtocolField.PROJECT_ID.value] = self._project_id
        data[ProtocolField.TASK_ID.value] = self._task_id
        data[ProtocolField.RUN_ID.value] = self._run_id
        data[ProtocolField.TYPE.value] = protocol_value(msg_type)
        if status is not None:
            data[ProtocolField.STATUS.value] = protocol_value(status)
        if progress is not None:
            data[ProtocolField.PROGRESS.value] = progress
        if eta_seconds is not None:
            data[ProtocolField.ETA_SECONDS.value] = eta_seconds
        if message:
            data[ProtocolField.MESSAGE.value] = message

        valid, err = validate_task_message(data)
        if not valid:
            raise ValueError(f"Invalid task message: {err}")

        if self._closed or self._writer is None:
            return

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
        expected = {int(task_id) for task_id in task_ids} if task_ids else {self._task_id}
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

            command_project_id = str(command.get(ProtocolField.PROJECT_ID.value, "")).strip()
            command_run_id = str(command.get(ProtocolField.RUN_ID.value, "")).strip()
            if command_project_id != self._project_id:
                continue
            if command_task_id not in expected or command_run_id != self._run_id:
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
                if protocol_value(data.get(ProtocolField.TYPE.value)) != MessageType.COMMAND.value:
                    continue
                try:
                    command_task_id = int(data.get(ProtocolField.TASK_ID.value, -1))
                except (TypeError, ValueError):
                    continue
                command_project_id = str(data.get(ProtocolField.PROJECT_ID.value, "")).strip()
                command_run_id = str(data.get(ProtocolField.RUN_ID.value, "")).strip()
                if (command_project_id != self._project_id or command_task_id != self._task_id
                        or command_run_id != self._run_id):
                    continue
                self._commands.put_nowait(data)
        except asyncio.CancelledError:
            raise
        except OSError:
            pass
        finally:
            self._closed = True


class TaskClient:
    def __init__(self, host: str, port: int, task_id: int, run_id: str, project_id: str):
        self._task_id = int(task_id)
        self._run_id = str(run_id).strip()
        self._project_id = str(project_id).strip()
        if self._task_id < 0 or not self._run_id or not self._project_id:
            raise ValueError("task_id, run_id and project_id are required for task communication")
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)
        self._client = AsyncTaskClient(host, int(port), self._task_id, self._run_id, self._project_id)
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
