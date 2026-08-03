from __future__ import annotations

import asyncio
import base64
import json
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib.resources import as_file, files
from threading import Lock
from typing import Any, Protocol

from wasmtime import Engine, Func, Instance, Memory, Module, Store

from .errors import UpstreamProtocolError


@dataclass(frozen=True, slots=True)
class Challenge:
    algorithm: str
    challenge: str
    salt: str
    difficulty: int
    expire_at: int
    signature: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Challenge:
        try:
            return cls(
                algorithm=str(payload["algorithm"]),
                challenge=str(payload["challenge"]),
                salt=str(payload["salt"]),
                difficulty=int(payload["difficulty"]),
                expire_at=int(payload["expire_at"]),
                signature=str(payload["signature"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise UpstreamProtocolError(
                "DeepSeek returned an invalid proof-of-work challenge"
            ) from exc


class PowSolverProtocol(Protocol):
    async def create_answer(self, challenge: Challenge, target_path: str) -> str: ...


class DeepSeekHashSolver:
    def __init__(self, *, timeout_seconds: float = 30.0, workers: int = 2) -> None:
        self._timeout_seconds = timeout_seconds
        self._engine = Engine()
        asset = files("deepseek_python_api.assets").joinpath("sha3_wasm_bg.7b9ca65ddd.wasm")
        with as_file(asset) as wasm_path:
            self._module = Module.from_file(self._engine, str(wasm_path))
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="deepseek-pow")

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    async def create_answer(self, challenge: Challenge, target_path: str) -> str:
        if challenge.algorithm != "DeepSeekHashV1":
            raise UpstreamProtocolError(
                f"Unsupported proof-of-work algorithm: {challenge.algorithm}"
            )

        loop = asyncio.get_running_loop()
        try:
            answer = await asyncio.wait_for(
                loop.run_in_executor(self._executor, self._calculate_hash, challenge),
                timeout=self._timeout_seconds,
            )
        except TimeoutError as exc:
            raise UpstreamProtocolError("DeepSeek proof-of-work calculation timed out") from exc

        payload = {
            "algorithm": challenge.algorithm,
            "challenge": challenge.challenge,
            "salt": challenge.salt,
            "answer": answer,
            "signature": challenge.signature,
            "target_path": target_path,
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        return base64.b64encode(encoded).decode()

    def _calculate_hash(self, challenge: Challenge) -> int | float:
        # wasmtime Stores are not thread-safe. A fresh Store/Instance per solve keeps the
        # compiled module cached while isolating mutable linear memory.
        store = Store(self._engine)
        instance = Instance(store, self._module, [])
        exports = instance.exports(store)
        memory = exports["memory"]
        allocate = exports["__wbindgen_export_0"]
        reallocate = exports["__wbindgen_export_1"]
        stack_pointer = exports["__wbindgen_add_to_stack_pointer"]
        solve = exports["wasm_solve"]
        if (
            not isinstance(memory, Memory)
            or not isinstance(allocate, Func)
            or not isinstance(reallocate, Func)
            or not isinstance(stack_pointer, Func)
            or not isinstance(solve, Func)
        ):
            raise UpstreamProtocolError("DeepSeek proof-of-work WASM exports are invalid")

        prefix = f"{challenge.salt}_{challenge.expire_at}_"
        retptr = int(stack_pointer(store, -16))
        try:
            challenge_ptr, challenge_len = self._write_string(
                store, memory, allocate, reallocate, challenge.challenge
            )
            prefix_ptr, prefix_len = self._write_string(
                store, memory, allocate, reallocate, prefix
            )
            solve(
                store,
                retptr,
                challenge_ptr,
                challenge_len,
                prefix_ptr,
                prefix_len,
                float(challenge.difficulty),
            )
            raw = bytes(memory.read(store, retptr, retptr + 16))
            status = struct.unpack_from("<i", raw, 0)[0]
            value = struct.unpack_from("<d", raw, 8)[0]
            if status == 0:
                raise UpstreamProtocolError("DeepSeek proof-of-work calculation failed")
            return int(value) if value.is_integer() else value
        finally:
            stack_pointer(store, 16)

    @staticmethod
    def _write_string(
        store: Store,
        memory: Any,
        allocate: Any,
        reallocate: Any,
        value: str,
    ) -> tuple[int, int]:
        encoded = value.encode("utf-8")
        initial_size = len(value)
        pointer = int(allocate(store, initial_size, 1))
        if len(encoded) != initial_size:
            pointer = int(reallocate(store, pointer, initial_size, len(encoded), 1))
        memory.write(store, encoded, pointer)
        return pointer, len(encoded)
