"""Sandboxed execution of user code against test cases.

Correctness in PrepWise is decided ONLY here, by running code, never by an LLM.

Local sandbox (default, ``CODE_RUNNER=local``):

* one fresh temporary directory per submission, deleted afterwards
* ``subprocess`` with a wall-clock timeout; the whole process group is killed on timeout
* ``resource`` limits in the child: CPU seconds, address space (memory), max file size
* a minimal environment (no proxy variables, no secrets) and, when the kernel allows
  unprivileged user namespaces, ``unshare --net`` for a network-less process;
  Python additionally gets a bootstrap that disables the ``socket`` module
* languages: Python 3, Java (17+), C++17 and SQL (SQLite in-memory)

For stronger isolation in production, use Docker with ``network_mode: none`` or set
``CODE_RUNNER=judge0`` to delegate execution to a Judge0 instance.
"""
from __future__ import annotations

import base64
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from functools import lru_cache

import httpx

from backend.core.config import get_settings

SUPPORTED_LANGUAGES = ("python", "java", "cpp", "sql")
MAX_OUTPUT_BYTES = 64 * 1024

PY_BOOT = r'''
import sys, runpy
sys.setrecursionlimit(10000)
try:
    import socket as _s
    _MSG = "Network access is disabled in the PrepWise sandbox"
    class _BlockedSocket(_s.socket):
        def __init__(self, *a, **k):
            raise OSError(_MSG)
    def _blocked(*a, **k):
        raise OSError(_MSG)
    _s.socket = _BlockedSocket; _s.create_connection = _blocked; _s.getaddrinfo = _blocked
except Exception:
    pass
sys.argv = ["main.py"]
runpy.run_path("main.py", run_name="__main__")
'''

SQL_RUNNER = r'''
import sqlite3, sys
setup = sys.stdin.read()
query = open("query.sql", encoding="utf-8").read().strip()
con = sqlite3.connect(":memory:")
con.executescript(setup)
cur = con.execute(query)
for row in cur.fetchall():
    print("|".join("NULL" if v is None else str(v) for v in row))
'''


class UnsupportedLanguage(ValueError):
    pass


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    runtime_ms: float
    memory_exceeded: bool = False


@dataclass
class CaseResult:
    index: int
    hidden: bool
    status: str  # passed | wrong_answer | runtime_error | timeout | memory_limit | compile_error
    passed: bool
    runtime_ms: float
    input: str = ""
    expected: str = ""
    actual: str = ""
    error: str = ""

    def public(self) -> dict:
        """Hidden cases only reveal status and runtime, never their data."""
        d = asdict(self)
        if self.hidden:
            d.update(input="(hidden)", expected="(hidden)", actual="(hidden)",
                     error=self.error[:200] if self.status != "wrong_answer" else "")
        return d


class CompileError(RuntimeError):
    pass


# ----------------------------------------------------------------- helpers --
def outputs_match(actual: str, expected: str, kind: str = "code") -> bool:
    """Token-wise comparison for programs; line-wise (trimmed) comparison for SQL."""
    if kind == "sql":
        a = [ln.strip() for ln in actual.strip().splitlines() if ln.strip()]
        e = [ln.strip() for ln in expected.strip().splitlines() if ln.strip()]
        return a == e
    return actual.split() == expected.split()


@lru_cache(maxsize=1)
def available_languages() -> dict[str, bool]:
    return {
        "python": True,
        "sql": True,
        "java": bool(shutil.which("javac") and shutil.which("java")),
        "cpp": bool(shutil.which("g++")),
    }


@lru_cache(maxsize=1)
def _netns_prefix() -> list[str]:
    """Use ``unshare -rn`` (new user + network namespace) when the kernel permits it."""
    exe = shutil.which("unshare")
    if not exe:
        return []
    try:
        r = subprocess.run([exe, "-rn", "true"], capture_output=True, timeout=5)
        return [exe, "-rn"] if r.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        return []


def network_isolation_mode() -> str:
    return "network namespace (unshare -rn)" if _netns_prefix() else "best-effort (clean env + socket guard)"


def _clean_env(workdir: str) -> dict[str, str]:
    return {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": workdir, "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}


def _limits(cpu_sec: int, mem_bytes: int | None):
    def apply():  # runs in the child just before exec
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if mem_bytes:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    return apply


def _run(cmd: list[str], stdin: str, workdir: str, timeout: float, mem_mb: int | None) -> ExecResult:
    start = time.perf_counter()
    proc = subprocess.Popen(
        _netns_prefix() + cmd,
        cwd=workdir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_clean_env(workdir),
        preexec_fn=_limits(int(timeout) + 1, mem_mb * 1024 * 1024 if mem_mb else None),
    )
    timed_out = False
    try:
        out, err = proc.communicate(stdin.encode(), timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        out, err = proc.communicate()
    runtime = (time.perf_counter() - start) * 1000
    stdout = out[:MAX_OUTPUT_BYTES].decode("utf-8", "replace")
    stderr = err[:MAX_OUTPUT_BYTES].decode("utf-8", "replace")
    mem = bool(re.search(r"MemoryError|bad_alloc|OutOfMemoryError|Cannot allocate memory", stderr))
    if proc.returncode in (-signal.SIGXCPU, -signal.SIGKILL) and not timed_out:
        timed_out = proc.returncode == -signal.SIGXCPU
    return ExecResult(stdout, stderr, proc.returncode if proc.returncode is not None else -1,
                      timed_out, runtime, mem)


def _java_class_name(code: str) -> str:
    m = re.search(r"public\s+(?:final\s+)?class\s+(\w+)", code)
    if m:
        return m.group(1)
    m = re.search(r"class\s+(\w+)[^{]*\{[^}]*?static\s+void\s+main", code, re.DOTALL)
    return m.group(1) if m else "Main"


# ------------------------------------------------------------ local sandbox --
class LocalSandbox:
    """Runs code in a throwaway directory with resource limits."""

    def __init__(self, timeout: float | None = None, memory_mb: int | None = None,
                 compile_timeout: float | None = None):
        s = get_settings()
        self.timeout = timeout or s.sandbox_timeout_sec
        self.memory_mb = memory_mb or s.sandbox_memory_mb
        self.compile_timeout = compile_timeout or s.sandbox_compile_timeout_sec

    def _prepare(self, language: str, code: str, wd: str) -> tuple[list[str], int | None]:
        """Write/compile the program; return (run command, memory limit for RLIMIT_AS)."""
        if language not in SUPPORTED_LANGUAGES:
            raise UnsupportedLanguage(f"Unsupported language '{language}'. Use one of {SUPPORTED_LANGUAGES}.")
        if not available_languages().get(language):
            raise UnsupportedLanguage(f"The {language} toolchain is not installed on this server.")
        if language == "python":
            with open(os.path.join(wd, "main.py"), "w", encoding="utf-8") as f:
                f.write(code)
            with open(os.path.join(wd, "_boot.py"), "w", encoding="utf-8") as f:
                f.write(PY_BOOT)
            return [sys.executable, "-I", "-S", "_boot.py"], self.memory_mb
        if language == "sql":
            with open(os.path.join(wd, "query.sql"), "w", encoding="utf-8") as f:
                f.write(code)
            with open(os.path.join(wd, "_sql.py"), "w", encoding="utf-8") as f:
                f.write(SQL_RUNNER)
            return [sys.executable, "-I", "-S", "_sql.py"], self.memory_mb
        if language == "cpp":
            src = os.path.join(wd, "main.cpp")
            with open(src, "w", encoding="utf-8") as f:
                f.write(code)
            r = _run(["g++", "-O2", "-std=c++17", "-pipe", "-o", "main", "main.cpp"], "", wd,
                     self.compile_timeout, None)
            if r.exit_code != 0 or r.timed_out:
                raise CompileError(r.stderr[-2000:] or "Compilation timed out")
            return ["./main"], self.memory_mb
        # java
        cls = _java_class_name(code)
        with open(os.path.join(wd, f"{cls}.java"), "w", encoding="utf-8") as f:
            f.write(code)
        r = _run(["javac", "-J-Xmx512m", "-encoding", "UTF-8", f"{cls}.java"], "", wd, self.compile_timeout, None)
        if r.exit_code != 0 or r.timed_out:
            raise CompileError(r.stderr[-2000:] or "Compilation timed out")
        # JVMs reserve large virtual memory, so the heap is capped with -Xmx instead of RLIMIT_AS.
        return ["java", f"-Xmx{self.memory_mb}m", "-Xss64m", "-XX:+UseSerialGC",
                "-XX:TieredStopAtLevel=1", "-cp", ".", cls], None

    def run_once(self, language: str, code: str, stdin: str) -> ExecResult:
        """Run a program once on custom input (the "Run" button)."""
        with tempfile.TemporaryDirectory(prefix="pw_") as wd:
            try:
                cmd, mem = self._prepare(language, code, wd)
            except CompileError as exc:
                return ExecResult("", str(exc), 1, False, 0.0)
            timeout = self.timeout * (2.5 if language == "java" else 1)
            return _run(cmd, stdin, wd, timeout, mem)

    def run_tests(self, language: str, code: str, tests: list[dict], kind: str = "code") -> list[CaseResult]:
        """Run every test case; compile errors fail all cases with the same message."""
        results: list[CaseResult] = []
        with tempfile.TemporaryDirectory(prefix="pw_") as wd:
            try:
                cmd, mem = self._prepare(language, code, wd)
            except CompileError as exc:
                return [CaseResult(i, t.get("hidden", True), "compile_error", False, 0.0,
                                   t["input"], t["output"], "", str(exc)[:2000]) for i, t in enumerate(tests)]
            timeout = self.timeout * (2.5 if language == "java" else 1)
            for i, t in enumerate(tests):
                r = _run(cmd, t["input"], wd, timeout, mem)
                if r.timed_out:
                    status = "timeout"
                elif r.memory_exceeded:
                    status = "memory_limit"
                elif r.exit_code != 0:
                    status = "runtime_error"
                elif outputs_match(r.stdout, t["output"], kind):
                    status = "passed"
                else:
                    status = "wrong_answer"
                results.append(CaseResult(i, t.get("hidden", True), status, status == "passed",
                                          round(r.runtime_ms, 1), t["input"], t["output"], r.stdout[:4000],
                                          r.stderr[-1500:] if status != "passed" else ""))
        return results


# ----------------------------------------------------------- Judge0 sandbox --
JUDGE0_LANG_IDS = {"python": 71, "java": 62, "cpp": 54}


class Judge0Sandbox:
    """Delegates execution to a Judge0 instance (``CODE_RUNNER=judge0``).

    SQL questions still run locally because Judge0's SQLite language does not use
    the stdin-setup contract PrepWise uses.
    """

    def __init__(self):
        self.s = get_settings()
        self.local = LocalSandbox()

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.s.judge0_api_key:
            h.update({"X-RapidAPI-Key": self.s.judge0_api_key, "X-RapidAPI-Host": self.s.judge0_api_host})
        return h

    def _submit(self, language: str, code: str, stdin: str) -> dict:
        b64 = lambda s: base64.b64encode(s.encode()).decode()  # noqa: E731
        payload = {"source_code": b64(code), "language_id": JUDGE0_LANG_IDS[language], "stdin": b64(stdin),
                   "cpu_time_limit": self.s.sandbox_timeout_sec, "memory_limit": self.s.sandbox_memory_mb * 1024}
        url = self.s.judge0_url.rstrip("/") + "/submissions?base64_encoded=true&wait=true"
        r = httpx.post(url, json=payload, headers=self._headers(), timeout=60)
        r.raise_for_status()
        data = r.json()
        dec = lambda v: base64.b64decode(v).decode("utf-8", "replace") if v else ""  # noqa: E731
        return {"stdout": dec(data.get("stdout")), "stderr": dec(data.get("stderr")) + dec(data.get("compile_output")),
                "status_id": data.get("status", {}).get("id", 0), "time": float(data.get("time") or 0) * 1000}

    def run_once(self, language: str, code: str, stdin: str) -> ExecResult:
        if language == "sql":
            return self.local.run_once(language, code, stdin)
        d = self._submit(language, code, stdin)
        return ExecResult(d["stdout"], d["stderr"], 0 if d["status_id"] in (3, 4) else 1, d["status_id"] == 5, d["time"])

    def run_tests(self, language: str, code: str, tests: list[dict], kind: str = "code") -> list[CaseResult]:
        if language == "sql":
            return self.local.run_tests(language, code, tests, kind)
        if language not in JUDGE0_LANG_IDS:
            raise UnsupportedLanguage(language)
        out = []
        for i, t in enumerate(tests):
            d = self._submit(language, code, t["input"])
            sid = d["status_id"]
            if sid == 6:
                status = "compile_error"
            elif sid == 5:
                status = "timeout"
            elif sid >= 7:
                status = "runtime_error"
            else:
                status = "passed" if outputs_match(d["stdout"], t["output"], kind) else "wrong_answer"
            out.append(CaseResult(i, t.get("hidden", True), status, status == "passed", d["time"], t["input"],
                                  t["output"], d["stdout"][:4000], d["stderr"][-1500:]))
        return out


def get_sandbox():
    """Return the configured sandbox implementation."""
    return Judge0Sandbox() if get_settings().code_runner == "judge0" else LocalSandbox()
