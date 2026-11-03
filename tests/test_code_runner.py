"""Sandbox tests: correctness, limits, isolation and error handling."""
import os
import tempfile

import pytest

from backend.services.code_runner import CaseResult, LocalSandbox, UnsupportedLanguage, available_languages, outputs_match

SB = LocalSandbox(timeout=4)
TESTS = [{"input": "3\n1 2 3", "output": "6", "hidden": False}, {"input": "1\n5", "output": "5", "hidden": True}]
PY_SUM = "import sys\nd = sys.stdin.read().split()\nprint(sum(map(int, d[1:])))"


def statuses(results):
    return [r.status for r in results]


def test_python_pass():
    r = SB.run_tests("python", PY_SUM, TESTS)
    assert statuses(r) == ["passed", "passed"] and all(isinstance(x, CaseResult) for x in r)
    assert all(x.runtime_ms > 0 for x in r)


def test_wrong_answer_and_hidden_masking():
    r = SB.run_tests("python", "print(0)", TESTS)
    assert statuses(r) == ["wrong_answer", "wrong_answer"]
    hidden = r[1].public()
    assert hidden["input"] == hidden["expected"] == hidden["actual"] == "(hidden)"
    visible = r[0].public()
    assert visible["expected"] == "6" and visible["actual"].strip() == "0"


def test_runtime_error_is_reported():
    r = SB.run_tests("python", "raise ValueError('boom')", TESTS[:1])[0]
    assert r.status == "runtime_error" and "boom" in r.error


def test_infinite_loop_times_out():
    sb = LocalSandbox(timeout=1)
    r = sb.run_tests("python", "while True: pass", TESTS[:1])[0]
    assert r.status == "timeout" and not r.passed


def test_memory_limit():
    sb = LocalSandbox(timeout=4, memory_mb=128)
    r = sb.run_tests("python", "x = bytearray(10**9)\nprint(len(x))", TESTS[:1])[0]
    assert r.status in ("memory_limit", "runtime_error")


def test_network_is_blocked():
    code = ("import socket\ntry:\n    socket.create_connection(('example.com', 80), timeout=2)\n    print('CONNECTED')\n"
            "except Exception as e:\n    print('BLOCKED')")
    out = SB.run_once("python", code, "")
    assert "CONNECTED" not in out.stdout


def test_no_secrets_in_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret")
    out = SB.run_once("python", "import os\nprint(os.environ.get('GEMINI_API_KEY'))", "")
    assert out.stdout.strip() == "None"


def test_temp_directory_cleaned_up():
    before = {d for d in os.listdir(tempfile.gettempdir()) if d.startswith("pw_")}
    SB.run_tests("python", PY_SUM, TESTS)
    after = {d for d in os.listdir(tempfile.gettempdir()) if d.startswith("pw_")}
    assert after <= before


def test_unsupported_language():
    with pytest.raises(UnsupportedLanguage):
        SB.run_tests("ruby", "puts 1", TESTS)


@pytest.mark.skipif(not available_languages()["cpp"], reason="g++ not installed")
def test_cpp_pass_and_compile_error():
    ok = '#include <bits/stdc++.h>\nint main(){int n;long long t=0,x;std::cin>>n;while(n--){std::cin>>x;t+=x;}std::cout<<t;}'
    assert statuses(SB.run_tests("cpp", ok, TESTS)) == ["passed", "passed"]
    bad = SB.run_tests("cpp", "int main(){ return undefined_symbol; }", TESTS)
    assert statuses(bad) == ["compile_error", "compile_error"] and bad[0].error


@pytest.mark.skipif(not available_languages()["java"], reason="JDK not installed")
def test_java_pass():
    code = ("import java.util.*;\npublic class Main{public static void main(String[] a){Scanner s=new Scanner(System.in);"
            "int n=s.nextInt();long t=0;while(n-->0)t+=s.nextInt();System.out.println(t);}}")
    assert statuses(SB.run_tests("java", code, TESTS)) == ["passed", "passed"]


def test_sql_runner_orders_and_nulls():
    setup = "CREATE TABLE t(a INT, b TEXT); INSERT INTO t VALUES (2,'y'),(1,NULL);"
    out = SB.run_once("sql", "SELECT a, b FROM t ORDER BY a;", setup)
    assert out.stdout.splitlines() == ["1|NULL", "2|y"]
    res = SB.run_tests("sql", "SELECT a, b FROM t ORDER BY a;", [{"input": setup, "output": "1|NULL\n2|y", "hidden": False}], "sql")
    assert res[0].passed


def test_output_matching_is_token_based():
    assert outputs_match("1  2\n3\n", "1 2 3")
    assert not outputs_match("1 2", "1 2 3")
    assert outputs_match(" a|b \n", "a|b", kind="sql")
