from ctxkernel.codecs import CodecRegistry
from ctxkernel.identity import IdentityRegistry, ToolSemantics
from ctxkernel.ir import ResourceId

reg = CodecRegistry()
ids = IdentityRegistry()

PY = '''
import redis
from jwt import decode

class TokenStore:
    def __init__(self): ...

def refresh_token(user, ttl):
    return None

async def validate_jwt(token):
    return True
'''


def test_source_outline_is_parsed_not_guessed():
    extract, media = reg.encode(PY, ResourceId("file", "src/auth.py"), "read_file")
    assert media == "text/x-source"
    assert "src/auth.py" in extract and "python" in extract
    assert "TokenStore" in extract
    assert "refresh_token" in extract
    assert "validate_jwt" in extract
    assert "redis" in extract  # imports


def test_test_output_keeps_failures_drops_passes():
    out = "\n".join(
        [f"tests/test_x.py::test_{i} PASSED" for i in range(200)]
        + [
            "FAILED tests/test_auth.py::test_refresh_race",
            "FAILED tests/test_auth.py::test_expiry",
            "=== 2 failed, 200 passed in 4.10s ===",
        ]
    )
    extract, media = reg.encode(out, None, "bash")
    assert media == "text/x-test-output"
    assert "test_refresh_race" in extract and "test_expiry" in extract
    assert "2 failed" in extract
    assert "PASSED" not in extract
    assert len(extract) < len(out) / 10


def test_diff_counts_files_and_lines():
    diff = (
        "diff --git a/src/auth.py b/src/auth.py\n"
        "--- a/src/auth.py\n+++ b/src/auth.py\n@@ -1,2 +1,3 @@\n"
        "+added one\n+added two\n-removed one\n"
    )
    extract, media = reg.encode(diff, None, "bash")
    assert media == "text/x-diff"
    assert "src/auth.py" in extract and "+2" in extract and "−1" in extract


def test_json_sketches_schema_not_contents():
    payload = '[{"id": 1, "name": "ada", "active": true}, {"id": 2, "name": "bob", "active": false}]'
    extract, media = reg.encode(payload, None, "http_get")
    assert media == "application/json"
    assert "array[2]" in extract and "id:int" in extract and "active:bool" in extract


def test_traceback_keeps_exception_and_innermost_frames():
    tb = (
        "Traceback (most recent call last):\n"
        '  File "/app/main.py", line 10, in <module>\n    run()\n'
        '  File "/app/auth.py", line 42, in refresh\n    raise ValueError("bad token")\n'
        "ValueError: bad token\n"
    )
    extract, media = reg.encode(tb, None, "bash")
    assert media == "text/x-traceback"
    assert "ValueError: bad token" in extract and "auth.py:42" in extract


def test_default_codec_always_matches():
    extract, media = reg.encode("x\n" * 500, None, "whatever")
    assert media == "text/plain"
    assert "500 lines" in extract or "501 lines" in extract


# -- identity --------------------------------------------------------------


def test_read_and_write_tools_map_to_the_same_resource():
    r = ids.resolve("read_file", {"path": "./src/auth.py"})
    w = ids.resolve("write_file", {"file_path": "src/auth.py"})
    assert r.resource == ResourceId("file", "src/auth.py") and not r.is_write
    assert w.resource == ResourceId("file", "src/auth.py") and w.is_write


def test_shell_read_is_not_a_write_but_a_redirect_is():
    assert not ids.resolve("bash", {"command": "pytest tests/"}).is_write
    sem = ids.resolve("bash", {"command": "echo hi > out.txt"})
    assert sem.is_write and sem.resource == ResourceId("file", "out.txt")


def test_http_identity_includes_method():
    sem = ids.resolve("http_request", {"url": "https://x/users", "method": "post"})
    assert sem.resource == ResourceId("http", "POST https://x/users")


def test_unknown_tool_yields_no_identity():
    assert ids.resolve("frobnicate", {"widget": 3}).resource is None


def test_custom_extractor_takes_priority():
    ids.register("db_query", lambda n, a: ToolSemantics(ResourceId("db", a["table"])))
    assert ids.resolve("db_query", {"table": "users"}).resource == ResourceId("db", "users")
