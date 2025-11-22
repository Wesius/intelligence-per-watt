from ipw.clients import ensure_registered
from ipw.core.registry import ClientRegistry


def test_ensure_registered_registers_openai_client() -> None:
    # Ensure a clean registry for the test run
    ClientRegistry.clear()
    try:
        ensure_registered()
        client_cls = ClientRegistry.get("openai")
        assert getattr(client_cls, "client_id", None) == "openai"
    finally:
        ClientRegistry.clear()
