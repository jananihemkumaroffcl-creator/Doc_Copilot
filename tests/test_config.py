from copilot.config import settings


def test_defaults_sane():
    assert settings.chunk_size > settings.chunk_overlap
    assert settings.top_k >= 1
