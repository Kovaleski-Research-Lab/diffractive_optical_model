import pytest
from loguru import logger


@pytest.fixture(autouse=True)
def _silence_loguru():
    logger.remove()
    yield
