import sys
from loguru import logger

logger.remove()

# Console logger configuration
console_logger = logger.bind(source="console")
console_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"
)

# Configure console logger
console_logger.add(
    sys.stderr,
    format=console_format,
    level="DEBUG",
    filter=lambda r: "console" == r["extra"].get("source"),
)

# File logger configuration
file_logger = logger.bind(source="file")
file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"

# Create a new file logger
# file_logger.remove()  # Remove any default handlers
file_logger.add(
    "app.log",
    rotation="1 day",
    format=file_format,
    level="DEBUG",
    filter=lambda r: "file" == r["extra"].get("source"),
)
