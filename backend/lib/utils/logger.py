# utils/logger.py

import logging
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any

# Optional dependency - added to requirements.txt as pymongo
try:
    from pymongo import MongoClient
except Exception:  # pragma: no cover
    MongoClient = None  # type: ignore

# ------------------------------------------------------------------------------
# Mongo helpers
# ------------------------------------------------------------------------------

_MONGO_CLIENT: Optional["MongoClient"] = None


def get_mongo_client(uri: Optional[str] = None) -> Optional["MongoClient"]:
    """
    Get a cached MongoClient instance.
    Reads URI from env MONGO_URI if not provided. Defaults to mongodb://localhost:27017
    """
    global _MONGO_CLIENT
    if _MONGO_CLIENT is not None:
        return _MONGO_CLIENT

    if MongoClient is None:
        return None

    mongo_uri = uri or os.getenv("MONGO_URI") or "mongodb://localhost:27017"
    try:
        _MONGO_CLIENT = MongoClient(mongo_uri, tz_aware=True)
        return _MONGO_CLIENT
    except Exception:
        return None


def get_mongo_collection(collection_name: str, db_name: Optional[str] = None, uri: Optional[str] = None):
    """
    Return a Mongo collection handle.
    DB name from env TRADING_LOGS_DB or default 'trading_logs'
    """
    client = get_mongo_client(uri)
    if client is None:
        return None
    database_name = db_name or os.getenv("TRADING_LOGS_DB") or "trading_logs"
    db = client[database_name]
    return db[collection_name]


class MongoHandler(logging.Handler):
    """
    Logging handler that writes structured log records to MongoDB.
    """
    def __init__(self, collection_name: str = "app_logs", uri: Optional[str] = None, db_name: Optional[str] = None):
        super().__init__()
        self.collection_name = collection_name
        self.uri = uri
        self.db_name = db_name
        self.collection = get_mongo_collection(collection_name, db_name=self.db_name, uri=self.uri)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if self.collection is None:
                # Lazy re-init if client wasn't available at construction time
                self.collection = get_mongo_collection(self.collection_name, db_name=self.db_name, uri=self.uri)
                if self.collection is None:
                    return

            # Build structured document
            doc: Dict[str, Any] = {
                "timestamp": datetime.utcfromtimestamp(record.created),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "filename": record.filename,
                "funcName": record.funcName,
                "lineno": record.lineno,
                "process": record.process,
                "thread": record.thread,
            }

            # Capture extras (custom attributes) excluding standard ones
            std_keys = set(vars(logging.makeLogRecord({})).keys())
            extras = {k: v for k, v in record.__dict__.items() if k not in std_keys}
            if extras:
                # Avoid serializing non-serializable objects; convert to str as fallback
                safe_extras: Dict[str, Any] = {}
                for k, v in extras.items():
                    try:
                        _ = str(v) if isinstance(v, (object,)) else v
                        safe_extras[k] = v
                    except Exception:
                        safe_extras[k] = str(v)
                doc["extras"] = safe_extras

            # Insert document
            self.collection.insert_one(doc)
        except Exception:
            # Never raise from logging
            pass


# ------------------------------------------------------------------------------
# Logger setup
# ------------------------------------------------------------------------------

def setup_logger(
    name: str = "trading_bot",
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_dir: Optional[str] = None,
    # Mongo options
    log_to_mongo: bool = True,
    mongo_uri: Optional[str] = None,
    mongo_db: Optional[str] = None,
    mongo_collection: Optional[str] = None
) -> logging.Logger:
    """
    Advanced logger setup with MongoDB integration.

    Args:
        name: Logger name
        level: Logging level
        log_to_file: File logging flag (ignored when log_to_mongo=True unless USE_FILE_LOGS=1)
        log_to_console: Console logging flag
        log_dir: Legacy log directory for files
        log_to_mongo: Enable MongoDB logging (default: True)
        mongo_uri: Mongo URI (default from env MONGO_URI or mongodb://localhost:27017)
        mongo_db: Mongo DB name (default from env TRADING_LOGS_DB or 'trading_logs')
        mongo_collection: Target collection (default: derived from logger name)
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # UTF-8 console safety (Windows)
    try:
        os.environ["PYTHONIOENCODING"] = "UTF-8"
        os.environ["PYTHONUTF8"] = "1"
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Clean any FileHandlers if present to comply with "no file logs"
    if logger.handlers:
        new_handlers = []
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                continue  # drop file handlers
            new_handlers.append(h)
        logger.handlers = new_handlers

    # Console handler
    if log_to_console and not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stdout_stream = sys.stdout
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            import io
            if not (getattr(stdout_stream, "encoding", "") or "").lower().startswith("utf"):
                stdout_stream = io.TextIOWrapper(stdout_stream.buffer, encoding="utf-8", errors="replace")
        except Exception:
            pass
        ch = logging.StreamHandler(stdout_stream)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Mongo handler
    if log_to_mongo and MongoClient is not None:
        # Choose collection name if not passed
        coll = mongo_collection
        if not coll:
            base = name.lower()
            if "trade" in base:
                coll = "trade_logs"
            elif "performance" in base:
                coll = "performance_logs"
            elif "error" in base:
                coll = "error_logs"
            else:
                coll = "app_logs"

        if not any(isinstance(h, MongoHandler) for h in logger.handlers):
            mh = MongoHandler(collection_name=coll, uri=mongo_uri, db_name=mongo_db)
            mh.setLevel(level)
            logger.addHandler(mh)

    # File handler (explicit opt-in only)
    if log_to_file and os.getenv("USE_FILE_LOGS", "0") == "1":
        resolved_dir = log_dir or "logs"
        os.makedirs(resolved_dir, exist_ok=True)
        log_file = os.path.join(resolved_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_performance_logger(log_dir: Optional[str] = None) -> logging.Logger:
    """Performance logger routed to MongoDB collection 'performance_logs'"""
    return setup_logger(
        "performance",
        logging.INFO,
        log_to_file=False,
        log_to_console=True,
        log_dir=log_dir,
        log_to_mongo=True,
        mongo_collection="performance_logs"
    )


def get_trade_logger(log_dir: Optional[str] = None) -> logging.Logger:
    """Trade logger routed to MongoDB collection 'trade_logs'"""
    return setup_logger(
        "trades",
        logging.INFO,
        log_to_file=False,
        log_to_console=True,
        log_dir=log_dir,
        log_to_mongo=True,
        mongo_collection="trade_logs"
    )


def get_error_logger(log_dir: Optional[str] = None) -> logging.Logger:
    """Error logger routed to MongoDB collection 'error_logs'"""
    return setup_logger(
        "errors",
        logging.ERROR,
        log_to_file=False,
        log_to_console=True,
        log_dir=log_dir,
        log_to_mongo=True,
        mongo_collection="error_logs"
    )