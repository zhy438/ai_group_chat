import psycopg2
from psycopg2.extras import RealDictCursor
import time
from loguru import logger
from typing import Optional, Any

# Docker Compose Configuration
DB_CONFIG = {
    "dbname": "ai_chat_db",
    "user": "admin",
    "password": "password",
    "host": "localhost",
    "port": "5432"
}

class Database:
    def __init__(self):
        self._wait_for_db()
        self._init_db()

    def _get_conn(self):
        return psycopg2.connect(**DB_CONFIG)

    def _wait_for_db(self):
        """Wait for Postgres availability"""
        retries = 5
        while retries > 0:
            try:
                conn = self._get_conn()
                conn.close()
                return
            except psycopg2.OperationalError as e:
                logger.warning(f"⏳ Waiting for Postgres... ({e})")
                time.sleep(2)
                retries -= 1
        logger.error("❌ Could not connect to Postgres database.")

    def _init_db(self):
        """Initialize Postgres schema"""
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            
            # Groups Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS groups (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    manager_model TEXT,
                    manager_thinking BOOLEAN DEFAULT FALSE,
                    manager_temperature REAL DEFAULT 0.7,
                    discussion_mode TEXT DEFAULT 'free',
                    max_rounds INTEGER DEFAULT 10
                )
            """)
            
            # Members Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS members (
                    id TEXT PRIMARY KEY,
                    group_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    description TEXT,
                    persona TEXT,
                    thinking BOOLEAN DEFAULT FALSE,
                    temperature REAL DEFAULT 0.7,
                    FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE
                )
            """)
            
            # Messages Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    group_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sender_name TEXT,
                    mode TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_type TEXT DEFAULT 'normal',
                    is_compressed BOOLEAN DEFAULT FALSE,
                    original_content TEXT,
                    value_score REAL,
                    FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE
                )
            """)
            
            # Context Snapshots Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS group_context_snapshots (
                    id SERIAL PRIMARY KEY,
                    group_id TEXT NOT NULL,
                    last_message_id TEXT,
                    context_content TEXT NOT NULL,  -- JSON serialized list of messages
                    token_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE
                )
            """)
            conn.commit()  # <--- Fix: Commit immediately to prevent rollback by subsequent errors
            
            # 为已存在的表添加新字段（兼容性处理）
            memory_columns = [
                ("message_type", "TEXT DEFAULT 'normal'"),
                ("is_compressed", "BOOLEAN DEFAULT FALSE"),
                ("original_content", "TEXT"),
                ("value_score", "REAL"),
            ]
            for col_name, col_type in memory_columns:
                try:
                    cur.execute(f"ALTER TABLE messages ADD COLUMN {col_name} {col_type}")
                except psycopg2.errors.DuplicateColumn:
                    pass  # Column already exists
                conn.commit()
                
            # compression_threshold
            try:
                cur.execute("ALTER TABLE groups ADD COLUMN compression_threshold REAL DEFAULT 0.8")
                conn.commit()
            except psycopg2.errors.DuplicateColumn:
                conn.rollback()
            
            conn.commit()
            conn.close()
            logger.info(f"🔗 数据库: postgresql://{DB_CONFIG['user']}:***@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
        except Exception as e:
            logger.error(f"❌ Database Init Failed: {e}")

    def execute(self, sql: str, params: tuple = ()) -> Any:
        """Execute SQL statement (INSERT/UPDATE/DELETE)"""
        # Auto-transpile SQLite ? placeholder to Postgres %s
        pg_sql = sql.replace('?', '%s')
        
        conn = self._get_conn()
        try:
            cur = conn.cursor()
            cur.execute(pg_sql, params)
            conn.commit()
            return cur  # Returns cursor which has rowcount
        except Exception as e:
            conn.rollback()
            logger.error(f"SQL Error: {e} | SQL: {pg_sql} | Params: {params}")
            raise e
        finally:
            conn.close()

    def fetch_all(self, sql: str, params: tuple = ()) -> list[dict]:
        """Fetch multiple rows"""
        pg_sql = sql.replace('?', '%s')
        conn = self._get_conn()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(pg_sql, params)
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def fetch_one(self, sql: str, params: tuple = ()) -> Optional[dict]:
        """Fetch single row"""
        pg_sql = sql.replace('?', '%s')
        conn = self._get_conn()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(pg_sql, params)
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

# Global DB Instance
db = Database()
