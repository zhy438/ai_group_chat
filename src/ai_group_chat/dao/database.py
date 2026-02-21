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
                    max_rounds INTEGER DEFAULT 10,
                    compression_threshold REAL DEFAULT 0.8,
                    memory_enabled BOOLEAN DEFAULT TRUE,
                    archive_enabled BOOLEAN DEFAULT TRUE,
                    retrieve_enabled BOOLEAN DEFAULT TRUE,
                    scope_user_global BOOLEAN DEFAULT TRUE,
                    scope_group_local BOOLEAN DEFAULT TRUE,
                    scope_agent_local BOOLEAN DEFAULT TRUE,
                    memory_injection_ratio REAL DEFAULT 0.2,
                    memory_top_n INTEGER DEFAULT 5,
                    memory_min_confidence REAL DEFAULT 0.75,
                    memory_score_threshold REAL DEFAULT 0.35
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
                    sender_id TEXT,
                    user_id TEXT DEFAULT 'default-user',
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

            # 长期记忆主表
            cur.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id TEXT PRIMARY KEY,
                    group_id TEXT,
                    user_id TEXT NOT NULL,
                    member_id TEXT,
                    scope TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL DEFAULT 0.8,
                    fingerprint TEXT NOT NULL,
                    persona_version TEXT,
                    source_message_id TEXT,
                    source_created_at TIMESTAMP,
                    last_used_at TIMESTAMP,
                    decay_score REAL DEFAULT 1.0,
                    is_active BOOLEAN DEFAULT TRUE,
                    expires_at TIMESTAMP,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_ltm_fingerprint_scope
                ON long_term_memories(
                    scope,
                    user_id,
                    COALESCE(group_id, ''),
                    COALESCE(member_id, ''),
                    COALESCE(persona_version, ''),
                    fingerprint
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_ltm_scope_lookup
                ON long_term_memories(scope, user_id, group_id, member_id, persona_version, is_active)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_ltm_group_lookup
                ON long_term_memories(group_id, scope, is_active, updated_at)
            """)

            # 增量归档游标
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memory_checkpoints (
                    group_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    last_message_id TEXT,
                    last_message_created_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (group_id, user_id),
                    FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE
                )
            """)

            # 失败任务（死信）
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memory_dead_letters (
                    id SERIAL PRIMARY KEY,
                    group_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    error TEXT,
                    payload TEXT,
                    retry_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 审计日志
            cur.execute("""
                CREATE TABLE IF NOT EXISTS memory_audit_logs (
                    id SERIAL PRIMARY KEY,
                    request_id TEXT,
                    group_id TEXT,
                    user_id TEXT,
                    event_type TEXT NOT NULL,
                    scope TEXT,
                    memory_ids TEXT,
                    detail TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

            # 尝试启用 pgvector（可选能力，失败不阻塞启动）
            vector_enabled = self._try_enable_pgvector(conn, cur)

            # ===== 兼容性：对旧库补列 =====
            message_columns = [
                ("message_type", "TEXT DEFAULT 'normal'"),
                ("is_compressed", "BOOLEAN DEFAULT FALSE"),
                ("original_content", "TEXT"),
                ("value_score", "REAL"),
                ("sender_id", "TEXT"),
                ("user_id", "TEXT DEFAULT 'default-user'"),
            ]
            for col_name, col_type in message_columns:
                self._safe_add_column(conn, cur, "messages", col_name, col_type)

            group_columns = [
                ("compression_threshold", "REAL DEFAULT 0.8"),
                ("memory_enabled", "BOOLEAN DEFAULT TRUE"),
                ("archive_enabled", "BOOLEAN DEFAULT TRUE"),
                ("retrieve_enabled", "BOOLEAN DEFAULT TRUE"),
                ("scope_user_global", "BOOLEAN DEFAULT TRUE"),
                ("scope_group_local", "BOOLEAN DEFAULT TRUE"),
                ("scope_agent_local", "BOOLEAN DEFAULT TRUE"),
                ("memory_injection_ratio", "REAL DEFAULT 0.2"),
                ("memory_top_n", "INTEGER DEFAULT 5"),
                ("memory_min_confidence", "REAL DEFAULT 0.75"),
                ("memory_score_threshold", "REAL DEFAULT 0.35"),
            ]
            for col_name, col_type in group_columns:
                self._safe_add_column(conn, cur, "groups", col_name, col_type)

            if vector_enabled:
                self._safe_add_column(conn, cur, "long_term_memories", "embedding", "VECTOR(1536)")
                self._safe_add_column(conn, cur, "long_term_memories", "embedding_model", "TEXT")
                self._safe_add_column(conn, cur, "long_term_memories", "embedding_updated_at", "TIMESTAMP")
                self._safe_execute(
                    conn,
                    cur,
                    """
                    CREATE INDEX IF NOT EXISTS idx_ltm_embedding_ivfflat
                    ON long_term_memories
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                    """,
                )
            
            conn.commit()
            conn.close()
            logger.info(f"🔗 数据库: postgresql://{DB_CONFIG['user']}:***@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}")
        except Exception as e:
            logger.error(f"❌ Database Init Failed: {e}")

    @staticmethod
    def _try_enable_pgvector(conn, cur) -> bool:
        """尝试启用 pgvector 扩展"""
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
            logger.info("✅ pgvector 扩展已启用")
            return True
        except Exception as e:
            conn.rollback()
            logger.warning(f"⚠️ pgvector 扩展不可用，将降级为非向量检索: {e}")
            return False

    @staticmethod
    def _safe_add_column(conn, cur, table_name: str, col_name: str, col_type: str) -> None:
        """为已存在表安全补列（重复列自动忽略）"""
        try:
            cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}")
            conn.commit()
        except psycopg2.errors.DuplicateColumn:
            conn.rollback()

    @staticmethod
    def _safe_execute(conn, cur, sql: str) -> None:
        """安全执行 DDL，异常时仅回滚不抛出"""
        try:
            cur.execute(sql)
            conn.commit()
        except Exception:
            conn.rollback()

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
