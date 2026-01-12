import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class DatabaseConnector:
    """Robust database connector dengan error handling"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conn = None
    
    def connect(self) -> bool:
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 5432),
                database=self.db_config.get('database', 'samsara_db'),
                user=self.db_config.get('user', 'postgres'),
                password=self.db_config.get('password', ''),
                connect_timeout=5
            )
            logger.info(" Connected to PostgreSQL successfully")
            return True
        except psycopg2.OperationalError as e:
            logger.error(f" Connection failed: {e}")
            return False
        except Exception as e:
            logger.error(f" Unexpected error: {e}")
            return False
    
    def execute_query(self, query: str, fetch: bool = True) -> Optional[pd.DataFrame]:
        """Execute query safely"""
        if not self.conn:
            if not self.connect():
                return None
        
        try:
            cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query)
            
            if fetch:
                rows = cursor.fetchall()
                cursor.close()
                
                if rows:
                    return pd.DataFrame(rows)
                else:
                    logger.warning("Query returned no results")
                    return None
            else:
                self.conn.commit()
                cursor.close()
                logger.info(f" Query executed successfully")
                return None
                
        except psycopg2.ProgrammingError as e:
            logger.error(f" SQL Error: {e}")
            return None
        except Exception as e:
            logger.error(f" Unexpected error: {e}")
            return None
    
    def check_table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        query = f"""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name = '{table_name}'
        );
        """
        result = self.execute_query(query)
        return result.iloc[0, 0] if result is not None else False
    
    def check_column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if column exists in table"""
        query = f"""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name = '{table_name}' 
            AND column_name = '{column_name}'
        );
        """
        result = self.execute_query(query)
        return result.iloc[0, 0] if result is not None else False
    
    def get_table_stats(self, table_name: str) -> Optional[Dict]:
        """Get table statistics"""
        query = f"""
        SELECT 
            '{table_name}' as table_name,
            COUNT(*) as row_count,
            (pg_total_relation_size('{table_name}') / 1024.0 / 1024.0)::NUMERIC(10,2) as size_mb
        FROM {table_name};
        """
        result = self.execute_query(query)
        return result.to_dict('records')[0] if result is not None else None
    
    def close(self):
        """Close connection"""
        if self.conn:
            self.conn.close()
            logger.info(" Connection closed")