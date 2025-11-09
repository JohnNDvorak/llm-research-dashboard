"""SQLite database operations for papers."""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from src.utils.logger import logger


class PaperDB:
    """CRUD operations for papers database."""

    def __init__(self, db_path: str = "data/papers.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Establish database connection."""
        # Create data directory if it doesn't exist
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, timeout=30.0)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        logger.info(f"Connected to database: {self.db_path}")

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def execute_migration(self, migration_path: str) -> None:
        """
        Execute SQL migration file.

        Args:
            migration_path: Path to SQL migration file

        Raises:
            FileNotFoundError: If migration file doesn't exist
            sqlite3.Error: If SQL execution fails
        """
        migration_file = Path(migration_path)
        if not migration_file.exists():
            raise FileNotFoundError(f"Migration file not found: {migration_path}")

        with open(migration_file, 'r', encoding='utf-8') as f:
            sql = f.read()

        if not self.conn:
            self.connect()

        try:
            self.conn.executescript(sql)
            self.conn.commit()
            logger.info(f"Migration executed successfully: {migration_path}")
        except sqlite3.Error as e:
            logger.error(f"Migration failed: {e}")
            raise

    def create_tables(self) -> None:
        """Create database tables from schema."""
        schema_path = Path(__file__).parent / "migrations" / "001_initial_schema.sql"
        self.execute_migration(str(schema_path))

    def insert_paper(self, paper: Dict[str, Any]) -> str:
        """
        Insert a paper into the database.

        Args:
            paper: Paper dictionary with fields matching schema

        Returns:
            Paper ID of inserted paper

        Raises:
            ValueError: If required fields are missing
            sqlite3.Error: If insertion fails
        """
        if not paper.get('id'):
            raise ValueError("Paper must have an 'id' field")
        if not paper.get('title'):
            raise ValueError("Paper must have a 'title' field")
        if not paper.get('abstract'):
            raise ValueError("Paper must have an 'abstract' field")

        if not self.conn:
            self.connect()

        # Serialize JSON fields
        paper_data = paper.copy()
        if 'authors' in paper_data and isinstance(paper_data['authors'], list):
            paper_data['authors'] = json.dumps(paper_data['authors'])
        if 'stages' in paper_data and isinstance(paper_data['stages'], list):
            paper_data['stages'] = json.dumps(paper_data['stages'])
        if 'key_insights' in paper_data and isinstance(paper_data['key_insights'], list):
            paper_data['key_insights'] = json.dumps(paper_data['key_insights'])
        if 'metrics' in paper_data and isinstance(paper_data['metrics'], dict):
            paper_data['metrics'] = json.dumps(paper_data['metrics'])

        # Build dynamic INSERT query based on provided fields
        columns = list(paper_data.keys())
        placeholders = ['?' for _ in columns]
        values = [paper_data[col] for col in columns]

        query = f"""
        INSERT OR REPLACE INTO papers ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        """

        try:
            cursor = self.conn.cursor()
            cursor.execute(query, values)
            self.conn.commit()
            logger.info(f"Inserted paper: {paper['id']}")
            return paper['id']
        except sqlite3.Error as e:
            logger.error(f"Failed to insert paper {paper.get('id')}: {e}")
            raise

    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a paper by ID.

        Args:
            paper_id: Paper ID to retrieve

        Returns:
            Paper dictionary or None if not found
        """
        if not self.conn:
            self.connect()

        query = "SELECT * FROM papers WHERE id = ?"

        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (paper_id,))
            row = cursor.fetchone()

            if not row:
                return None

            # Convert Row to dict and deserialize JSON fields
            paper = dict(row)

            # Deserialize JSON fields
            if paper.get('authors'):
                try:
                    paper['authors'] = json.loads(paper['authors'])
                except (json.JSONDecodeError, TypeError):
                    pass

            if paper.get('stages'):
                try:
                    paper['stages'] = json.loads(paper['stages'])
                except (json.JSONDecodeError, TypeError):
                    pass

            if paper.get('key_insights'):
                try:
                    paper['key_insights'] = json.loads(paper['key_insights'])
                except (json.JSONDecodeError, TypeError):
                    pass

            if paper.get('metrics'):
                try:
                    paper['metrics'] = json.loads(paper['metrics'])
                except (json.JSONDecodeError, TypeError):
                    pass

            return paper
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve paper {paper_id}: {e}")
            raise

    def get_all_papers(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all papers with optional filtering.

        Args:
            limit: Maximum number of papers to return
            offset: Number of papers to skip
            filters: Optional filters (e.g., {'source': 'arxiv', 'analyzed': True})

        Returns:
            List of paper dictionaries
        """
        if not self.conn:
            self.connect()

        query = "SELECT * FROM papers"
        params = []

        # Add filters
        if filters:
            where_clauses = []
            for key, value in filters.items():
                where_clauses.append(f"{key} = ?")
                params.append(value)

            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)

        # Add ordering
        query += " ORDER BY created_at DESC"

        # Add pagination
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        if offset:
            query += " OFFSET ?"
            params.append(offset)

        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

            papers = []
            for row in rows:
                paper = dict(row)

                # Deserialize JSON fields
                for json_field in ['authors', 'stages', 'key_insights', 'metrics']:
                    if paper.get(json_field):
                        try:
                            paper[json_field] = json.loads(paper[json_field])
                        except (json.JSONDecodeError, TypeError):
                            pass

                papers.append(paper)

            return papers
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve papers: {e}")
            raise

    def update_paper(self, paper_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update paper fields.

        Args:
            paper_id: Paper ID to update
            updates: Dictionary of fields to update

        Returns:
            True if update successful, False if paper not found

        Raises:
            sqlite3.Error: If update fails
        """
        if not self.conn:
            self.connect()

        # Serialize JSON fields
        updates_data = updates.copy()
        for field in ['authors', 'stages', 'key_insights']:
            if field in updates_data and isinstance(updates_data[field], list):
                updates_data[field] = json.dumps(updates_data[field])
        if 'metrics' in updates_data and isinstance(updates_data['metrics'], dict):
            updates_data['metrics'] = json.dumps(updates_data['metrics'])

        # Add updated_at timestamp
        updates_data['updated_at'] = datetime.now(timezone.utc).isoformat()

        # Build UPDATE query
        set_clauses = [f"{key} = ?" for key in updates_data.keys()]
        values = list(updates_data.values()) + [paper_id]

        query = f"""
        UPDATE papers
        SET {', '.join(set_clauses)}
        WHERE id = ?
        """

        try:
            cursor = self.conn.cursor()
            cursor.execute(query, values)
            self.conn.commit()

            if cursor.rowcount == 0:
                logger.warning(f"Paper not found for update: {paper_id}")
                return False

            logger.info(f"Updated paper: {paper_id}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to update paper {paper_id}: {e}")
            raise

    def delete_paper(self, paper_id: str) -> bool:
        """
        Delete a paper from database.

        Args:
            paper_id: Paper ID to delete

        Returns:
            True if deleted, False if not found

        Raises:
            sqlite3.Error: If deletion fails
        """
        if not self.conn:
            self.connect()

        query = "DELETE FROM papers WHERE id = ?"

        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (paper_id,))
            self.conn.commit()

            if cursor.rowcount == 0:
                logger.warning(f"Paper not found for deletion: {paper_id}")
                return False

            logger.info(f"Deleted paper: {paper_id}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to delete paper {paper_id}: {e}")
            raise

    def paper_exists(self, paper_id: str) -> bool:
        """
        Check if paper exists in database.

        Args:
            paper_id: Paper ID to check

        Returns:
            True if paper exists, False otherwise
        """
        if not self.conn:
            self.connect()

        query = "SELECT 1 FROM papers WHERE id = ? LIMIT 1"

        try:
            cursor = self.conn.cursor()
            cursor.execute(query, (paper_id,))
            return cursor.fetchone() is not None
        except sqlite3.Error as e:
            logger.error(f"Failed to check paper existence {paper_id}: {e}")
            raise

    def get_paper_count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Get total count of papers.

        Args:
            filters: Optional filters

        Returns:
            Count of papers matching filters
        """
        if not self.conn:
            self.connect()

        query = "SELECT COUNT(*) FROM papers"
        params = []

        if filters:
            where_clauses = []
            for key, value in filters.items():
                where_clauses.append(f"{key} = ?")
                params.append(value)

            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)

        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()[0]
        except sqlite3.Error as e:
            logger.error(f"Failed to count papers: {e}")
            raise

    def insert_cost_record(self, cost_data: Dict[str, Any]) -> int:
        """
        Insert cost tracking record.

        Args:
            cost_data: Cost record data

        Returns:
            ID of inserted record

        Raises:
            sqlite3.Error: If insertion fails
        """
        if not self.conn:
            self.connect()

        query = """
        INSERT INTO cost_tracking
        (provider, model, paper_id, operation_type, input_tokens, output_tokens, cost)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        values = (
            cost_data.get('provider'),
            cost_data.get('model'),
            cost_data.get('paper_id'),
            cost_data.get('operation_type'),
            cost_data.get('input_tokens', 0),
            cost_data.get('output_tokens', 0),
            cost_data.get('cost', 0.0)
        )

        try:
            cursor = self.conn.cursor()
            cursor.execute(query, values)
            self.conn.commit()
            logger.info(f"Inserted cost record for {cost_data.get('operation_type')}")
            return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Failed to insert cost record: {e}")
            raise
