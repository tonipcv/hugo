from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import sqlite3
from datetime import datetime
import hashlib


class Leaderboard:
    """
    Community leaderboard system for tracking and comparing RL agent performance.
    
    Features:
    - Environment-specific leaderboards
    - Algorithm comparisons
    - Reproducibility tracking
    - Community submissions
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path.home() / ".rl_llm_toolkit" / "leaderboard.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database for leaderboard."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                env_name TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                mean_reward REAL NOT NULL,
                std_reward REAL NOT NULL,
                total_timesteps INTEGER,
                model_id TEXT,
                username TEXT,
                submission_date TEXT,
                config_hash TEXT,
                hyperparameters TEXT,
                llm_config TEXT,
                verified BOOLEAN DEFAULT 0,
                UNIQUE(env_name, algorithm, config_hash)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_env_reward 
            ON submissions(env_name, mean_reward DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_algorithm 
            ON submissions(algorithm)
        """)
        
        conn.commit()
        conn.close()
    
    def submit(
        self,
        env_name: str,
        algorithm: str,
        mean_reward: float,
        std_reward: float,
        total_timesteps: int,
        hyperparameters: Dict[str, Any],
        model_id: Optional[str] = None,
        username: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Submit a new result to the leaderboard.
        
        Returns:
            Submission ID
        """
        config_hash = self._hash_config(hyperparameters)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO submissions 
                (env_name, algorithm, mean_reward, std_reward, total_timesteps,
                 model_id, username, submission_date, config_hash, 
                 hyperparameters, llm_config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                env_name,
                algorithm,
                mean_reward,
                std_reward,
                total_timesteps,
                model_id,
                username or "anonymous",
                datetime.now().isoformat(),
                config_hash,
                json.dumps(hyperparameters),
                json.dumps(llm_config) if llm_config else None,
            ))
            
            submission_id = cursor.lastrowid
            conn.commit()
            
            print(f"✅ Submission #{submission_id} added to leaderboard")
            return submission_id
        
        except sqlite3.IntegrityError:
            print("⚠️  Identical submission already exists")
            cursor.execute("""
                SELECT id FROM submissions 
                WHERE env_name = ? AND algorithm = ? AND config_hash = ?
            """, (env_name, algorithm, config_hash))
            
            result = cursor.fetchone()
            return result[0] if result else -1
        
        finally:
            conn.close()
    
    def get_leaderboard(
        self,
        env_name: str,
        limit: int = 10,
        algorithm: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get top submissions for an environment.
        
        Args:
            env_name: Environment name
            limit: Number of results to return
            algorithm: Filter by algorithm (optional)
            
        Returns:
            List of top submissions
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT id, algorithm, mean_reward, std_reward, total_timesteps,
                   model_id, username, submission_date, verified
            FROM submissions
            WHERE env_name = ?
        """
        params = [env_name]
        
        if algorithm:
            query += " AND algorithm = ?"
            params.append(algorithm)
        
        query += " ORDER BY mean_reward DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "algorithm": row[1],
                "mean_reward": row[2],
                "std_reward": row[3],
                "total_timesteps": row[4],
                "model_id": row[5],
                "username": row[6],
                "submission_date": row[7],
                "verified": bool(row[8]),
            })
        
        conn.close()
        return results
    
    def get_submission(self, submission_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a submission."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM submissions WHERE id = ?
        """, (submission_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return {
            "id": row[0],
            "env_name": row[1],
            "algorithm": row[2],
            "mean_reward": row[3],
            "std_reward": row[4],
            "total_timesteps": row[5],
            "model_id": row[6],
            "username": row[7],
            "submission_date": row[8],
            "config_hash": row[9],
            "hyperparameters": json.loads(row[10]) if row[10] else {},
            "llm_config": json.loads(row[11]) if row[11] else None,
            "verified": bool(row[12]),
        }
    
    def compare_algorithms(
        self,
        env_name: str,
        algorithms: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare performance of different algorithms on an environment.
        
        Returns:
            Dictionary mapping algorithm names to their best performance
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT algorithm, MAX(mean_reward) as best_reward,
                   AVG(mean_reward) as avg_reward,
                   COUNT(*) as num_submissions
            FROM submissions
            WHERE env_name = ?
        """
        params = [env_name]
        
        if algorithms:
            placeholders = ",".join("?" * len(algorithms))
            query += f" AND algorithm IN ({placeholders})"
            params.extend(algorithms)
        
        query += " GROUP BY algorithm ORDER BY best_reward DESC"
        
        cursor.execute(query, params)
        
        results = {}
        for row in cursor.fetchall():
            results[row[0]] = {
                "best_reward": row[1],
                "avg_reward": row[2],
                "num_submissions": row[3],
            }
        
        conn.close()
        return results
    
    def verify_submission(self, submission_id: int) -> bool:
        """Mark a submission as verified (for reproducibility)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE submissions SET verified = 1 WHERE id = ?
        """, (submission_id,))
        
        conn.commit()
        conn.close()
        
        return True
    
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Create hash of configuration for deduplication."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def export_leaderboard(
        self,
        env_name: str,
        output_path: Path,
        format: str = "json",
    ) -> None:
        """Export leaderboard to file."""
        leaderboard = self.get_leaderboard(env_name, limit=100)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(leaderboard, f, indent=2)
        
        elif format == "markdown":
            with open(output_path, "w") as f:
                f.write(f"# Leaderboard: {env_name}\n\n")
                f.write("| Rank | Algorithm | Mean Reward | Std | Timesteps | User | Date |\n")
                f.write("|------|-----------|-------------|-----|-----------|------|------|\n")
                
                for i, entry in enumerate(leaderboard, 1):
                    f.write(
                        f"| {i} | {entry['algorithm']} | "
                        f"{entry['mean_reward']:.2f} | {entry['std_reward']:.2f} | "
                        f"{entry['total_timesteps']} | {entry['username']} | "
                        f"{entry['submission_date'][:10]} |\n"
                    )
        
        print(f"✅ Leaderboard exported to {output_path}")
