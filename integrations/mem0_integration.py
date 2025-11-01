"""
Mem0 AI Memory Integration

Stores and retrieves project memories for continuity and learning.
Tracks: progress, decisions, learnings, issues, resolutions
"""

import requests
import json
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Mem0Client:
    """
    Mem0 AI client for storing and retrieving project memories.

    Tracks:
    - Project progress through stages
    - Technical decisions and rationale
    - Issues encountered and resolutions
    - Model performance metrics
    - Integration learnings
    - Configuration preferences
    """

    def __init__(self, api_key: str, user_id: str = "defi-neural-network", api_version: str = "v2"):
        self.api_key = api_key
        self.user_id = user_id
        self.api_version = api_version
        self.base_url = "https://api.mem0.ai"
        self.headers = {
            'Authorization': f'Token {api_key}',
            'Content-Type': 'application/json'
        }
        self._validate_connection()

    def _validate_connection(self) -> bool:
        """Validate Mem0 API connection."""
        try:
            response = requests.get(
                f"{self.base_url}/{self.api_version}/memories/",
                headers=self.headers,
                timeout=5
            )
            if response.status_code in [200, 401]:
                logger.info("✓ Mem0 API connection validated")
                return True
            else:
                logger.warning(f"Mem0 connection status: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Mem0 connection failed: {e}")
            return False

    def add_memory(self, messages: List[Dict], tags: Optional[List[str]] = None) -> Dict:
        """
        Add a memory to Mem0.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tags: Optional list of tags for categorization

        Returns:
            Response from Mem0 API
        """
        try:
            data = {
                "messages": messages,
                "user_id": self.user_id,
                "version": self.api_version
            }

            if tags:
                data["tags"] = tags

            response = requests.post(
                f"{self.base_url}/{self.api_version}/memories/",
                headers=self.headers,
                json=data,
                timeout=10
            )

            if response.status_code == 201:
                logger.info(f"✓ Memory added successfully")
                return response.json()
            else:
                logger.warning(f"Memory add failed: {response.status_code}")
                return {}

        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return {}

    def search_memories(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for memories.

        Args:
            query: Search query
            limit: Maximum results to return

        Returns:
            List of matching memories
        """
        try:
            data = {
                "query": query,
                "filters": {
                    "OR": [{"user_id": self.user_id}]
                },
                "limit": limit
            }

            response = requests.post(
                f"{self.base_url}/{self.api_version}/memories/search/",
                headers=self.headers,
                json=data,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                memories = result.get('results', [])
                logger.info(f"✓ Found {len(memories)} memories")
                return memories
            else:
                logger.warning(f"Search failed: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []

    def get_all_memories(self, limit: int = 100) -> List[Dict]:
        """Get all memories for this user."""
        try:
            response = requests.get(
                f"{self.base_url}/{self.api_version}/memories/",
                headers=self.headers,
                params={"limit": limit},
                timeout=10
            )

            if response.status_code == 200:
                return response.json().get('results', [])
            else:
                return []

        except Exception as e:
            logger.error(f"Error getting memories: {e}")
            return []

    def add_progress_memory(self, stage: int, status: str, metrics: Dict) -> bool:
        """
        Add a progress memory for a stage.

        Args:
            stage: Stage number
            status: Current status
            metrics: Performance metrics

        Returns:
            Success flag
        """
        messages = [
            {
                "role": "user",
                "content": f"Update on Stage {stage} progress"
            },
            {
                "role": "assistant",
                "content": f"""Stage {stage} Status: {status}

Metrics:
{json.dumps(metrics, indent=2)}

Timestamp: {datetime.now().isoformat()}
"""
            }
        ]

        tags = [f"stage-{stage}", "progress", status.lower()]
        result = self.add_memory(messages, tags)
        return bool(result)

    def add_technical_decision(self, decision: str, rationale: str, alternatives: List[str]) -> bool:
        """
        Add a technical decision memory.

        Args:
            decision: The decision made
            rationale: Why this decision was made
            alternatives: Other options considered

        Returns:
            Success flag
        """
        messages = [
            {
                "role": "user",
                "content": f"Technical decision: {decision}"
            },
            {
                "role": "assistant",
                "content": f"""Decision: {decision}

Rationale:
{rationale}

Alternatives Considered:
{json.dumps(alternatives, indent=2)}

Timestamp: {datetime.now().isoformat()}
"""
            }
        ]

        tags = ["decision", "technical", decision.lower().replace(" ", "-")]
        result = self.add_memory(messages, tags)
        return bool(result)

    def add_issue_resolution(self, issue: str, cause: str, solution: str, stage: int) -> bool:
        """
        Add an issue and its resolution.

        Args:
            issue: Description of the issue
            cause: Root cause
            solution: Solution applied
            stage: Which stage this occurred in

        Returns:
            Success flag
        """
        messages = [
            {
                "role": "user",
                "content": f"Issue encountered: {issue}"
            },
            {
                "role": "assistant",
                "content": f"""Issue: {issue}

Root Cause:
{cause}

Solution Applied:
{solution}

Stage: {stage}
Timestamp: {datetime.now().isoformat()}
"""
            }
        ]

        tags = [f"stage-{stage}", "issue", "resolution"]
        result = self.add_memory(messages, tags)
        return bool(result)

    def add_model_performance(self, model_name: str, metrics: Dict, stage: int) -> bool:
        """
        Add model performance metrics.

        Args:
            model_name: Name of the model
            metrics: Performance metrics dict
            stage: Which stage

        Returns:
            Success flag
        """
        messages = [
            {
                "role": "user",
                "content": f"Performance metrics for {model_name}"
            },
            {
                "role": "assistant",
                "content": f"""Model: {model_name}
Stage: {stage}

Performance Metrics:
{json.dumps(metrics, indent=2)}

Timestamp: {datetime.now().isoformat()}
"""
            }
        ]

        tags = ["model", "performance", model_name.lower(), f"stage-{stage}"]
        result = self.add_memory(messages, tags)
        return bool(result)

    def get_stage_memories(self, stage: int) -> List[Dict]:
        """Get all memories for a specific stage."""
        query = f"Stage {stage} progress and decisions"
        return self.search_memories(query, limit=20)

    def get_recent_decisions(self, limit: int = 5) -> List[Dict]:
        """Get recent technical decisions."""
        return self.search_memories("technical decision", limit=limit)

    def get_issues_and_resolutions(self) -> List[Dict]:
        """Get all recorded issues and their resolutions."""
        return self.search_memories("issue resolution", limit=50)

    def generate_summary(self) -> str:
        """Generate a summary of all memories."""
        try:
            memories = self.get_all_memories(limit=200)
            summary = f"""
Project Memory Summary
======================

Total Memories: {len(memories)}

Recent Entries:
"""
            for memory in memories[-10:]:
                summary += f"\n- {memory.get('content', 'No content')[:100]}..."

            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Unable to generate summary"


# Singleton instance
_mem0_instance = None


def get_mem0_client(api_key: str, user_id: str = "defi-neural-network") -> Optional[Mem0Client]:
    """Get or create global Mem0 client."""
    global _mem0_instance
    if _mem0_instance is None:
        _mem0_instance = Mem0Client(api_key, user_id)
    return _mem0_instance


def log_stage_completion(api_key: str, stage: int, metrics: Dict) -> bool:
    """
    Convenience function to log stage completion to Mem0.

    Args:
        api_key: Mem0 API key
        stage: Stage number
        metrics: Performance metrics

    Returns:
        Success flag
    """
    client = get_mem0_client(api_key)
    if client:
        return client.add_progress_memory(stage, "COMPLETE", metrics)
    return False


def log_decision(api_key: str, decision: str, rationale: str, alternatives: List[str]) -> bool:
    """Log a technical decision to Mem0."""
    client = get_mem0_client(api_key)
    if client:
        return client.add_technical_decision(decision, rationale, alternatives)
    return False


def log_issue(api_key: str, issue: str, cause: str, solution: str, stage: int) -> bool:
    """Log an issue and resolution to Mem0."""
    client = get_mem0_client(api_key)
    if client:
        return client.add_issue_resolution(issue, cause, solution, stage)
    return False
