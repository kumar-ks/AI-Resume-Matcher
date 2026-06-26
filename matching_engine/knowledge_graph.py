"""
Skill Knowledge Graph
======================

A NetworkX-based skill ontology that models relationships between
technical skills, tools, domains, and concepts.

PURPOSE:
    - Expand JD skills to find related/equivalent skills in resumes
    - Determine if two skills are related (and how strongly)
    - Replace the hardcoded abbreviations dict in scoring.py
    - Enable hierarchical skill matching (MLOps → Kubeflow → ML Pipelines)

RELATIONSHIPS:
    - IS_ALIAS_OF: Same skill, different name (K8s ↔ Kubernetes)
    - IS_TOOL_FOR: Tool implements a concept (Kubeflow IS_TOOL_FOR MLOps)
    - IS_PART_OF: Skill belongs to a domain (Docker IS_PART_OF Containerization)
    - IS_PREREQUISITE_FOR: Skill enables another (Docker IS_PREREQUISITE_FOR Kubernetes)
    - IS_RELATED_TO: General relationship (DevOps IS_RELATED_TO MLOps)

USAGE:
    from matching_engine.knowledge_graph import SkillGraph

    kg = SkillGraph()
    kg.is_related("MLOps", "Kubeflow")           → True
    kg.skill_distance("MLOps", "DevOps")         → 0.7 (related)
    kg.expand_skill("MLOps")                     → ["Kubeflow", "MLflow", ...]
    kg.fuzzy_match("k8s", {"kubernetes", "docker"}) → True

CALLED BY:
    - matching_engine.scoring (replaces _fuzzy_skill_match)
    - matching_engine.semantic_matching (optional skill expansion)
"""

import logging
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)


# Edge relationship types with match strength (0.0 to 1.0)
RELATIONSHIP_WEIGHTS = {
    "IS_ALIAS_OF": 1.0,        # Exact equivalence (k8s = kubernetes)
    "IS_TOOL_FOR": 0.85,       # Tool implements concept (Kubeflow → MLOps)
    "IS_PART_OF": 0.80,        # Belongs to domain (React → Frontend)
    "IS_PREREQUISITE_FOR": 0.7, # Enables another skill (Docker → Kubernetes)
    "IS_RELATED_TO": 0.6,      # General relation (DevOps → MLOps)
}


class SkillGraph:
    """
    NetworkX-based skill knowledge graph.

    Models ~200 tech skills with their relationships, enabling
    intelligent skill matching beyond exact string comparison.
    """

    def __init__(self):
        """Initialize the skill graph with the built-in ontology."""
        self.graph = nx.Graph()
        self._build_ontology()
        logger.info(
            f"SkillGraph initialized: {self.graph.number_of_nodes()} skills, "
            f"{self.graph.number_of_edges()} relationships"
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def is_related(self, skill_a: str, skill_b: str) -> bool:
        """
        Check if two skills are related in the knowledge graph.

        Args:
            skill_a: First skill name
            skill_b: Second skill name

        Returns:
            True if skills are connected (directly or within 2 hops)
        """
        a = skill_a.lower().strip()
        b = skill_b.lower().strip()

        if a == b:
            return True

        # Check if both exist in graph
        if a not in self.graph or b not in self.graph:
            return False

        # Check direct connection
        if self.graph.has_edge(a, b):
            return True

        # Check 2-hop connection (A → X → B)
        try:
            path_length = nx.shortest_path_length(self.graph, a, b)
            return path_length <= 2
        except nx.NetworkXNoPath:
            return False

    def skill_distance(self, skill_a: str, skill_b: str) -> float:
        """
        Compute similarity between two skills (0.0 = unrelated, 1.0 = same).

        Uses graph distance and edge weights to determine relatedness.

        Args:
            skill_a: First skill name
            skill_b: Second skill name

        Returns:
            Float 0.0-1.0 representing skill similarity
        """
        a = skill_a.lower().strip()
        b = skill_b.lower().strip()

        if a == b:
            return 1.0

        if a not in self.graph or b not in self.graph:
            return 0.0

        # Direct edge — use the relationship weight
        if self.graph.has_edge(a, b):
            rel_type = self.graph[a][b].get("relationship", "IS_RELATED_TO")
            return RELATIONSHIP_WEIGHTS.get(rel_type, 0.5)

        # Multi-hop — decay by distance
        try:
            path_length = nx.shortest_path_length(self.graph, a, b)
            if path_length <= 3:
                return max(0.3, 1.0 - (path_length * 0.25))
            return 0.0
        except nx.NetworkXNoPath:
            return 0.0

    def expand_skill(self, skill: str, max_hops: int = 2) -> list[str]:
        """
        Expand a skill to find all related skills within N hops.

        Useful for expanding JD requirements to match broader resume terminology.

        Args:
            skill: The skill to expand
            max_hops: Maximum graph distance to traverse (default: 2)

        Returns:
            List of related skill names (sorted by relevance)
        """
        s = skill.lower().strip()
        if s not in self.graph:
            return []

        # BFS to find all nodes within max_hops
        related = []
        for node, distance in nx.single_source_shortest_path_length(self.graph, s, cutoff=max_hops).items():
            if node != s:
                weight = 1.0 - (distance * 0.3)  # Decay by distance
                related.append((node, weight))

        # Sort by relevance (weight) descending
        related.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in related]

    def fuzzy_match(self, skill: str, candidate_skills: set[str]) -> bool:
        """
        Check if a skill matches any candidate skill using the knowledge graph.

        This REPLACES the hardcoded abbreviations dict in scoring.py.

        Strategy:
            1. Exact match in candidate skills
            2. Check if skill is an alias of any candidate skill
            3. Check if skill is directly related (1 hop) to any candidate skill
            4. Substring containment (fallback)

        Args:
            skill: Lowercased skill name to search for
            candidate_skills: Set of lowercased candidate skill names

        Returns:
            True if the skill matches any candidate skill
        """
        # Strategy 1: Exact match
        if skill in candidate_skills:
            return True

        # Strategy 2: Check aliases via graph
        if skill in self.graph:
            for neighbor in self.graph.neighbors(skill):
                rel = self.graph[skill][neighbor].get("relationship", "")
                if rel == "IS_ALIAS_OF" and neighbor in candidate_skills:
                    return True

        # Strategy 3: Check direct relationships (1 hop)
        if skill in self.graph:
            for neighbor in self.graph.neighbors(skill):
                if neighbor in candidate_skills:
                    return True

        # Strategy 4: Substring containment (fallback for non-graph skills)
        for cs in candidate_skills:
            if skill in cs or cs in skill:
                return True

        return False

    def get_skill_category(self, skill: str) -> Optional[str]:
        """Get the top-level domain/category for a skill."""
        s = skill.lower().strip()
        if s not in self.graph:
            return None
        return self.graph.nodes[s].get("category")

    # =========================================================================
    # ONTOLOGY BUILDER — defines the skill knowledge graph
    # =========================================================================

    def _build_ontology(self) -> None:
        """
        Build the skill ontology with ~200 skills and their relationships.

        Covers: Cloud, DevOps, MLOps, Data Science, Backend, Frontend,
        Databases, Programming Languages, Frameworks, and Tools.
        """
        # ── ALIASES (same skill, different names) ─────────────────────────────
        aliases = [
            ("kubernetes", "k8s"),
            ("javascript", "js"),
            ("typescript", "ts"),
            ("python", "py"),
            ("react", "reactjs"),
            ("react", "react.js"),
            ("node", "nodejs"),
            ("node", "node.js"),
            ("aws", "amazon web services"),
            ("gcp", "google cloud platform"),
            ("gcp", "google cloud"),
            ("azure", "microsoft azure"),
            ("ci/cd", "cicd"),
            ("ci/cd", "continuous integration"),
            ("ci/cd", "continuous delivery"),
            ("ml", "machine learning"),
            ("dl", "deep learning"),
            ("nlp", "natural language processing"),
            ("cv", "computer vision"),
            ("sql", "structured query language"),
            ("nosql", "no-sql"),
            ("api", "rest api"),
            ("api", "restful"),
            ("microservices", "micro-services"),
            ("docker", "containerization"),
            ("docker", "containers"),
            ("terraform", "iac"),
            ("terraform", "infrastructure as code"),
            ("git", "version control"),
            ("agile", "scrum"),
            ("postgresql", "postgres"),
            ("mongodb", "mongo"),
            ("elasticsearch", "elastic"),
            ("rabbitmq", "message queue"),
            ("kafka", "event streaming"),
            ("grafana", "monitoring"),
            ("prometheus", "metrics"),
            ("jenkins", "ci server"),
            ("github actions", "gh actions"),
            ("sre", "site reliability engineering"),
            ("devsecops", "secure devops"),
        ]

        # ── IS_TOOL_FOR relationships ────────────────────────────────────────
        tools_for = [
            # MLOps tools
            ("kubeflow", "mlops"),
            ("mlflow", "mlops"),
            ("airflow", "mlops"),
            ("dvc", "mlops"),
            ("seldon", "mlops"),
            ("bentoml", "mlops"),
            ("sagemaker", "mlops"),
            ("vertex ai", "mlops"),
            ("weights & biases", "mlops"),
            # DevOps tools
            ("jenkins", "devops"),
            ("jenkins", "ci/cd"),
            ("github actions", "devops"),
            ("github actions", "ci/cd"),
            ("gitlab ci", "devops"),
            ("gitlab ci", "ci/cd"),
            ("argocd", "devops"),
            ("argocd", "ci/cd"),
            ("ansible", "devops"),
            ("puppet", "devops"),
            ("chef", "devops"),
            ("terraform", "devops"),
            ("helm", "devops"),
            # Data Science tools
            ("pandas", "data science"),
            ("numpy", "data science"),
            ("scikit-learn", "data science"),
            ("matplotlib", "data science"),
            ("seaborn", "data science"),
            ("jupyter", "data science"),
            ("spark", "data science"),
            ("pyspark", "data science"),
            # Deep Learning frameworks
            ("tensorflow", "deep learning"),
            ("pytorch", "deep learning"),
            ("keras", "deep learning"),
            # Cloud tools
            ("ec2", "aws"),
            ("s3", "aws"),
            ("lambda", "aws"),
            ("eks", "aws"),
            ("ecs", "aws"),
            ("rds", "aws"),
            ("dynamodb", "aws"),
            ("cloudformation", "aws"),
            ("gke", "gcp"),
            ("bigquery", "gcp"),
            ("cloud run", "gcp"),
            ("aks", "azure"),
            ("azure devops", "azure"),
            # Monitoring tools
            ("prometheus", "monitoring"),
            ("grafana", "monitoring"),
            ("datadog", "monitoring"),
            ("new relic", "monitoring"),
            ("elk stack", "monitoring"),
            ("splunk", "monitoring"),
            # Database tools
            ("postgresql", "sql"),
            ("mysql", "sql"),
            ("oracle", "sql"),
            ("sqlite", "sql"),
            ("mongodb", "nosql"),
            ("cassandra", "nosql"),
            ("redis", "nosql"),
            ("dynamodb", "nosql"),
            ("elasticsearch", "nosql"),
        ]

        # ── IS_PART_OF relationships (skill belongs to domain) ────────────────
        part_of = [
            ("docker", "containerization"),
            ("kubernetes", "container orchestration"),
            ("container orchestration", "cloud"),
            ("containerization", "cloud"),
            ("aws", "cloud"),
            ("gcp", "cloud"),
            ("azure", "cloud"),
            ("ci/cd", "devops"),
            ("monitoring", "devops"),
            ("sre", "devops"),
            ("devsecops", "devops"),
            ("mlops", "devops"),
            ("mlops", "data science"),
            ("data science", "ai"),
            ("deep learning", "ai"),
            ("nlp", "ai"),
            ("cv", "ai"),
            ("machine learning", "ai"),
            ("react", "frontend"),
            ("angular", "frontend"),
            ("vue", "frontend"),
            ("html", "frontend"),
            ("css", "frontend"),
            ("spring", "backend"),
            ("spring boot", "backend"),
            ("django", "backend"),
            ("flask", "backend"),
            ("fastapi", "backend"),
            ("express", "backend"),
            ("node", "backend"),
            ("java", "backend"),
            ("python", "backend"),
            ("python", "data science"),
            ("sql", "databases"),
            ("nosql", "databases"),
            ("redis", "caching"),
            ("kafka", "messaging"),
            ("rabbitmq", "messaging"),
        ]

        # ── IS_PREREQUISITE_FOR relationships ─────────────────────────────────
        prerequisites = [
            ("docker", "kubernetes"),
            ("python", "data science"),
            ("python", "machine learning"),
            ("linux", "devops"),
            ("git", "devops"),
            ("sql", "data engineering"),
            ("statistics", "data science"),
            ("machine learning", "deep learning"),
            ("java", "spring boot"),
            ("javascript", "react"),
            ("javascript", "node"),
            ("html", "frontend"),
        ]

        # ── IS_RELATED_TO (general associations) ─────────────────────────────
        related = [
            ("devops", "mlops"),
            ("devops", "sre"),
            ("devops", "devsecops"),
            ("backend", "frontend"),
            ("cloud", "devops"),
            ("microservices", "docker"),
            ("microservices", "kubernetes"),
            ("microservices", "api"),
            ("agile", "devops"),
            ("data engineering", "data science"),
            ("data engineering", "mlops"),
            ("etl", "data engineering"),
            ("spark", "data engineering"),
        ]

        # ── Build the graph ───────────────────────────────────────────────────
        for a, b in aliases:
            self.graph.add_edge(a, b, relationship="IS_ALIAS_OF")

        for tool, concept in tools_for:
            self.graph.add_edge(tool, concept, relationship="IS_TOOL_FOR")

        for part, whole in part_of:
            self.graph.add_edge(part, whole, relationship="IS_PART_OF")

        for prereq, skill in prerequisites:
            self.graph.add_edge(prereq, skill, relationship="IS_PREREQUISITE_FOR")

        for a, b in related:
            self.graph.add_edge(a, b, relationship="IS_RELATED_TO")

        # ── Add category metadata to nodes ────────────────────────────────────
        categories = {
            "cloud": ["aws", "gcp", "azure", "ec2", "s3", "lambda", "eks", "gke", "aks"],
            "devops": ["docker", "kubernetes", "jenkins", "terraform", "ansible", "helm", "ci/cd", "argocd"],
            "mlops": ["kubeflow", "mlflow", "airflow", "sagemaker", "dvc", "seldon"],
            "data_science": ["pandas", "numpy", "scikit-learn", "jupyter", "spark", "pyspark"],
            "ai": ["tensorflow", "pytorch", "keras", "machine learning", "deep learning", "nlp", "cv"],
            "backend": ["java", "python", "spring", "django", "flask", "fastapi", "node"],
            "frontend": ["react", "angular", "vue", "html", "css", "javascript", "typescript"],
            "databases": ["sql", "nosql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch"],
        }

        for category, skills in categories.items():
            for skill in skills:
                if skill in self.graph:
                    self.graph.nodes[skill]["category"] = category


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON INSTANCE
# ─────────────────────────────────────────────────────────────────────────────
# The graph is expensive to build but never changes at runtime.
# Use a module-level singleton so it's built once and shared.

_skill_graph_instance: Optional[SkillGraph] = None


def get_skill_graph() -> SkillGraph:
    """
    Get the singleton SkillGraph instance.

    Builds the graph on first call, returns cached instance after.
    """
    global _skill_graph_instance
    if _skill_graph_instance is None:
        _skill_graph_instance = SkillGraph()
    return _skill_graph_instance
