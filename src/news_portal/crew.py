import os
import json
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai_tools import SerperDevTool

load_dotenv()

# Ensure folders exist
Path("./memory").mkdir(parents=True, exist_ok=True)
Path("./output").mkdir(parents=True, exist_ok=True)

# ----- Tools -----
serper_tool = SerperDevTool()  # requires SERPER_API_KEY

# ----- Helpers for structured outputs -----
def ensure_json(s: str) -> Dict[str, Any]:
    """Try to parse JSON from an LLM response; fallback to empty structure."""
    try:
        # extract first JSON block if extra text present
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end >= 0 and end > start:
            return json.loads(s[start:end+1])
        return json.loads(s)
    except Exception:
        return {"error": "non_json_output", "raw": s}

# ----- CrewBase container -----
@CrewBase
class NewsPortalCrew:
    """ News Portal Crew """

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # Agents
    @agent
    def sub_topics_picker_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["sub_topics_picker_agent"],            
            memory=True,
        )

    @agent
    def news_article_picker_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["news_article_picker_agent"],            
            tools=[serper_tool],
            memory=True,
        )

    @agent
    def news_article_researcher_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["news_article_researcher_agent"],            
            memory=True,
        )    

    # Tasks
    @task
    def pick_sub_topics_task(self) -> Task:
        return Task(
            config=self.tasks_config["pick_sub_topics_task"]
        )

    @task
    def pick_news_articles_task(self) -> Task:
        return Task(
            config=self.tasks_config["pick_news_articles_task"]
        )

    @task
    def research_and_select_best_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_and_select_best_task"]
        )

    @crew
    def crew(self) -> Crew:
        """Create a News Portal Crew."""

        manager = Agent(config=self.agents_config['manager'], allow_delegation=True)
        
        long_term_memory = LongTermMemory(storage=LTMSQLiteStorage(db_path="./memory/long_term_memory_storage.db"))
        short_term_memory = ShortTermMemory(storage=RAGStorage(
            embedder_config={
                "provider": "openai", 
                "config": {
                    "model_name": 'text-embedding-3-small',
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    }}, 
            type="short_term", 
            path="./memory/"))
        entity_memory = EntityMemory(
            storage=RAGStorage(
                embedder_config={
                    "provider": "openai", 
                    "config": {
                        "model_name": 'text-embedding-3-small',
                        "api_key": os.getenv("OPENAI_API_KEY"),
                        }}, 
                    type="short_term", 
                    path="./memory/"))

        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            process=Process.hierarchical, 
            verbose=True, 
            manager_agent=manager,
            memory=True,
            long_term_memory=long_term_memory,
            short_term_memory=short_term_memory,
            entity_memory=entity_memory            
            )

    # High-level runner the UI will call
    def run_pipeline(self, topic: str) -> Dict[str, Any]:
        inputs = {"topic": topic}
        result = self.crew().kickoff(inputs=inputs)

        # Expect 3 artifacts (one per task) in result.tasks_output
        # Normalize to a single merged JSON payload for the UI.
        merged: Dict[str, Any] = {"topic": topic}
        artifacts = getattr(result, "tasks_output", []) or []
        for idx, t in enumerate(artifacts):
            data = ensure_json(str(getattr(t, "raw", "") or getattr(t, "output", "")))
            merged[f"step_{idx+1}"] = data

        # Optional: create a compact combined structure the UI expects
        # subtopics: list[str]
        # per_subtopic: { subtopic: { candidates:[{title,url,source,date,is_editorial}], best_article:{...}, editorial_candidates:[...]} }
        # home: { best_articles:[...], best_editorial:{...} }

        # The researcher step (step_3) should output the "final" structure; if missing, attempt building from step_1 & step_2
        if "final" in merged.get("step_3", {}):
            merged["final"] = merged["step_3"]["final"]
        else:
            # Fallback: pass through what’s available
            merged["final"] = {
                "subtopics": merged.get("step_1", {}).get("subtopics", []),
                "per_subtopic": merged.get("step_2", {}).get("per_subtopic", {}),
                "home": merged.get("step_3", {}).get("home", {})
            }

        # Save a JSON copy to output
        out_path = Path("./output") / f"{topic.replace(' ','_').lower()}_result.json"
        out_path.write_text(json.dumps(merged, indent=2))
        return merged
