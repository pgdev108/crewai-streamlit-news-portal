import os
import json
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from crewai import Agent, Crew, Process, Task  # LLM import not needed if YAML sets llm
from crewai.project import CrewBase, agent, crew, task

# Memory imports (kept, but we're not wiring them to the Crew to avoid token overflows)
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage

from crewai_tools import SerperDevTool, ScrapeWebsiteTool

load_dotenv()

# Ensure folders exist
Path("./memory").mkdir(parents=True, exist_ok=True)
Path("./output").mkdir(parents=True, exist_ok=True)

# ----- Fixed topic and sub-topics -----
TOPIC = "Cancer Health Care"
SUBTOPICS = [
    "Cancer Research & Prevention",
    "Early Detection and Diagnosis",
    "Cancer Drug Discovery and Development",
    "Cancer Treatment Methods",
    "Precision Oncology",
]

# ----- Tools -----
serper_tool = SerperDevTool(params={
                "type": "news",   # news vertical
                "num": 20,        # ask for more than we need
                "hl": "en",
                "gl": "us",
                "tbs": "qdr:m2",  # last ~2 months (fallback from 21 days)
            })  # requires SERPER_API_KEY
scrape_tool = ScrapeWebsiteTool()
# ----- Helpers for structured outputs -----
def ensure_json(s: str) -> Dict[str, Any]:
    """Try to parse JSON from an LLM response; fallback to wrapped raw."""
    try:
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end >= 0 and end > start:
            return json.loads(s[start:end+1])
        return json.loads(s)
    except Exception:
        # Sometimes agents return lists at top-level; handle that too.
        try:
            return json.loads(s)
        except Exception:
            return {"error": "non_json_output", "raw": s}


@CrewBase
class NewsPortalCrew:
    """ Cancer Health Care News Portal Crew """

    # Keep your config locations
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # -----------------------
    # Agents (use LLM from YAML)
    # -----------------------
    @agent
    def news_picker_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["news_picker_agent"],
            tools=[serper_tool],
            memory=False,
            allow_delegation=False,
        )

    @agent
    def editor_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["editor_agent"],
            memory=False,
            allow_delegation=False,
            tools=[scrape_tool, serper_tool],
        )

    @agent
    def chief_editor_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["chief_editor_agent"],
            memory=False,
            allow_delegation=False,
        )

    @agent
    def qa_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["qa_agent"],
            memory=False,
            allow_delegation=False,
        )

    # -----------------------
    # Tasks (new spec)
    # -----------------------
    # 5x news pick
    @task
    def pick_news_crp(self) -> Task: return Task(config=self.tasks_config["pick_news_crp"])

    @task
    def pick_news_edd(self) -> Task: return Task(config=self.tasks_config["pick_news_edd"])

    @task
    def pick_news_cddd(self) -> Task: return Task(config=self.tasks_config["pick_news_cddd"])

    @task
    def pick_news_ctm(self) -> Task: return Task(config=self.tasks_config["pick_news_ctm"])

    @task
    def pick_news_po(self) -> Task: return Task(config=self.tasks_config["pick_news_po"])

    # 5x edit (summaries + editorial)
    @task
    def edit_crp(self) -> Task: return Task(config=self.tasks_config["edit_crp"])

    @task
    def edit_edd(self) -> Task: return Task(config=self.tasks_config["edit_edd"])

    @task
    def edit_cddd(self) -> Task: return Task(config=self.tasks_config["edit_cddd"])

    @task
    def edit_ctm(self) -> Task: return Task(config=self.tasks_config["edit_ctm"])

    @task
    def edit_po(self) -> Task: return Task(config=self.tasks_config["edit_po"])

    # chief editor + QA
    @task
    def chief_editor(self) -> Task: return Task(config=self.tasks_config["chief_editor"])

    @task
    def qa_task(self) -> Task: return Task(config=self.tasks_config["qa_task"])

    @task
    def qa_remediate(self) -> Task: return Task(config=self.tasks_config["qa_remediate"])

    @task
    def rebuild_final(self) -> Task: return Task(config=self.tasks_config["rebuild_final"])

    @task
    def qa_task_final(self) -> Task: return Task(config=self.tasks_config["qa_task_final"])

    # -----------------------
    # Crew (sequential, no manager, no auto-RAG to avoid token errors)
    # -----------------------
    @crew
    def crew(self) -> Crew:
        """Create a Cancer Health Care News Portal Crew (sequential)."""

        # Prepare memory objects (for future use), but DO NOT wire them into Crew
        # to avoid large auto-embeds that caused 8k token errors earlier.
        _ = LongTermMemory(storage=LTMSQLiteStorage(
            db_path="./memory/long_term_memory_storage.db"
        ))
        _ = ShortTermMemory(storage=RAGStorage(
            embedder_config={
                "provider": "openai",
                "config": {
                    "model_name": "text-embedding-3-small",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                }
            },
            type="short_term",
            path="./memory/"
        ))
        _ = EntityMemory(storage=RAGStorage(
            embedder_config={
                "provider": "openai",
                "config": {
                    "model_name": "text-embedding-3-small",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                }
            },
            type="short_term",
            path="./memory/"
        ))

        return Crew(
            agents=self.agents,                   # all @agent-decorated above
            tasks=self.tasks,                     # all @task-decorated above
            process=Process.sequential,           # deterministic; no planning loops
            verbose=True,
            # No manager_agent / No memory wiring to keep runs stable
        )

    # -----------------------
    # High-level runner the UI will call
    # -----------------------
    def run_pipeline(self, topic: str | None = None) -> Dict[str, Any]:
        """
        Runs the full pipeline for the fixed TOPIC.
        The 'topic' arg is ignored (kept for backward-compatibility with your UI).
        ALWAYS writes ./output/cancer_health_care_result.json (even if 'final' missing).
        """
        inputs = {"topic": TOPIC}
        result = self.crew().kickoff(inputs=inputs)

        merged: Dict[str, Any] = {"topic": TOPIC}
        artifacts = getattr(result, "tasks_output", []) or []

        for idx, t in enumerate(artifacts):
            raw = str(getattr(t, "raw", "") or getattr(t, "output", "") or "")
            data = ensure_json(raw) if raw else {"empty": True}
            merged[f"step_{idx+1}"] = data

        # Prefer the QA-fixed "final" (last step), else chief editor's "final",
        # else assemble a minimal fallback so Streamlit can render.
        merged["final"] = (
            merged.get("step_12", {}).get("final")
            or merged.get("step_11", {}).get("final")
            or self._minimal_final_fallback(merged)
        )

        # Save merged output (ALWAYS)
        out_path = Path("./output") / "cancer_health_care_result.json"
        try:
            out_path.write_text(json.dumps(merged, indent=2))
        except Exception as e:
            merged["file_write_error"] = str(e)

        return merged

    # -----------------------
    # Minimal fallback so UI can still render if chief/QA didn't emit "final"
    # -----------------------
    def _minimal_final_fallback(self, merged: Dict[str, Any]) -> Dict[str, Any]:
        # Build per_subtopic using pick+edit steps if available
        per_subtopic: Dict[str, Dict[str, Any]] = {}

        # Map subtopic → (pick_step_index, edit_step_index) best guess by position:
        pick_steps = ["step_1", "step_2", "step_3", "step_4", "step_5"]
        edit_steps = ["step_6", "step_7", "step_8", "step_9", "step_10"]

        for i, sub in enumerate(SUBTOPICS):
            pick = merged.get(pick_steps[i], {}) if i < len(pick_steps) else {}
            edit = merged.get(edit_steps[i], {}) if i < len(edit_steps) else {}

            # articles from pick
            arts = []
            if isinstance(pick.get("articles"), list):
                arts = [dict(a) for a in pick["articles"]]

            # attach summaries from editor (by position)
            summaries = edit.get("summaries", []) if isinstance(edit.get("summaries"), list) else []
            for j, a in enumerate(arts):
                a["summary"] = summaries[j] if j < len(summaries) else None

            per_subtopic[sub] = {
                "articles": arts[:5],
                "editorial": edit.get("editorial"),
                "best_article": arts[0] if arts else None,
            }

        # home.best_articles = first best from each subtopic (if any)
        best_articles = []
        for sub in SUBTOPICS:
            ba = per_subtopic.get(sub, {}).get("best_article")
            if ba:
                x = dict(ba)
                x["subtopic"] = sub
                best_articles.append(x)

        return {
            "topic": TOPIC,
            "subtopics": SUBTOPICS,
            "per_subtopic": per_subtopic,
            "home": {
                "best_articles": best_articles[:5],
                "main_editorial": None,
            },
        }
