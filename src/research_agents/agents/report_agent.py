from research_agents.base_agent import BaseAgent


class ReportAgent(BaseAgent):
    name = "report"

    def run(self) -> None:
        raise NotImplementedError
