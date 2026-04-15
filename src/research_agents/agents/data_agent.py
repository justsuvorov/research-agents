from research_agents.base_agent import BaseAgent


class DataAgent(BaseAgent):
    name = "data"

    def run(self) -> None:
        raise NotImplementedError
