from research_agents.base_agent import BaseAgent


class MLAgent(BaseAgent):
    name = "ml"

    def run(self) -> None:
        raise NotImplementedError
