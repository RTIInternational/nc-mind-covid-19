import numpy as np

from src.state import LifeState, EventState
from src.calibration_collection import EventStorage


class Life(EventState):
    def __init__(self, model, enum, transition_dict, key_types):
        super().__init__(enum, transition_dict, key_types)

        self.model = model
        self.agents_to_recreate = []
        self.initiate_values(count=len(self.model.population), value=LifeState.ALIVE.value)
        self.events = EventStorage(column_names=["Unique_ID", "Time", "Location"], tracking=self.model.track_events)

    def step(self):
        """ See if anyone dies of causes not related to diseases
        """
        use_agents = np.where(self.values == LifeState.ALIVE)[0]
        probabilities = self.probabilities[use_agents]
        selected_agents = probabilities > self.model.rng.rand(len(probabilities))
        unique_ids = use_agents[selected_agents]
        for unique_id in unique_ids:
            self.life_update(unique_id)

    def life_update(self, unique_id: int):
        """ Perform a life update. Add agent to the list of agents to recreate.
        """
        self.events.record_state_change((unique_id, self.model.time, self.model.movement.location.values[unique_id]))
        self.values[unique_id] = LifeState.DEAD.value

        # If agent is scheduled to move locations, stop them, as they are now dead.
        if unique_id in self.model.movement.moved_agents:
            self.model.movement.moved_agents.remove(unique_id)
        # If agent is not at home, send them home
        if self.model.movement.location.values[unique_id] != self.model.nodes.community:
            self.model.movement.go_home(
                unique_id=unique_id, current_location=self.model.movement.location.values[unique_id]
            )

        self.agents_to_recreate.append(unique_id)
