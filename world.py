import json
import logging
from typing import List, Dict, Optional
from endpoints.database import redis
import numpy as np
from utils import add_log
from config import CONNECTIVITY_GRAPH
import asyncio

logger = logging.getLogger("simulation_app")

class World:
    def __init__(self, world_id: int, grid_size: int, num_agents: int, tasks: List[Dict], resources: Dict):
        self.world_id = world_id
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.agents = self.initialize_agents(num_agents)
        self.tasks = tasks
        self.resources = resources
        self.communications = []  # Store communication activities
        self.environment = {}  # Store environment metrics

        # Stream-based communication setup
        self.stream_key = f"world:{self.world_id}:stream"
        self.group_name = f"group_{self.world_id}"
        self.consumer_name = f"consumer_{self.world_id}"

        # Ensure consumer group exists
        asyncio.create_task(self.initialize_consumer_group())

    def initialize_agents(self, num_agents: int) -> List[Dict]:
        agents = [
            {
                "id": i,
                "x": np.random.randint(0, self.grid_size),
                "y": np.random.randint(0, self.grid_size),
                "memory": "",
                "cooperation_score": np.random.rand(),
                "conflict_score": np.random.rand(),
            }
            for i in range(num_agents)
        ]
        logger.info(f"Initialized {num_agents} agents for World {self.world_id}.")
        return agents

    async def initialize_consumer_group(self):
        try:
            # Attempt to create a consumer group for the world stream
            await redis.xgroup_create(name=self.stream_key, groupname=self.group_name, id="0", mkstream=True)
            add_log(f"Consumer group '{self.group_name}' created for World {self.world_id}.")
        except Exception as e:
            if "BUSYGROUP" in str(e):  # Ignore if the group already exists
                add_log(f"Consumer group '{self.group_name}' already exists for World {self.world_id}.")
            else:
                logger.error(f"Error creating consumer group for World {self.world_id}: {e}")

    def send_message(self, target_world_id: Optional[int], message_type: str, payload: Dict):
        """
        Send a message to a specific world or broadcast if target_world_id is None.
        """
        message = {
            "source_world_id": self.world_id,
            "message_type": message_type,
            "payload": json.dumps(payload),
        }
        try:
            if target_world_id:
                # Send a directed message via Redis Stream
                stream_key = f"world:{target_world_id}:stream"
                asyncio.create_task(redis.xadd(stream_key, message))
                logger.info(f"Message sent from World {self.world_id} to World {target_world_id}.")
            else:
                # Broadcast to all neighbors
                neighbors = CONNECTIVITY_GRAPH.neighbors(self.world_id)
                for neighbor in neighbors:
                    stream_key = f"world:{neighbor}:stream"
                    asyncio.create_task(redis.xadd(stream_key, message))
                logger.info(f"Broadcast message from World {self.world_id}.")
        except Exception as e:
            logger.error(f"Error sending message from World {self.world_id}: {e}")

    async def receive_messages(self):
        """
        Continuously read and process messages from the Redis Stream.
        """
        try:
            while True:
                messages = await redis.xreadgroup(
                    groupname=self.group_name,
                    consumername=self.consumer_name,
                    streams={self.stream_key: ">"},
                    count=10,
                    block=30000,  # Wait up to 5 seconds for new messages
                )
                for _, msgs in messages:
                    for msg_id, msg_data in msgs:
                        await self.process_stream_message(msg_id, msg_data)
        except Exception as e:
            logger.error(f"Error receiving messages for World {self.world_id}: {e}")

    async def process_stream_message(self, msg_id: str, msg_data: Dict[bytes, bytes]):
        """
        Process a single message from the Redis Stream.
        """
        try:
            message_type = msg_data.get(b"message_type", b"").decode("utf-8")
            payload = json.loads(msg_data.get(b"payload", b"{}").decode("utf-8"))
            source_world_id = msg_data.get(b"source_world_id", b"").decode("utf-8")

            logger.info(f"World {self.world_id} received message '{message_type}' from World {source_world_id}.")
            add_log(f"World {self.world_id} received message '{message_type}' from World {source_world_id}'.")

            # Process the message based on type
            if message_type == "state_summary_request":
                self.handle_state_summary_request(int(source_world_id))
            elif message_type == "cooperation_request":
                self.handle_cooperation_request(int(source_world_id), payload)
            elif message_type == "alert":
                self.handle_alert(int(source_world_id), payload)
            elif message_type == "response":
                self.handle_response(int(source_world_id), payload)
            else:
                logger.warning(f"Unknown message type '{message_type}' received by World {self.world_id} from World {source_world_id}.")

            # Acknowledge the message
            await redis.xack(self.stream_key, self.group_name, msg_id)
        except Exception as e:
            logger.error(f"Error processing message for World {self.world_id}: {e}")

    def broadcast_message(self, message_type: str, payload: Dict):
        """
        Broadcast a message to all connected worlds.
        """
        self.send_message(target_world_id=None, message_type=message_type, payload=payload)

    def handle_state_summary_request(self, requester_world_id: int):
        """
        Respond to a state summary request from another world.
        """
        state_summary = self.summarize_state()
        self.send_message(
            target_world_id=requester_world_id,
            message_type="state_summary",
            payload={"summary": state_summary},
        )
        logger.info(f"World {self.world_id} sent state summary to World {requester_world_id}.")
        add_log(f"World {self.world_id} sent state summary to World {requester_world_id}.")

    def handle_cooperation_request(self, requester_world_id: int, payload: Dict):
        """
        Handle a cooperation request from another world.
        """
        resource_share = payload.get("resource_share", 0)
        if self.resources["total"] >= resource_share:
            self.resources["total"] -= resource_share
            logger.info(f"World {self.world_id} shared {resource_share} resources with World {requester_world_id}.")
            add_log(f"World {self.world_id} shared {resource_share} resources with World {requester_world_id}.")
        else:
            logger.warning(f"World {self.world_id} cannot share {resource_share} resources with World {requester_world_id}: Insufficient resources.")
            add_log(f"World {self.world_id} cannot share {resource_share} resources with World {requester_world_id}.")

    def handle_alert(self, source_world_id: int, payload: Dict):
        """
        Handle an alert from another world.
        """
        level = payload.get("level", "info")
        message = payload.get("message", "")
        logger.warning(f"World {self.world_id} received alert from World {source_world_id}: [{level}] {message}")
        add_log(f"World {self.world_id} received alert from World {source_world_id}: [{level}] {message}")

    def handle_response(self, source_world_id: int, payload: Dict):
        """
        Handle a response from another world.
        """
        ack = payload.get("ack", False)
        if ack:
            logger.info(f"World {self.world_id} received acknowledgment from World {source_world_id}.")
            add_log(f"World {self.world_id} received acknowledgment from World {source_world_id}.")
        else:
            logger.warning(f"World {self.world_id} received negative acknowledgment from World {source_world_id}.")
            add_log(f"World {self.world_id} received negative acknowledgment from World {source_world_id}.")

    def summarize_state(self) -> List[float]:
        """
        Generate a normalized state summary vector for this world.
        """
        try:
            # Task Metrics
            task_progress = [task.get("progress", 0) for task in self.tasks]
            avg_task_progress = np.mean(task_progress) if task_progress else 0
            fraction_completed_tasks = np.sum([1 for p in task_progress if p >= 1.0]) / len(task_progress) if task_progress else 0

            # Agent Metrics
            agent_x = [agent["x"] for agent in self.agents]
            agent_y = [agent["y"] for agent in self.agents]
            mean_x = np.mean(agent_x) if agent_x else 0
            mean_y = np.mean(agent_y) if agent_y else 0

            # Resource Metrics
            total_resources = self.resources.get("total", 0)
            consumed_resources = self.resources.get("consumed", 0)
            percentage_resources_consumed = consumed_resources / total_resources if total_resources else 0

            # Emergent Metrics
            avg_cooperation = np.mean([agent["cooperation_score"] for agent in self.agents]) if self.agents else 0
            avg_conflict = np.mean([agent["conflict_score"] for agent in self.agents]) if self.agents else 0

            return [
                avg_task_progress,
                fraction_completed_tasks,
                mean_x / self.grid_size,
                mean_y / self.grid_size,
                percentage_resources_consumed,
                avg_cooperation,
                avg_conflict,
            ]
        except Exception as e:
            logger.error(f"Error summarizing state for World {self.world_id}: {e}")
            return [0.0] * 7

    def process_feedback(self, feedback):
        """
        Process and apply feedback from mTNN.
        """
        try:
            # Validate feedback
            feedback = self.validate_feedback(feedback)

            # Normalize feedback
            feedback = self.normalize_feedback(feedback)

            # Resolve conflicts
            task_feedback, resource_feedback = self.resolve_conflicts(
                feedback.get("task_priorities", {}),
                feedback.get("resource_allocation", {}),
            )

            # Apply task adjustments
            for task_id, priority in task_feedback.items():
                if task_id in self.tasks:
                    old_priority = self.tasks[task_id]["priority"]
                    self.tasks[task_id]["priority"] = priority
                    logger.info(f"Task {task_id} priority updated from {old_priority} to {priority}.")
                else:
                    logger.warning(f"Task ID {task_id} not found in world {self.world_id}.")

            # Apply agent behavior modifications
            for behavior, value in feedback.get("agent_behaviors", {}).items():
                for agent in self.agents:
                    old_value = agent.get(behavior, 0.5)  # Default behavior value
                    agent.update_behavior(behavior, value)
                    logger.info(f"Agent behavior '{behavior}' updated from {old_value} to {value} for world {self.world_id}.")

            # Apply resource adjustments
            for resource, amount in resource_feedback.items():
                if resource in self.resources["distribution"]:
                    old_amount = self.resources["distribution"][resource]
                    self.resources["distribution"][resource] = amount
                    logger.info(f"Resource '{resource}' updated from {old_amount} to {amount}.")
                else:
                    logger.warning(f"Resource '{resource}' not found in world {self.world_id}.")

            # Apply environmental changes
            for env_param, value in feedback.get("environment_changes", {}).items():
                old_value = self.environment.get(env_param, 0.0)
                self.environment[env_param] = value
                logger.info(f"Environment parameter '{env_param}' updated from {old_value} to {value}.")

            logger.info(f"Feedback processed for world {self.world_id}: {feedback}")

        except Exception as e:
            logger.error(f"Error processing feedback for world {self.world_id}: {str(e)}")

    # def send_message(self, target_world_id: Optional[int], message_type: str, payload: Dict):
    #     message = {
    #         "source_world_id": self.world_id,
    #         "target_world_id": target_world_id,
    #         "message_type": message_type,
    #         "payload": payload,
    #     }
    #     message_json = json.dumps(message)

    #     try:
    #         if target_world_id is not None:
    #             # Directed communication: Check if a direct connection exists
    #             if CONNECTIVITY_GRAPH.has_edge(self.world_id, target_world_id):
    #                 asyncio.create_task(redis.rpush(f"world:{target_world_id}:inbox", message_json))
    #                 logger.info(f"Message sent from World {self.world_id} to World {target_world_id}")
    #             else:
    #                 logger.warning(f"No direct connection from World {self.world_id} to World {target_world_id}")
    #         else:
    #             # Broadcast communication: Send to all directly connected worlds
    #             neighbors = CONNECTIVITY_GRAPH.neighbors(self.world_id)
    #             for neighbor in neighbors:
    #                 asyncio.create_task(redis.rpush(f"world:{neighbor}:inbox", message_json))
    #             logger.info(f"Broadcast message from World {self.world_id}")
    #     except Exception as e:
    #         logger.error(f"Error sending message from World {self.world_id}: {e}")

    # def broadcast_message(self, message_type: str, payload: Dict):
    #     """
    #     Broadcast a message to all connected worlds.
    #     """
    #     self.send_message(target_world_id=None, message_type=message_type, payload=payload)

    # async def receive_messages(self):
    #     """
    #     Receive and store incoming messages from the inbox.
    #     """
    #     try:
    #         while True:
    #             message_json = await redis.lpop(f"world:{self.world_id}:inbox")
    #             if message_json:
    #                 message = json.loads(message_json)
    #                 self.inbox_messages.append(message)
    #                 logger.info(f"World {self.world_id} received message: {message}")
    #                 add_log(f"World {self.world_id} received message: {message}")
    #             else:
    #                 break  # No more messages
    #     except Exception as e:
    #         logger.error(f"Error receiving messages for World {self.world_id}: {e}")
    #         add_log(f"Error receiving messages for World {self.world_id}: {e}")

    # def process_messages(self):
    #     """
    #     Process all received messages in the inbox.
    #     """
    #     for message in self.inbox_messages:
    #         message_type = message.get("message_type")
    #         payload = message.get("payload", {})
    #         source_world_id = message.get("source_world_id")

    #         if message_type == "state_summary_request":
    #             self.handle_state_summary_request(source_world_id)
    #         elif message_type == "cooperation_request":
    #             self.handle_cooperation_request(source_world_id, payload)
    #         elif message_type == "alert":
    #             self.handle_alert(source_world_id, payload)
    #         elif message_type == "response":
    #             self.handle_response(source_world_id, payload)
    #         else:
    #             logger.warning(f"Unknown message type '{message_type}' received by World {self.world_id} from World {source_world_id}.")

    #     # Clear the inbox after processing
    #     self.inbox_messages.clear()

    def handle_state_summary_request(self, requester_world_id: int):
        """
        Respond to a state summary request from another world.
        """
        state_summary = self.summarize_state()
        self.send_message(
            target_world_id=requester_world_id,
            message_type="state_summary",
            payload={"summary": state_summary}
        )
        logger.info(f"World {self.world_id} sent state summary to World {requester_world_id}.")
        add_log(f"World {self.world_id} sent state summary to World {requester_world_id}.")

    def handle_cooperation_request(self, requester_world_id: int, payload: Dict):
        """
        Handle a cooperation request from another world.
        """
        # Example payload: {"resource_share": 100}
        resource_share = payload.get("resource_share", 0)
        if self.resources["total"] >= resource_share:
            self.resources["total"] -= resource_share
            # Assume that the recipient world will handle receiving the resource
            logger.info(f"World {self.world_id} shared {resource_share} resources with World {requester_world_id}.")
            add_log(f"World {self.world_id} shared {resource_share} resources with World {requester_world_id}.")
        else:
            logger.warning(f"World {self.world_id} cannot share {resource_share} resources with World {requester_world_id}: Insufficient resources.")
            add_log(f"World {self.world_id} cannot share {resource_share} resources with World {requester_world_id}: Insufficient resources.")

    def handle_alert(self, source_world_id: int, payload: Dict):
        """
        Handle an alert from another world.
        """
        # Example payload: {"level": "critical", "message": "Resource scarcity detected"}
        level = payload.get("level", "info")
        message = payload.get("message", "")
        logger.warning(f"World {self.world_id} received alert from World {source_world_id}: [{level}] {message}")
        add_log(f"World {self.world_id} received alert from World {source_world_id}: [{level}] {message}")

    def handle_response(self, source_world_id: int, payload: Dict):
        """
        Handle a response from another world.
        """
        # Example payload: {"ack": True}
        ack = payload.get("ack", False)
        if ack:
            logger.info(f"World {self.world_id} received acknowledgment from World {source_world_id}.")
            add_log(f"World {self.world_id} received acknowledgment from World {source_world_id}.")
        else:
            logger.warning(f"World {self.world_id} received negative acknowledgment from World {source_world_id}.")
            add_log(f"World {self.world_id} received negative acknowledgment from World {source_world_id}.")

    def validate_feedback(self, feedback):
        """
        Validate and handle incomplete or invalid feedback.
        """
        valid_feedback = {"task_priorities": {}, "agent_behaviors": {}, "resource_allocation": {}, "environment_changes": {}}

        # Validate task priorities
        for task_id, priority in feedback.get("task_priorities", {}).items():
            if task_id in self.tasks:
                valid_feedback["task_priorities"][task_id] = priority
            else:
                logger.warning(f"Invalid task ID {task_id} in feedback for world {self.world_id}.")

        # Validate agent behaviors
        for behavior, value in feedback.get("agent_behaviors", {}).items():
            if 0 <= value <= 1:  # Ensure values are in range
                valid_feedback["agent_behaviors"][behavior] = value
            else:
                logger.warning(f"Invalid behavior value {value} for {behavior} in world {self.world_id}.")

        # Validate resource allocation
        for resource, amount in feedback.get("resource_allocation", {}).items():
            if resource in self.resources["distribution"]:
                valid_feedback["resource_allocation"][resource] = amount
            else:
                logger.warning(f"Invalid resource {resource} in feedback for world {self.world_id}.")

        # Validate environmental changes
        valid_feedback["environment_changes"] = feedback.get("environment_changes", {})

        return valid_feedback

    def normalize_feedback(self, feedback):
        """
        Normalize feedback values to prevent extreme changes.
        """
        # Normalize task priorities (0-1)
        task_feedback = feedback.get("task_priorities", {})
        max_priority = max(task_feedback.values(), default=1)
        if max_priority > 1.0:
            for task_id in task_feedback:
                task_feedback[task_id] /= max_priority

        # Normalize agent behaviors (0-1)
        agent_feedback = feedback.get("agent_behaviors", {})
        max_behavior = max(agent_feedback.values(), default=1)
        if max_behavior > 1.0:
            for behavior in agent_feedback:
                agent_feedback[behavior] /= max_behavior

        feedback["task_priorities"] = task_feedback
        feedback["agent_behaviors"] = agent_feedback
        return feedback

    def resolve_conflicts(self, task_feedback, resource_feedback):
        """
        Resolve conflicting feedback using weighted priorities.
        """
        # Resolve task priorities
        total_priority = sum(task_feedback.values())
        if total_priority > 1.0:
            for task_id in task_feedback:
                task_feedback[task_id] /= total_priority  # Normalize to sum to 1

        # Resolve resource allocation
        total_resources = sum(resource_feedback.values())
        if total_resources > self.resources["total"]:
            scaling_factor = self.resources["total"] / total_resources
            for resource in resource_feedback:
                resource_feedback[resource] *= scaling_factor  # Scale down to fit limits

        return task_feedback, resource_feedback

    def calculate_entropy(self) -> float:
        try:
            # Example: Calculate Shannon entropy for agent positions
            positions = [(agent["x"], agent["y"]) for agent in self.agents]
            unique_positions, counts = np.unique(positions, axis=0, return_counts=True)
            probabilities = counts / counts.sum()
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Add small value to prevent log(0)
            return entropy
        except Exception as e:
            logger.error(f"Error calculating entropy for World {self.world_id}: {e}")
            return 0.0

    # Additional methods to update agents, tasks, resources, and communications can be added here
