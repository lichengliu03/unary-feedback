import os
import json
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from .. import (
    HistoryManager,
    EvaluationTurnLog,
    Room,
    ExplorationTurnLog,
    CognitiveMapTurnLog,
    Agent,
    RoomPlotter
)
from .visualization.visualization import HTMLGenerator


@dataclass
class EnvTurnLog:
    """Log data for a single environment turn."""
    turn_number: int
    user_message: str = ""  # Environment observation
    assistant_raw_message: str = ""  # Raw assistant input
    assistant_think_message: str = ""  # Think part of assistant message
    assistant_parsed_message: str = ""  # Parsed assistant action
    is_exploration_phase: bool = False
    is_last_exp: bool = False
    exploration_log: Optional["ExplorationTurnLog"] = None
    evaluation_log: Optional["EvaluationTurnLog"] = None
    cogmap_log: Optional["CognitiveMapTurnLog"] = None
    room_state: Optional["Room"] = None
    agent_state: Optional["Agent"] = None
    room_image: Optional[str] = None
    message_images: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    
    def to_dict(self):
        return {
            "turn_number": self.turn_number,
            "user_message": self.user_message,
            "assistant_raw_message": self.assistant_raw_message,
            "assistant_think_message": self.assistant_think_message,
            "assistant_parsed_message": self.assistant_parsed_message,
            "is_exploration_phase": self.is_exploration_phase,
            "is_last_exp": self.is_last_exp,
            "exploration_log": self.exploration_log.to_dict() if self.exploration_log else {},
            "evaluation_log": self.evaluation_log.to_dict() if self.evaluation_log else {},
            "cogmap_log": self.cogmap_log.to_dict() if self.cogmap_log else {},
            "room_state": self.room_state.to_dict() if self.room_state else {},
            "agent_state": self.agent_state.to_dict() if self.agent_state else {},
            "room_image": self.room_image,
            "message_images": self.message_images,
            "info": self.info
        }

class SpatialEnvLogger:
    """Logger for spatial environment data aggregation and visualization."""
    
    @staticmethod
    def _convert_omegaconf_to_python(obj):
        """Recursively convert OmegaConf objects to standard Python types for JSON serialization."""
        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
        elif isinstance(obj, dict):
            return {key: SpatialEnvLogger._convert_omegaconf_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [SpatialEnvLogger._convert_omegaconf_to_python(item) for item in obj]
        else:
            return obj

    @staticmethod
    def _plot_room(room_dict: Dict, agent_dict: Dict, out_dir: str, config_name: str, sample_idx: int, turn_idx: int) -> Optional[str]:
        """Plot room from state and return image filename"""
        img_name = f"room_turn_{turn_idx+1}.png" if turn_idx > 0 else "room_initial.png"
        img_path = os.path.join(out_dir, img_name)
        RoomPlotter.plot(Room.from_dict(room_dict), Agent.from_dict(agent_dict), mode='img', save_path=img_path)

        return os.path.join("images", config_name, f"sample_{sample_idx+1}", img_name)

    @staticmethod
    def _validate_and_assign_messages(message: List[Dict], turn_logs: List[Dict]) -> bool:
        """Validate message structure and assign raw assistant messages to turn logs."""
        if not message:
            return False
        
        # Remove system messages and check turn structure
        filtered_msgs = [msg for msg in message if msg.get("role") != "system"]
        
        # Check alternating user/assistant pattern
        for i in range(0, len(filtered_msgs), 2):
            if i >= len(filtered_msgs) or filtered_msgs[i].get("role") != "user":
                print(f"User message missing at index {i}")
                return False
            if i + 1 >= len(filtered_msgs) or filtered_msgs[i + 1].get("role") != "assistant":
                print(f"Assistant message missing at index {i+1}")
                return False
        
        # Assign raw assistant messages to turn logs
        assistant_messages = [msg['content'] for msg in message if msg.get("role") == "assistant"]
        
        if len(assistant_messages) != len(turn_logs):
            print(f"Mismatch: {len(assistant_messages)} assistant messages vs {len(turn_logs)} turns")
            return False
        
        return True


    @staticmethod
    def _save_data(aggregated_data: Dict, output_dir: str, model_name: str):
        """Save aggregated data to JSON and generate HTML dashboard."""
        saved_data = {
            'meta_info': {
                'model_name': model_name,
                'n_envs': len(aggregated_data.get('samples', {})),
            },
            **aggregated_data,
        }

        # Convert OmegaConf objects to standard Python types
        saved_data = SpatialEnvLogger._convert_omegaconf_to_python(saved_data)
        
        # os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        with open(os.path.join(output_dir, "env_data.json"), "w") as f:
            json.dump(saved_data, f, indent=2)

        # Generate HTML dashboard
        html_path = os.path.join(output_dir, "env_data.html")
        viz = HTMLGenerator(saved_data, html_path, True)
        dashboard_path = viz.generate_html()
        
        print(f"Environment data logged to {os.path.abspath(output_dir)}")
        print(f"Dashboard written to {os.path.abspath(dashboard_path)}")
        return output_dir
    


    @staticmethod
    def log_each_env_info(output_dir: str, model_config, save_images: bool = True):
        """Logs detailed information for each environment and overall performance metrics.

        New implementation that reads from directory structure:
        model_name/hash_value/vision_or_text/active_or_passive/

        Note: env_summaries and messages parameters are no longer used in this implementation,
        but kept for backwards compatibility.
        """

        # Use new directory-based aggregation instead of env_summaries
        # output_dir is results/debug, model_name from kwargs
        model_dir = HistoryManager.get_model_dir(output_dir, model_config)
        aggregated_data = HistoryManager.aggregate_from_directories(
            model_dir=model_dir,
            save_images=save_images,
        )

        return SpatialEnvLogger._save_data(aggregated_data, model_dir, model_name=model_config['model_name'])