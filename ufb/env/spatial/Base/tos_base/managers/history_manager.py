from typing import Optional, List, Dict
import os
import shutil
import json

from ..utils.cogmap.correlation import compute_correlation_metrics
from ..utils.utils import hash
from ..utils.room_utils import RoomPlotter
from .. import (
    Agent,
    Room,
    EvaluationManager,
    ExplorationManager,
    CognitiveMapManager,
)

# -------- Filenames (constants) --------
EXPLORATION_LOG_BASENAME = "exploration_turn_logs.json"
EVALUATION_LOG_BASENAME = "evaluation_turn_logs.json"
CONFIG_BASENAME = "config.json"
METRICS_BASENAME = "metrics.json"
IMAGES_DIRNAME = "images"
class HistoryManager:
    """Simple conversation history manager, one history manager for one run.
    Store only env turn logs in a single JSON file
    Directory structure: model_name/room_hash_key/vision_or_text/active_or_passive/
    Example: gpt-4o/1d54fa/vision/active/
    """

    def __init__(self, observation_config: Dict, model_config: Dict , room_dict: Dict, agent_dict: Dict, output_dir:str, 
                 image_dir:str = None, eval_override: bool = False, all_override: bool = False, task_type: str = None):
        # only explore turn logs are saved
        self.exploration_turn_logs: List[Dict] = []
        self.evaluation_turn_logs: Dict[str, Dict[str, Dict]] = {}
        self.exp_type = observation_config['exp_type']
        self.model_path= HistoryManager.get_model_dir(output_dir, model_config)
        self.sample_path = os.path.join(self.model_path,self._generate_room_key(room_dict, agent_dict))
        self.output_dir = os.path.abspath(os.path.join(
            self.sample_path,
            observation_config['render_mode'],
            observation_config['exp_type'],
            "think" if observation_config['prompt_config']["enable_think"] else "nothink",
        ))
        if observation_config['exp_type'] == 'passive':
            self.output_dir = os.path.join(self.output_dir, observation_config["proxy_agent"])
        self.exploration_path = os.path.join(self.output_dir, EXPLORATION_LOG_BASENAME)
        self.evaluation_path = os.path.join(self.output_dir, EVALUATION_LOG_BASENAME)
        self.model_config_path = os.path.join(self.model_path, CONFIG_BASENAME)
        self.sample_config_path = os.path.join(self.sample_path, CONFIG_BASENAME)
        self.metrics_path = os.path.join(self.output_dir, METRICS_BASENAME)

        # Apply granular overrides
        if all_override:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)

        self._load()
        if eval_override:
            if task_type in self.evaluation_turn_logs:
                self.evaluation_turn_logs[task_type] = {}

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, IMAGES_DIRNAME), exist_ok=True)
        if not os.path.exists(self.model_config_path):
            with open(self.model_config_path, "w") as f:
                json.dump(model_config, f, ensure_ascii=False, indent=2)
        if not os.path.exists(self.sample_config_path):
            assert image_dir is not None
            sample_cfg = {
                "room_dict": room_dict,
                "agent_dict": agent_dict,
                "image_dir": image_dir,
            }
            with open(self.sample_config_path, "w") as f:
                json.dump(sample_cfg, f, ensure_ascii=False, indent=2)


    def has_exploration(self, index):
        return 0 <= index < len(self.exploration_turn_logs)

    def _generate_room_key(self, room_dict, agent_dict):
        room_str = json.dumps({**room_dict, **agent_dict}, sort_keys=True)

        return hash(room_str)
        
    def _load(self):
        """Load env turn logs from JSON file"""
        if os.path.exists(self.exploration_path):
            with open(self.exploration_path, "r") as f:
                self.exploration_turn_logs = json.load(f)
        if os.path.exists(self.evaluation_path):
            with open(self.evaluation_path, "r") as f:
                self.evaluation_turn_logs = json.load(f)

    def save_exploration(self) -> None:
        """Save env turn logs to JSON file"""
        if self.exploration_turn_logs:
            with open(self.exploration_path, "w") as f:
                json.dump(self.exploration_turn_logs, f, ensure_ascii=False, indent=2)

    def save(self) -> None:
        """Save env turn logs to JSON file"""
        if self.exploration_turn_logs:
            with open(self.exploration_path, "w") as f:
                json.dump(self.exploration_turn_logs, f, ensure_ascii=False, indent=2)
        with open(self.evaluation_path, "w") as f:
            json.dump(self.evaluation_turn_logs, f, ensure_ascii=False, indent=2)
        # Also compute and save metrics for this sample
        metrics = self._compute_sample_metrics()
        with open(self.metrics_path, "w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


    
    def update_turn_log(self, turn_log: Dict):
        """
            Add or update a turn log with all necessary data including images
            Must be added in sequence
        """
        if turn_log['is_exploration_phase']:
            assert not self.has_exploration(turn_log['turn_number'] - 1)
            if turn_log['room_state'] and turn_log['agent_state']:
                img_path = os.path.join(self.output_dir, IMAGES_DIRNAME, f"room_turn_{turn_log['turn_number']}.png")
                RoomPlotter.plot(Room.from_dict(turn_log['room_state']), Agent.from_dict(turn_log['agent_state']), mode='img', save_path=img_path)
                turn_log['room_image'] = img_path
            #invalid
            self.exploration_turn_logs.append(turn_log)
        else:
            assert turn_log['evaluation_log']
            assert turn_log['room_state'] and turn_log['agent_state']

            task_type = turn_log['evaluation_log']['task_type']
            question_id = turn_log['evaluation_log']['evaluation_data']['id']

            img_path = os.path.join(self.output_dir, IMAGES_DIRNAME, f"room_{task_type}_{question_id}.png")
            RoomPlotter.plot(Room.from_dict(turn_log['room_state']), Agent.from_dict(turn_log['agent_state']), mode='img', save_path=img_path)
            turn_log['room_image'] = img_path

            # Initialize task type if it doesn't exist
            if task_type not in self.evaluation_turn_logs:
                self.evaluation_turn_logs[task_type] = {}

            # Store the question with its ID
            self.evaluation_turn_logs[task_type][question_id] = turn_log

    def get_responses(self) -> List[Dict]:
        return [log.get('assistant_raw_message') for log in self.exploration_turn_logs if log.get('assistant_raw_message') is not None]


    def update_cogmap(self, turn_log: Dict) -> None:
        """Update cognitive map response for a specific turn"""
        if turn_log['is_exploration_phase']:
            turn_idx = turn_log['turn_number'] - 1
            assert 0 <= turn_idx < len(self.exploration_turn_logs)
            self.exploration_turn_logs[turn_idx]['cogmap_log'] = turn_log['cogmap_log']
        else:
            assert turn_log['evaluation_log']
            assert self.exp_type == 'active'
            task_type = turn_log['evaluation_log']['task_type']
            question_id = turn_log['evaluation_log']['evaluation_data']['id']

            assert task_type in self.evaluation_turn_logs
            assert question_id in self.evaluation_turn_logs[task_type]

            self.evaluation_turn_logs[task_type][question_id]['cogmap_log'] = turn_log['cogmap_log']

    def get_cogmap(self, turn_idx) -> Optional[Dict]:
        """Get cognitive map response for a specific turn (0-indexed)"""
        if not (0 <= turn_idx < len(self.exploration_turn_logs)):
            return None
        return self.exploration_turn_logs[turn_idx].get('cogmap_log')

    def has_question(self, question_id: str) -> bool:
        """Check if a question with the given ID already exists in evaluation logs"""
        return any(question_id in questions for questions in self.evaluation_turn_logs.values())

    def get_eval_counts(self) -> Dict[str, int]:
        """Return number of completed eval questions per task class name."""
        return {task_type: len(questions or {}) for task_type, questions in self.evaluation_turn_logs.items()}

    def get_existing_question_ids(self, task_type: str) -> List[str]:
        """Return list of existing question ids for a given task class name."""
        return list((self.evaluation_turn_logs.get(task_type) or {}).keys())

    @staticmethod
    def get_model_dir(output_dir: str, model_config: Dict) -> str:
        """Generate a unique directory name for the model configuration"""
        #TODO may be a minor diff leads to a different hash
        for k in [k for k, v in model_config.items() if v is None]:
            model_config.pop(k)
        model_config.pop("api_key", None) 
        model_config.pop("base_url", None)
        model_config.pop("max_retries", None)
        model_config.pop("timeout", None)
        model_config_str = json.dumps(model_config, sort_keys=True)
        # model_name = model_config['model_name'].replace("/", "-") + "_" + hash(model_config_str)
        model_name = model_config['model_name'].replace("/", "-")
        return os.path.join(output_dir, model_name)
    
    @staticmethod
    def aggregate_from_directories(model_dir: str, save_images: bool = True) -> Dict:
        """
        Aggregate data from new directory structure:
        base_dir/model_name/hash_value/vision_or_text/active_or_passive/

        Returns:
            Aggregated data dictionary with config_groups organized by text/vision + active/passive combinations
        """
        assert os.path.exists(model_dir), f"Model directory {model_dir} does not exist"

        samples = {}
        all_config_keys = set()

        # Scan all sample directories (room keys)
        sample_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]

        # Sequential numbering only for valid samples (with valid subdirs)
        for sample_dir in sample_dirs:
            sample_path = os.path.join(model_dir, sample_dir)

            # Collect subdirectories containing log files
            subdirs: List[str] = []
            for root, _, files in os.walk(sample_path):
                if EXPLORATION_LOG_BASENAME in files or EVALUATION_LOG_BASENAME in files:
                    subdirs.append(root)

            # Skip if subdirs is empty (invalid sample)
            if not subdirs:
                continue
            with open(os.path.join(sample_path, CONFIG_BASENAME), 'r') as f:
                sample_cfg = json.load(f)
            sample_key = f"sample_{os.path.basename(sample_cfg['image_dir'])}"
            assert sample_key not in samples, f"Duplicate sample key {sample_key}"
            samples[sample_key] = {}

            for combo_path in subdirs:
                rel = os.path.relpath(combo_path, sample_path)
                if rel in (".", ""):
                    continue
                config_key = rel.replace(os.sep, "_")

                sample_data = HistoryManager._load_sample_data(
                    combo_path=combo_path,
                    sample_key=sample_key,
                    save_images=save_images,
                    model_dir=model_dir,
                )
                if sample_data:
                    samples[sample_key][config_key] = sample_data
                    all_config_keys.add(config_key)

        # Initialize result structure with samples
        result = {
            "samples": samples,
            "exp_summary": {"group_performance": {}},
            "eval_summary": {"group_performance": {}},
            "cogmap_summary": {"group_performance": {}},
            "correlation": {"group_performance": {}},
        }

        # Aggregate performance for each config combination across all samples
        for config_name in sorted(all_config_keys):
            env_data_list = []
            for sample_data_dict in samples.values(): # config_key -> sample_data
                if config_name in sample_data_dict and sample_data_dict[config_name] is not None:
                    env_data_list.append(sample_data_dict[config_name])

            if env_data_list:
                result["exp_summary"]["group_performance"][config_name] = ExplorationManager.aggregate_group_performance(env_data_list)
                result["eval_summary"]["group_performance"][config_name] = EvaluationManager.aggregate_group_performance(env_data_list)
                # Provide both exploration and evaluation cogmap summaries
                exp_type = "active" if "active" in config_name else "passive"
                result["cogmap_summary"]["group_performance"][config_name] = CognitiveMapManager.aggregate_group_performance(env_data_list, exp_type=exp_type)
                result["correlation"]["group_performance"][config_name] = compute_correlation_metrics(env_data_list, exp_type=exp_type)
        return result

    @staticmethod
    def _load_sample_data(combo_path: str, sample_key: str, save_images: bool, model_dir: str) -> Optional[Dict]:
        """Load data from a single sample's combination directory"""
        exploration_file = os.path.join(combo_path, EXPLORATION_LOG_BASENAME)
        evaluation_file = os.path.join(combo_path, EVALUATION_LOG_BASENAME)
        config_file = os.path.join(combo_path, CONFIG_BASENAME)
        metrics_file = os.path.join(combo_path, METRICS_BASENAME)

        sample_data = {
            "sample_id": sample_key,
            "env_turn_logs": [],  # Only exploration turn logs
            "evaluation_tasks": {},  # Separate storage for evaluation tasks
            "config": {},
            "metrics": {},
        }

        # Load exploration turn logs
        if os.path.exists(exploration_file):
            with open(exploration_file, 'r') as f:
                exploration_logs = json.load(f)
            sample_data["env_turn_logs"] = exploration_logs if exploration_logs else [] # Only exploration logs

        # Load evaluation turn logs - store each task separately
        if os.path.exists(evaluation_file):
            with open(evaluation_file, 'r') as f:
                evaluation_logs = json.load(f)
            # Store each evaluation task separately
            sample_data["evaluation_tasks"] = evaluation_logs if evaluation_logs else {}

        # Load config and metrics if present
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                sample_data["config"] = json.load(f)
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                sample_data["metrics"] = json.load(f)

        # Process image paths if save_images is enabled
        if save_images:
            # Process exploration turn logs
            for turn_log in sample_data["env_turn_logs"]:
                if turn_log.get("room_image"):
                    turn_log['room_image'] = os.path.relpath(turn_log['room_image'], model_dir)
                if turn_log.get('message_images'):
                    turn_log['message_images'] = [os.path.relpath(img_path, model_dir) for img_path in turn_log['message_images']]

            # Process evaluation tasks
            for task in sample_data["evaluation_tasks"].values():
                for question_data in task.values():
                    if question_data.get("room_image"):
                        question_data['room_image'] = os.path.relpath(question_data['room_image'], model_dir)
                    if question_data.get('message_images'):
                        question_data['message_images'] = [os.path.relpath(img_path, model_dir) for img_path in question_data['message_images']]

        return sample_data if sample_data["env_turn_logs"] or sample_data["evaluation_tasks"] else None        

    def _compute_sample_metrics(self) -> Dict:
        env_data = {
            "env_turn_logs": self.exploration_turn_logs,
            "evaluation_tasks": self.evaluation_turn_logs,
        }
        return {
            "exploration": ExplorationManager.aggregate_per_sample(env_data),
            "evaluation": EvaluationManager.aggregate_per_sample(env_data),
            "cogmap": CognitiveMapManager.aggregate_per_sample(env_data, exp_type=self.exp_type),
        }

