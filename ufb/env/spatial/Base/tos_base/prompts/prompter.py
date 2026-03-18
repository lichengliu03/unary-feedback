import numpy as np
from typing import Optional
from .. import Room, Agent, ActionSequence, EvaluationManager
from ..utils.room_utils import get_room_description
from ..core.relationship import (
    PairwiseRelationship, 
    PairwiseRelationshipDiscrete, 
    ProximityRelationship, 
    DegreeRel, OrientationRel
)
from .prompts import *
from ..utils.utils import THINK_LABEL, ANSWER_LABEL

class Prompter:
    # Dynamic, reusable format blocks
    def _build_format_rules(self, is_exploration: bool) -> str:
        if self.enable_think:
            think = "[Your thoughts on next step actions]" if is_exploration else "[Your thoughts on the question]"
            answer = "Actions: [ ... ]" if is_exploration else "[your answer]"
            fmt = f"{THINK_LABEL}\n{think}\n{ANSWER_LABEL}\n{answer}"
        else:
            answer = "Actions: [ ... ]" if is_exploration else "[your answer]"
            fmt = f"{ANSWER_LABEL}\n{answer}"
        return (
            "!!! IMPORTANT OUTPUT RULES !!!\n"
            "1. You must always output in this format (labels followed by a newline):\n"
            f"   {fmt}\n"
            f"2. Inside {ANSWER_LABEL}, include ONLY the required answer. No extra text, notes, or formatting.\n"
            "   - No bullet points, prose, boxes, calculations, or explanations.\n"
            "3. Any deviation is invalid."
        )

    def get_format_footer(self, is_exploration: bool) -> str:
        # Decide answer hint
        if is_exploration:
            answer_hint = "Actions: [ ... ]"
        else:
            # Special stricter format for InternVL during evaluation
            if self._is_internvl_model():
                answer_hint = "[ONLY the letter (A, B, C, ...)]"
            else:
                answer_hint = "[your answer (only required answer, no extra text, notes, formatting or anything else)]"

        if self.enable_think:
            think = "[Your thoughts on next step actions]" if is_exploration else "[Your thoughts on the question]"
            return f"Strictly follow this format:\n{THINK_LABEL}\n{think}\n{ANSWER_LABEL}\n{answer_hint}"
        else:
            return f"Strictly follow this format:\n{ANSWER_LABEL}\n{answer_hint}"

    # Add image prompt constants
    TOPDOWN_PROMPT = "\n\nTopdown view: {placeholder}\n{object_info}"
    # OBLIQUE_PROMPT = "\n\nOblique view: {placeholder}\n{object_info}"

    def __init__(self, config, np_random: np.random.RandomState, image_handler = None):
        self.config = config
        self.image_handler = image_handler
        self.np_random = np_random
        self.enable_think = bool(self.config.prompt_config.get('enable_think', True))

    def _is_internvl_model(self) -> bool:
        """Return True if current model is InternVL (e.g., internvl3_5)."""
        model_cfg = self.config.get_model_config()
        model_name = str((model_cfg or {}).get('model_name', '')).lower()
        print(f"Model name: {model_name}")
        return 'internvl' in model_name

    def _get_topdown_prompt(self, prompt_template: str, room) -> str:
        """Generate topdown view prompt with object information."""
        obj_info = "Each object in the room is labeled with a numerical marker for easy identification."
        for idx, obj in enumerate(room.objects):
            obj_info += f"\nObject {idx + 1}: {obj.name}"
        return prompt_template.format(placeholder=self.config.image_placeholder, object_info=obj_info)

    def _get_oblique_prompt(self, prompt_template: str, room) -> str:
        """Generate oblique view prompt with object information."""
        obj_info = "Each object in the room is labeled with a numerical marker for easy identification."
        for idx, obj in enumerate(room.objects):
            obj_info += f"\nObject {idx + 1}: {obj.name}"
        return prompt_template.format(placeholder=self.config.image_placeholder, object_info=obj_info)

    def get_initial_observation_prompt(
            self,
            room: Room,
            agent: Agent,
            eval_manager: Optional[EvaluationManager] = None,
            exp_history = None
        ) -> dict:
        """
        Generates the initial observation prompt based on the exploration type.
        """
        obs = {}
        is_vision, is_active = self.config.render_mode == 'vision', self.config.exp_type == 'active'
        topdown = self.config.prompt_config['topdown']

        room_desc = get_room_description(room, agent, with_topdown=topdown)

        observation_instructions = (
            PairwiseRelationship.prompt()
            + f"\n{DegreeRel.prompt()}"
            + f"\n{OrientationRel.prompt()}"
            + f"\n{PairwiseRelationshipDiscrete.prompt()}"
        )
        if not is_vision:
            observation_instructions += f"\n{ProximityRelationship.prompt()}"

        exp_instructions = ""
        if is_active:
            exp_instructions = f"Action Instructions:\n{ActionSequence.get_usage_instructions(is_vision)}"
            exp_instructions += f"\n\nYou have a maximum of {self.config.max_exp_steps} exploration steps."

        images = None
        if is_vision:
            images = [self.image_handler.get_image('instruction'), self.image_handler.get_image('label')]
            if is_active and topdown:
                room_desc += self._get_topdown_prompt(self.TOPDOWN_PROMPT, room)
                images.append(self.image_handler.get_image('topdown'))
            if not is_active:
                if topdown:
                    images.append(self.image_handler.get_image('topdown'))
                else:
                    images.extend(exp_history['multi_modal_data'][self.config.image_placeholder])

        exp_history_str = ""
        if not is_active:
            exp_history_str = f"## Exploration History\n{exp_history['obs_str']}" if not topdown else ""

        template = INSTRUCTION_TEMPLATE_VISION if is_vision else INSTRUCTION_TEMPLATE_TEXT

        fmt_kwargs = {
            'title': 'Spatial Exploration Task' if is_active else 'Spatial Reasoning Task',
            'intro': SHARED_INTRO_TEXT if not is_vision else SHARED_INTRO_VISION,
            'goal_lines': (
                'Goal: **Minimize total COST** while building a complete and accurate map of the environment.'
                if is_active else ''
            ),
            'format_rules': self._build_format_rules(is_active),
            'observation_instructions': observation_instructions,
            'exp_instructions': exp_instructions,
            'room_info': room_desc,
            'multiroom_rules': SHARED_MULTIROOM_RULES,
            'active_rules_extra': ACTIVE_RULES_EXTRA if is_active else '',
            'rules_common': SHARED_RULES_COMMON,
            'exp_history': exp_history_str if not is_active else '',
            'vision_example': (VISION_EXAMPLE.format(image_placeholder=self.config.image_placeholder) if is_vision else ''),
        }

        obs_str = template.format(**fmt_kwargs)
        if not is_active:
            obs_str += f"\n{self.get_evaluation_prompt(eval_manager)}"
        if is_vision:
            obs['multi_modal_data'] = {self.config.image_placeholder: images}

        obs['obs_str'] = obs_str + "\n" + self.get_format_footer(is_active)
        return obs
        
            

    def get_evaluation_prompt(self, eval_manager: EvaluationManager) -> str:
        """Generate the evaluation prompt."""
        eval_question = eval_manager.get_current_question()
        assert eval_question, "No question found after exploration phase"
        return EVALUATION_INSTRUCTION.format(eval_question=f"## Evaluation Question\n{eval_question}")
