from typing import List, Tuple, Dict
import os
from llama_cpp import Llama
from ..engine.game import Action, GamePhase, ShortDeckPokerGame

class LLMPokerAgent:
    def __init__(self, player_id: int, personality: str = "balanced", model_path: str = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
        """
        Initialize a poker agent using a local LLM.
        Args:
            player_id: The ID of the player
            personality: The playing style of the agent
            model_path: Path to the local GGUF model file
        """
        self.player_id = player_id
        self.personality = personality
        self.model = Llama(
            model_path=model_path,
            n_ctx=2048,  # Context window
            n_threads=4,  # Number of CPU threads to use
            n_batch=512,  # Batch size for prompt processing
        )
        self.action_history: List[Dict] = []

    def get_action(self, game_state: dict) -> Tuple[Action, int]:
        """
        Use local LLM to decide on the next poker action based on the current game state.
        Returns a tuple of (action, amount).
        """
        # Format the game state into a prompt
        system_prompt = self._get_system_prompt()
        user_prompt = self._format_prompt(game_state)
        
        # Combine prompts in chat format
        full_prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_prompt} [/INST]"""
        
        # Get model response
        response = self.model(
            full_prompt,
            max_tokens=64,
            temperature=0.7,
            stop=["</s>", "[/INST]", "\n"],
            echo=False
        )

        # Parse the response into an action
        action_str = response['choices'][0]['text'].strip().lower()
        return self._parse_action(action_str, game_state['valid_actions'])

    def _get_system_prompt(self) -> str:
        """Get the system prompt based on the agent's personality."""
        base_prompt = """You are a professional poker player. Your goal is to make optimal decisions in a short-deck poker game.
You must respond with exactly one action in the format: ACTION AMOUNT
Valid actions are: FOLD, CHECK, CALL, or RAISE amount
Example responses:
FOLD
CHECK
CALL
RAISE 100

Consider pot odds, position, and opponent tendencies in your decision."""

        personality_traits = {
            "aggressive": "\nYou prefer an aggressive playing style, favoring raises and bluffs when you have reasonable equity.",
            "conservative": "\nYou prefer a tight-conservative playing style, only playing strong hands and avoiding marginal situations.",
            "balanced": "\nYou maintain a balanced playing style, mixing up your play to remain unpredictable.",
            "exploitative": "\nYou focus on exploiting opponent tendencies and adjusting your strategy based on their patterns."
        }

        return base_prompt + personality_traits.get(self.personality, personality_traits["balanced"])

    def _format_prompt(self, game_state: dict) -> str:
        """Format the game state into a prompt for the LLM."""
        hole_cards = [str(card) for card in game_state['player_cards']]
        community_cards = [str(card) for card in game_state['community_cards']]
        
        prompt = f"""Current game state:
Hole cards: {', '.join(hole_cards)}
Community cards: {', '.join(community_cards) if community_cards else 'None'}
Phase: {game_state['phase'].value}
Pot: {game_state['pot']}
Current bet: {game_state['current_bet']}
Your stack: {game_state['player_stacks'][self.player_id]}
Your total bet this round: {game_state['player_bets'][self.player_id]}

Active players and their stacks:"""

        for pid, is_active in game_state['active_players'].items():
            if is_active and pid != self.player_id:
                prompt += f"\nPlayer {pid}: Stack {game_state['player_stacks'][pid]}, Bet {game_state['player_bets'][pid]}"

        prompt += "\n\nValid actions:"
        for action, amount in game_state['valid_actions']:
            if action in [Action.FOLD, Action.CHECK]:
                prompt += f"\n- {action.value}"
            else:
                prompt += f"\n- {action.value} (amount: {amount})"

        prompt += "\n\nWhat action do you take?"
        return prompt

    def _parse_action(self, action_str: str, valid_actions: List[Tuple[Action, int]]) -> Tuple[Action, int]:
        """Parse the LLM's response into a valid action and amount."""
        parts = action_str.split()
        action_name = parts[0].upper()
        amount = int(parts[1]) if len(parts) > 1 else 0

        # Map the action string to a valid Action enum
        action_map = {
            "FOLD": Action.FOLD,
            "CHECK": Action.CHECK,
            "CALL": Action.CALL,
            "RAISE": Action.RAISE
        }

        action = action_map.get(action_name)
        if not action:
            # Default to the first valid action if parsing fails
            return valid_actions[0]

        # Validate the action and amount against valid actions
        for valid_action, valid_amount in valid_actions:
            if valid_action == action:
                if action in [Action.FOLD, Action.CHECK]:
                    return (action, 0)
                elif action == Action.CALL:
                    return (action, valid_amount)
                elif action == Action.RAISE:
                    # Ensure raise amount is valid
                    return (action, max(valid_amount, amount))

        # Default to the first valid action if no match found
        return valid_actions[0]

    def update_history(self, action: Action, amount: int, game_state: dict) -> None:
        """Update the agent's action history for learning."""
        self.action_history.append({
            'action': action,
            'amount': amount,
            'game_state': game_state
        }) 