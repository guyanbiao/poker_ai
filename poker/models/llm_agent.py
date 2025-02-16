from typing import List, Tuple, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..engine.game import Action, GamePhase, ShortDeckPokerGame

class LLMPokerAgent:
    def __init__(self, player_id: int, personality: str = "balanced", model=None, tokenizer=None):
        """
        Initialize a poker agent using the DeepSeek model.
        Args:
            player_id: The ID of the player
            personality: The playing style of the agent
            model: Optional pre-loaded model
            tokenizer: Optional pre-loaded tokenizer
        """
        print(f"Initializing LLM for agent {player_id}...")
        self.player_id = player_id
        self.personality = personality
        self.model = model
        self.tokenizer = tokenizer
        self.action_history: List[Dict] = []

    def get_action(self, game_state: dict) -> Tuple[Action, int]:
        """
        Use DeepSeek model to decide on the next poker action based on the current game state.
        Returns a tuple of (action, amount).
        """
        # Format the game state into a prompt
        system_prompt = self._get_system_prompt()
        user_prompt = self._format_prompt(game_state)
        
        # Combine prompts in DeepSeek's format
        full_prompt = (
            f"### Instruction: {system_prompt}\n\n"
            f"{user_prompt}\n\n"
            "### Response: Based on the game state, I choose to"
        )
        
        # Get model response
        print(f"Agent {self.player_id} thinking...")
        try:
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the response part
            response = response.split("### Response:")[-1].strip()
            print(f"Raw response: {response}")
            
            return self._parse_action(response, game_state['valid_actions'])
        except Exception as e:
            print(f"Error during inference: {e}")
            # Return a random valid action as fallback
            return self._get_fallback_action(game_state['valid_actions'])

    def _get_system_prompt(self) -> str:
        """Get the system prompt based on the agent's personality."""
        base_prompt = """You are a professional poker player. Your goal is to make optimal decisions in a short-deck poker game.
You must respond with exactly one action using these exact formats:
- To fold: "FOLD"
- To check: "CHECK"
- To call: "CALL"
- To raise: "RAISE X" (where X is the amount)

Example valid responses:
FOLD
CHECK
CALL
RAISE 100

Invalid response examples:
- "I want to fold"
- "Let's raise by 100"
- "Call the bet"
- "Fold this hand"

Consider pot odds, position, and opponent tendencies in your decision."""

        personality_traits = {
            "aggressive": "\nYou prefer an aggressive playing style, favoring raises and bluffs when you have reasonable equity.",
            "conservative": "\nYou prefer a tight-conservative playing style, only playing strong hands and avoiding marginal situations.",
            "balanced": "\nYou maintain a balanced playing style, mixing up your play to remain unpredictable.",
            "exploitative": "\nYou focus on exploiting opponent tendencies and adjusting your strategy based on their patterns.",
            "learner": "\nYou are learning to play poker through experience, focusing on making mathematically sound decisions based on pot odds and position."
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

        prompt += "\n\nWhat action do you take?, please respond with the action and amount like this: ACTION AMOUNT"
        return prompt

    def _parse_action(self, action_str: str, valid_actions: List[Tuple[Action, int]]) -> Tuple[Action, int]:
        """Parse the LLM's response into a valid action and amount."""
        print(f"Parsing action............: {action_str}")
        try:
            # Clean up the response
            action_str = action_str.strip().upper()
            if action_str.startswith("I CHOOSE TO "):
                action_str = action_str[12:]
            
            # Split into action and amount
            parts = action_str.split()
            action_name = parts[0]
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
                return self._get_fallback_action(valid_actions)

            # Validate the action and amount against valid actions
            for valid_action, valid_amount in valid_actions:
                if valid_action == action:
                    if action in [Action.FOLD, Action.CHECK]:
                        return (action, 0)
                    elif action == Action.CALL:
                        return (action, valid_amount)
                    elif action == Action.RAISE:
                        return (action, max(valid_amount, amount))

            # If we get here, the action wasn't valid
            return self._get_fallback_action(valid_actions)
            
        except (ValueError, IndexError) as e:
            print(f"Error parsing action: {e}")
            return self._get_fallback_action(valid_actions)

    def _get_fallback_action(self, valid_actions: List[Tuple[Action, int]]) -> Tuple[Action, int]:
        """Get a fallback action when parsing fails."""
        # Prefer CHECK or CALL if available
        for action, amount in valid_actions:
            if action in [Action.CHECK, Action.CALL]:
                return (action, amount)
        
        # Otherwise, take the first valid action (usually FOLD)
        return valid_actions[0]

    def update_history(self, action: Action, amount: int, game_state: dict) -> None:
        """Update the agent's action history for learning."""
        self.action_history.append({
            'action': action,
            'amount': amount,
            'game_state': game_state
        }) 