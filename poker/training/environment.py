from typing import List, Dict, Optional
from ..engine.game import ShortDeckPokerGame, Action, GamePhase
from ..engine.evaluator import evaluate_hand
from ..models.llm_agent import LLMPokerAgent

class PokerEnvironment:
    def __init__(self, num_players: int, starting_stack: int = 1000, small_blind: int = 5):
        self.game = ShortDeckPokerGame(num_players, starting_stack, small_blind)
        self.agents: Dict[int, LLMPokerAgent] = {}
        self.rewards: Dict[int, float] = {i: 0 for i in range(num_players)}
        
        # Initialize agents with different personalities
        personalities = ["aggressive", "conservative", "balanced", "exploitative"]
        for i in range(num_players):
            self.agents[i] = LLMPokerAgent(i, personalities[i % len(personalities)])

    def reset(self) -> None:
        """Reset the game state for a new hand."""
        self.game.reset_game()
        self.rewards = {i: 0 for i in range(self.game.num_players)}

    def step(self) -> bool:
        """
        Execute one step of the game (one player's action).
        Returns True if the hand is complete, False otherwise.
        """
        # Get the current player
        current_player = self.game.current_player
        
        # Skip if player is not active
        if not self.game.players[current_player].is_active:
            self._next_player()
            return False
        
        # Get game state and valid actions
        game_state = self.game.get_game_state(current_player)
        
        # Get action from agent
        agent = self.agents[current_player]
        action, amount = agent.get_action(game_state)
        
        # Apply the action
        self.game.apply_action(current_player, action, amount)
        
        # Update agent's history
        agent.update_history(action, amount, game_state)
        
        # Move to next player or phase
        return self._next_player()

    def _next_player(self) -> bool:
        """
        Move to the next player or phase.
        Returns True if the hand is complete, False otherwise.
        """
        active_players = [pid for pid, player in self.game.players.items() if player.is_active]
        
        # If only one player remains, they win the pot
        if len(active_players) == 1:
            winner = active_players[0]
            self.rewards[winner] += self.game.pot
            return True
        
        # Move to next player
        self.game.current_player = (self.game.current_player + 1) % self.game.num_players
        
        # If we've completed a betting round, move to next phase
        all_players_acted = all(
            not player.is_active or player.total_bet == self.game.current_bet
            for player in self.game.players.values()
        )
        
        if all_players_acted:
            return self._next_phase()
        
        return False

    def _next_phase(self) -> bool:
        """
        Move to the next phase of the game.
        Returns True if the hand is complete, False otherwise.
        """
        if self.game.phase == GamePhase.PREFLOP:
            self.game.phase = GamePhase.FLOP
            self.game.deal_community_cards()
        elif self.game.phase == GamePhase.FLOP:
            self.game.phase = GamePhase.TURN
            self.game.deal_community_cards()
        elif self.game.phase == GamePhase.TURN:
            self.game.phase = GamePhase.RIVER
            self.game.deal_community_cards()
        elif self.game.phase == GamePhase.RIVER:
            return self._showdown()
        
        # Reset betting for new phase
        self.game.current_bet = 0
        for player in self.game.players.values():
            player.total_bet = 0
        
        return False

    def _showdown(self) -> bool:
        """
        Evaluate hands and distribute pot to winner(s).
        Returns True as the hand is complete.
        """
        active_players = [pid for pid, player in self.game.players.items() if player.is_active]
        
        # Evaluate each player's hand
        hand_rankings = []
        for pid in active_players:
            player = self.game.players[pid]
            hand_rank, tiebreakers = evaluate_hand(player.cards, self.game.community_cards)
            hand_rankings.append((pid, hand_rank, tiebreakers))
        
        # Sort by hand rank and tiebreakers
        hand_rankings.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # Find winners (players with equal best hands)
        winners = []
        best_rank, best_tiebreakers = hand_rankings[0][1], hand_rankings[0][2]
        for pid, rank, tiebreakers in hand_rankings:
            if rank == best_rank and tiebreakers == best_tiebreakers:
                winners.append(pid)
        
        # Split pot among winners
        split_amount = self.game.pot // len(winners)
        for winner in winners:
            self.rewards[winner] += split_amount
        
        return True

    def run_episode(self, max_steps: int = 1000) -> Dict[int, float]:
        """
        Run a complete hand of poker.
        Returns the rewards for each player.
        """
        self.reset()
        step_count = 0
        
        while step_count < max_steps:
            step_count += 1
            if self.step():
                break
        
        return self.rewards

    def run_training_session(self, num_episodes: int = 100) -> List[Dict[int, float]]:
        """
        Run multiple episodes of poker for training.
        Returns the history of rewards for each episode.
        """
        reward_history = []
        
        for _ in range(num_episodes):
            rewards = self.run_episode()
            reward_history.append(rewards.copy())
        
        return reward_history 