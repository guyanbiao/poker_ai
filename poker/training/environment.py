from typing import List, Dict, Optional
from ..engine.game import ShortDeckPokerGame, Action, GamePhase
from ..engine.evaluator import evaluate_hand
from ..models.llm_agent import LLMPokerAgent
import time

class PokerEnvironment:
    def __init__(self, num_players: int, starting_stack: int = 1000, small_blind: int = 5):
        print(f"Initializing poker environment with {num_players} players...")
        self.game = ShortDeckPokerGame(num_players, starting_stack, small_blind)
        self.agents: Dict[int, LLMPokerAgent] = {}
        self.rewards: Dict[int, float] = {i: 0 for i in range(num_players)}
        
        # Initialize agents with different personalities
        personalities = ["aggressive", "conservative", "balanced", "exploitative"]
        for i in range(num_players):
            print(f"Creating agent {i} with {personalities[i % len(personalities)]} personality...")
            self.agents[i] = LLMPokerAgent(i, personalities[i % len(personalities)])

    def reset(self) -> None:
        """Reset the game state for a new hand."""
        print("\nStarting new hand...")
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
            print(f"Player {current_player} is not active, skipping...")
            self._next_player()
            return False
        
        # Get game state and valid actions
        game_state = self.game.get_game_state(current_player)
        
        # Log current game state
        print(f"\nPhase: {game_state['phase'].value}")
        print(f"Current player: {current_player}")
        print(f"Pot: {game_state['pot']}")
        print(f"Current bet: {game_state['current_bet']}")
        print(f"Player cards: {', '.join(str(card) for card in game_state['player_cards'])}")
        if game_state['community_cards']:
            print(f"Community cards: {', '.join(str(card) for card in game_state['community_cards'])}")
        
        # Get action from agent
        print(f"Getting action from agent {current_player}...")
        start_time = time.time()
        agent = self.agents[current_player]
        action, amount = agent.get_action(game_state)
        elapsed = time.time() - start_time
        print(f"Agent {current_player} chose {action.value}" + (f" with amount {amount}" if amount > 0 else ""))
        print(f"Decision took {elapsed:.2f} seconds")
        
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
            print(f"\nPlayer {winner} wins pot of {self.game.pot} (others folded)")
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
            print("\nMoving to FLOP...")
            self.game.phase = GamePhase.FLOP
            self.game.deal_community_cards()
        elif self.game.phase == GamePhase.FLOP:
            print("\nMoving to TURN...")
            self.game.phase = GamePhase.TURN
            self.game.deal_community_cards()
        elif self.game.phase == GamePhase.TURN:
            print("\nMoving to RIVER...")
            self.game.phase = GamePhase.RIVER
            self.game.deal_community_cards()
        elif self.game.phase == GamePhase.RIVER:
            print("\nMoving to SHOWDOWN...")
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
        print("\nSHOWDOWN")
        active_players = [pid for pid, player in self.game.players.items() if player.is_active]
        
        # Evaluate each player's hand
        hand_rankings = []
        for pid in active_players:
            player = self.game.players[pid]
            hand_rank, tiebreakers = evaluate_hand(player.cards, self.game.community_cards)
            hand_rankings.append((pid, hand_rank, tiebreakers))
            print(f"Player {pid} hand: {', '.join(str(card) for card in player.cards)}")
        
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
        print(f"\nPot ({self.game.pot}) split among winners: {winners}")
        for winner in winners:
            self.rewards[winner] += split_amount
            print(f"Player {winner} wins {split_amount}")
        
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
        
        for episode in range(num_episodes):
            print(f"\n=== Starting Episode {episode + 1}/{num_episodes} ===")
            start_time = time.time()
            rewards = self.run_episode()
            elapsed = time.time() - start_time
            print(f"Episode completed in {elapsed:.2f} seconds")
            print("Rewards:", rewards)
            reward_history.append(rewards.copy())
        
        return reward_history 