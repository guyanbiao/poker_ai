from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import random
from .card import Card, create_deck

class Action(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"

class GamePhase(Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"

@dataclass
class PlayerState:
    id: int
    stack: int
    cards: List[Card]
    is_active: bool = True
    total_bet: int = 0

class ShortDeckPokerGame:
    def __init__(self, num_players: int, starting_stack: int, small_blind: int):
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = small_blind * 2
        self.deck = create_deck()
        self.community_cards: List[Card] = []
        self.players: Dict[int, PlayerState] = {}
        self.current_player: int = 0
        self.pot: int = 0
        self.current_bet: int = 0
        self.phase = GamePhase.PREFLOP
        self.button_pos: int = 0
        self.reset_game()

    def reset_game(self) -> None:
        """Reset the game state for a new hand."""
        self.deck = create_deck()
        random.shuffle(self.deck)
        self.community_cards = []
        self.players = {
            i: PlayerState(id=i, stack=self.starting_stack, cards=[])
            for i in range(self.num_players)
        }
        self.pot = 0
        self.current_bet = 0
        self.phase = GamePhase.PREFLOP
        self.deal_hole_cards()
        self.post_blinds()

    def deal_hole_cards(self) -> None:
        """Deal two cards to each player."""
        for _ in range(2):
            for player in self.players.values():
                if player.is_active:
                    player.cards.append(self.deck.pop())

    def post_blinds(self) -> None:
        """Post small and big blinds."""
        sb_pos = (self.button_pos + 1) % self.num_players
        bb_pos = (self.button_pos + 2) % self.num_players
        
        # Post small blind
        self.players[sb_pos].stack -= self.small_blind
        self.players[sb_pos].total_bet = self.small_blind
        
        # Post big blind
        self.players[bb_pos].stack -= self.big_blind
        self.players[bb_pos].total_bet = self.big_blind
        
        self.current_bet = self.big_blind
        self.pot = self.small_blind + self.big_blind

    def get_valid_actions(self, player_id: int) -> List[Tuple[Action, int]]:
        """Get list of valid actions for the current player."""
        player = self.players[player_id]
        valid_actions = []

        # Can always fold
        valid_actions.append((Action.FOLD, 0))

        # Can check if no bet to call
        if self.current_bet == player.total_bet:
            valid_actions.append((Action.CHECK, 0))
        
        # Can call if there's a bet to call and have enough chips
        call_amount = self.current_bet - player.total_bet
        if call_amount > 0 and call_amount <= player.stack:
            valid_actions.append((Action.CALL, call_amount))

        # Can raise if have enough chips
        min_raise = self.current_bet * 2
        if min_raise <= player.stack:
            valid_actions.append((Action.RAISE, min_raise))

        return valid_actions

    def apply_action(self, player_id: int, action: Action, amount: int = 0) -> None:
        """Apply the chosen action to the game state."""
        player = self.players[player_id]

        if action == Action.FOLD:
            player.is_active = False
        elif action == Action.CALL:
            call_amount = self.current_bet - player.total_bet
            player.stack -= call_amount
            player.total_bet += call_amount
            self.pot += call_amount
        elif action == Action.RAISE:
            raise_amount = amount - player.total_bet
            player.stack -= raise_amount
            player.total_bet = amount
            self.current_bet = amount
            self.pot += raise_amount

    def deal_community_cards(self) -> None:
        """Deal community cards based on the current phase."""
        if self.phase == GamePhase.FLOP:
            self.community_cards.extend([self.deck.pop() for _ in range(3)])
        elif self.phase in [GamePhase.TURN, GamePhase.RIVER]:
            self.community_cards.append(self.deck.pop())

    def get_game_state(self, player_id: int) -> dict:
        """Get the observable game state for a given player."""
        return {
            'player_cards': self.players[player_id].cards,
            'community_cards': self.community_cards,
            'pot': self.pot,
            'current_bet': self.current_bet,
            'player_stacks': {pid: p.stack for pid, p in self.players.items()},
            'player_bets': {pid: p.total_bet for pid, p in self.players.items()},
            'active_players': {pid: p.is_active for pid, p in self.players.items()},
            'phase': self.phase,
            'valid_actions': self.get_valid_actions(player_id)
        } 