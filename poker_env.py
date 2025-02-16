import random
from typing import List, Tuple, Dict
from collections import Counter

class PokerEnvironment:
    def __init__(self):
        self.deck = self._create_deck()
        self.players = []  # Will hold two players' hands
        self.pot = 0
        self.current_bet = 0
        self.community_cards = []
        self.current_player = 0  # Track whose turn it is (0 or 1)
        self.deal_initial_cards()
        
    def _create_deck(self) -> List[str]:
        suits = ['♠', '♣', '♥', '♦']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        return [f"{rank}{suit}" for suit in suits for rank in ranks]
    
    def _calculate_hand_strength(self, hand: List[str]) -> float:
        """Calculate the strength of a poker hand"""
        # Extract ranks and suits
        ranks = [card[:-1] for card in hand]
        suits = [card[-1] for card in hand]
        
        # Convert face cards to numbers
        rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, 
                      '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        numeric_ranks = [rank_values[r] for r in ranks]
        
        # Calculate basic hand strength factors
        strength = 0.0
        
        # High card value
        strength += max(numeric_ranks) / 14.0
        
        # Pair bonus
        if len(set(ranks)) < len(ranks):
            strength += 0.5
        
        # Suited bonus
        if len(set(suits)) == 1:
            strength += 0.3
        
        # Connected cards bonus
        if abs(numeric_ranks[0] - numeric_ranks[1]) == 1:
            strength += 0.2
        
        # High card combination bonus
        if min(numeric_ranks) >= 10:
            strength += 0.2
            
        return strength
    
    def _calculate_position_reward(self, position: int, num_players: int = 6) -> float:
        """Calculate reward modifier based on position"""
        # Later positions are better in poker
        return 0.1 * (position / num_players)
    
    def _calculate_pot_odds_reward(self, action: str) -> float:
        """Calculate reward based on pot odds and action"""
        if self.pot == 0:
            return 0
            
        pot_odds = self.current_bet / (self.pot + self.current_bet)
        
        if action == 'fold':
            # Reward folding with bad pot odds
            return 0.2 if pot_odds > 0.3 else -0.2
        elif action == 'call':
            # Reward calling with good pot odds
            return 0.2 if pot_odds < 0.3 else -0.2
        else:  # raise
            # Reward raising with strong hands
            hand_strength = self._calculate_hand_strength(self.players[0])
            return 0.3 if hand_strength > 0.7 else -0.3
    
    def deal_initial_cards(self):
        """Deal two cards to each player"""
        random.shuffle(self.deck)
        self.players = [
            [self.deck.pop(), self.deck.pop()],  # Player 0's hand
            [self.deck.pop(), self.deck.pop()]   # Player 1's hand
        ]
    
    def get_state(self) -> Dict:
        return {
            'hand': self.players[self.current_player],  # Current player's hand
            'opponent_hand': self.players[1 - self.current_player],  # Opponent's hand
            'pot': self.pot,
            'current_bet': self.current_bet,
            'current_player': self.current_player
        }
    
    def step(self, action: str) -> Tuple[Dict, float, bool]:
        """Process player action with sophisticated reward shaping"""
        reward = 0
        done = False
        
        # Base reward from hand strength
        current_hand_strength = self._calculate_hand_strength(self.players[self.current_player])
        opponent_hand_strength = self._calculate_hand_strength(self.players[1 - self.current_player])
        
        # Calculate relative hand strength compared to opponent
        relative_strength = current_hand_strength - opponent_hand_strength
        
        # Action-specific rewards with adjusted values
        if action == 'fold':
            if current_hand_strength > 0.7:
                reward = -1.0  # Big penalty for folding a strong hand
            elif current_hand_strength < 0.3:
                reward = 0.2   # Small reward for folding a weak hand
            else:
                reward = -0.2  # Small penalty for folding a medium hand
                
        elif action == 'call':
            if 0.3 <= current_hand_strength <= 0.7:
                reward = 0.5   # Good reward for calling with medium hand
            else:
                reward = -0.2  # Small penalty for calling with very weak or very strong hand
                
        elif action == 'raise':
            if current_hand_strength > 0.7:
                reward = 1.0   # Big reward for raising with strong hand
            elif current_hand_strength < 0.3:
                reward = -0.8  # Big penalty for raising with weak hand
            else:
                reward = 0.0   # Neutral for raising with medium hand
        
        # Add position and pot odds considerations
        reward += self._calculate_position_reward(self.current_player, 2)
        reward += self._calculate_pot_odds_reward(action)
        
        done = True  # End the episode after each action for now
        
        return self.get_state(), reward, done
    
    def reset(self):
        """Reset the game state"""
        self.deck = self._create_deck()
        random.shuffle(self.deck)
        self.pot = 0
        self.current_bet = random.randint(10, 50)
        self.current_player = random.randint(0, 1)  # Randomly choose starting player
        self.deal_initial_cards()
        return self.get_state() 