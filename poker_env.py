import random
from typing import List, Tuple, Dict
from collections import Counter

class PokerEnvironment:
    def __init__(self):
        self.deck = self._create_deck()
        self.players = []
        self.pot = 0
        self.current_bet = 0
        self.community_cards = []  # Add community cards for Texas Hold'em
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
    
    def step(self, action: str) -> Tuple[Dict, float, bool]:
        """Process player action with sophisticated reward shaping"""
        reward = 0
        done = False
        
        # Base reward from hand strength
        hand_strength = self._calculate_hand_strength(self.players[0])
        reward += hand_strength - 0.5  # Normalize around 0
        
        # Position-based reward
        position_reward = self._calculate_position_reward(0)  # Assuming player position 0
        reward += position_reward
        
        # Pot odds and action-based reward
        pot_odds_reward = self._calculate_pot_odds_reward(action)
        reward += pot_odds_reward
        
        # Action-specific rewards
        if action == 'fold':
            reward -= self.current_bet / 100  # Small penalty for folding
            done = True
        elif action == 'call':
            self.pot += self.current_bet
            reward += 0.1  # Small reward for staying in
            done = True
        elif action == 'raise':
            self.pot += self.current_bet * 2
            reward += 0.2 * hand_strength  # Reward raising with strong hands
            done = True
            
        # Clip reward to reasonable range
        reward = max(min(reward, 1.0), -1.0)
        
        return self.get_state(), reward, done
    
    def reset(self):
        self.deck = self._create_deck()
        random.shuffle(self.deck)
        self.pot = 0
        self.current_bet = random.randint(10, 50)  # Random initial bet
        self.deal_initial_cards()
        return self.get_state()
    
    def get_state(self) -> Dict:
        return {
            'hand': self.players[0] if self.players else [],
            'pot': self.pot,
            'current_bet': self.current_bet
        }

    def deal_initial_cards(self):
        """Deal two cards to the player"""
        random.shuffle(self.deck)
        self.players = [[self.deck.pop(), self.deck.pop()]]  # Deal 2 cards to one player 