import random
from typing import List, Tuple, Dict
from collections import Counter

class PokerEnvironment:
    def __init__(self):
        self.players = []  # Will hold two players' hands
        self.pot = 0
        self.current_bet = 0
        self.current_player = 0  # Track whose turn it is (0 or 1)
        self.possible_hands = [
            ['A♠', 'A♣'],  # AA
            ['K♠', 'K♣'],  # KK
            ['Q♠', 'Q♣']   # QQ
        ]
        self.deal_initial_cards()
    
    def _calculate_hand_strength(self, hand: List[str]) -> float:
        """Calculate the strength of a pocket pair"""
        # Simplified hand strength: AA = 1.0, KK = 0.8, QQ = 0.6
        rank = hand[0][0]  # Get first card's rank
        if rank == 'A':
            return 1.0
        elif rank == 'K':
            return 0.8
        else:  # QQ
            return 0.6
    
    def deal_initial_cards(self):
        """Deal pocket pairs to both players"""
        # Randomly select two different hands
        hands = random.sample(self.possible_hands, 2)
        self.players = hands
    
    def step(self, action: str) -> Tuple[Dict, float, bool]:
        """Process player action with simplified reward shaping"""
        reward = 0
        done = True
        
        # Get hand strengths
        current_hand = self.players[self.current_player]
        opponent_hand = self.players[1 - self.current_player]
        current_strength = self._calculate_hand_strength(current_hand)
        opponent_strength = self._calculate_hand_strength(opponent_hand)
        
        # Calculate rewards based on action and hand strength
        if action == 'fold':
            if current_strength > opponent_strength:
                reward = -1.0  # Big penalty for folding better hand
            else:
                reward = 0.2   # Small reward for folding worse hand
                
        elif action == 'call':
            if current_strength > opponent_strength:
                reward = 0.5   # Reward for calling with better hand
            else:
                reward = -0.5  # Penalty for calling with worse hand
                
        elif action == 'raise':
            if current_strength > opponent_strength:
                reward = 1.0   # Big reward for raising with better hand
            else:
                reward = -1.0  # Big penalty for raising with worse hand
        
        # Print debug info
        print(f"\nAction taken: {action}")
        print(f"Your hand: {current_hand} (strength: {current_strength:.1f})")
        print(f"Opponent hand: {opponent_hand} (strength: {opponent_strength:.1f})")
        print(f"Reward: {reward:.1f}")
        
        return self.get_state(), reward, done
    
    def get_state(self) -> Dict:
        return {
            'hand': self.players[self.current_player],
            'opponent_hand': self.players[1 - self.current_player],
            'pot': self.pot,
            'current_bet': self.current_bet,
            'current_player': self.current_player
        }
    
    def reset(self):
        """Reset the game state"""
        self.pot = 0
        self.current_bet = random.randint(10, 50)
        self.current_player = random.randint(0, 1)
        self.deal_initial_cards()
        return self.get_state() 