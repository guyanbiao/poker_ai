from enum import Enum
from typing import List

class Suit(Enum):
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"
    SPADES = "♠"

class Rank(Enum):
    TEN = "T"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"

    @property
    def value_int(self) -> int:
        """Return the numeric value of the rank for comparison."""
        values = {
            "T": 10,
            "J": 11,
            "Q": 12,
            "K": 13,
            "A": 14
        }
        return values[self.value]

class Card:
    def __init__(self, rank: Rank, suit: Suit):
        self.rank = rank
        self.suit = suit
    
    def __str__(self) -> str:
        return f"{self.rank.value}{self.suit.value}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self) -> int:
        return hash((self.rank, self.suit))
    
    def __lt__(self, other: 'Card') -> bool:
        return self.rank.value_int < other.rank.value_int

def create_deck() -> List[Card]:
    """Create a short deck of 20 cards (Ten through Ace)."""
    return [Card(rank, suit) for rank in Rank for suit in Suit] 