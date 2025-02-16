from typing import List, Tuple
from collections import Counter
from .card import Card, Rank, Suit

class HandRank:
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9

def evaluate_hand(hole_cards: List[Card], community_cards: List[Card]) -> Tuple[int, List[int]]:
    """
    Evaluate a poker hand and return a tuple of (hand_rank, [tiebreaker values]).
    The tiebreaker values are used to break ties between hands of the same rank.
    """
    all_cards = hole_cards + community_cards
    ranks = [card.rank.value_int for card in all_cards]
    suits = [card.suit for card in all_cards]
    
    # Check for flush
    suit_counts = Counter(suits)
    flush_suit = next((suit for suit, count in suit_counts.items() if count >= 5), None)
    flush_cards = [card for card in all_cards if card.suit == flush_suit] if flush_suit else []
    
    # Check for straight
    unique_ranks = sorted(set(ranks))
    straight = find_straight(unique_ranks)
    
    # Check for straight flush
    straight_flush = None
    if flush_suit and straight:
        flush_ranks = [card.rank.value_int for card in flush_cards]
        straight_flush = find_straight(sorted(set(flush_ranks)))
    
    # Count rank frequencies
    rank_counts = Counter(ranks)
    sorted_ranks = sorted(rank_counts.items(), key=lambda x: (-x[1], -x[0]))
    
    # Determine hand rank and tiebreakers
    if straight_flush:
        high_card = max(straight_flush)
        if high_card == 14:  # Ace-high straight flush is a royal flush
            return (HandRank.ROYAL_FLUSH, [])
        return (HandRank.STRAIGHT_FLUSH, [high_card])
    
    if any(count == 4 for _, count in sorted_ranks):
        quads_rank = next(rank for rank, count in sorted_ranks if count == 4)
        kicker = next(rank for rank, count in sorted_ranks if count == 1 and rank != quads_rank)
        return (HandRank.FOUR_OF_A_KIND, [quads_rank, kicker])
    
    if any(count == 3 for _, count in sorted_ranks) and any(count == 2 for _, count in sorted_ranks):
        trips_rank = next(rank for rank, count in sorted_ranks if count == 3)
        pair_rank = next(rank for rank, count in sorted_ranks if count == 2)
        return (HandRank.FULL_HOUSE, [trips_rank, pair_rank])
    
    if flush_suit:
        flush_values = sorted([card.rank.value_int for card in flush_cards], reverse=True)[:5]
        return (HandRank.FLUSH, flush_values)
    
    if straight:
        return (HandRank.STRAIGHT, [max(straight)])
    
    if any(count == 3 for _, count in sorted_ranks):
        trips_rank = next(rank for rank, count in sorted_ranks if count == 3)
        kickers = [rank for rank, count in sorted_ranks if count == 1][:2]
        return (HandRank.THREE_OF_A_KIND, [trips_rank] + kickers)
    
    pairs = [rank for rank, count in sorted_ranks if count == 2]
    if len(pairs) >= 2:
        kicker = next(rank for rank, count in sorted_ranks if count == 1)
        return (HandRank.TWO_PAIR, pairs[:2] + [kicker])
    
    if len(pairs) == 1:
        kickers = [rank for rank, count in sorted_ranks if count == 1][:3]
        return (HandRank.PAIR, pairs + kickers)
    
    high_cards = [rank for rank, _ in sorted_ranks][:5]
    return (HandRank.HIGH_CARD, high_cards)

def find_straight(unique_ranks: List[int]) -> List[int]:
    """Find the highest straight in a list of unique ranks."""
    if len(unique_ranks) < 5:
        return []
    
    # Check for regular straight
    for i in range(len(unique_ranks) - 4):
        if unique_ranks[i+4] - unique_ranks[i] == 4:
            return unique_ranks[i:i+5]
    
    # Check for Ace-low straight (A-5)
    if 14 in unique_ranks:  # Ace present
        ace_low = [14] + [r for r in unique_ranks if r <= 5]
        if len(ace_low) >= 5 and ace_low[-1] - ace_low[1] == 3:
            return sorted(ace_low[:5])
    
    return [] 