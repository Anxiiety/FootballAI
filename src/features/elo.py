import math
from collections import defaultdict

class Elo:
    def __init__(self, k=20, base=1500, home_adv=60):
        self.k = k
        self.base = base
        self.home_adv = home_adv
        self.r = defaultdict(lambda: base)

    def expected(self, Ra, Rb):
        return 1.0 / (1 + math.pow(10, (Rb - Ra)/400))

    def update_match(self, home, away, home_goals, away_goals):
        Ra = self.r[home] + self.home_adv
        Rb = self.r[away]
        Ea = self.expected(Ra, Rb)
        Eb = 1 - Ea
        # risultato
        if home_goals > away_goals:
            Sa, Sb = 1, 0
        elif home_goals < away_goals:
            Sa, Sb = 0, 1
        else:
            Sa, Sb = 0.5, 0.5
        # aggiorna
        self.r[home] += self.k * (Sa - Ea)
        self.r[away] += self.k * (Sb - Eb)

    def rating(self, team):
        return self.r[team]
