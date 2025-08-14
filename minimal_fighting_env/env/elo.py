import math

class eloCalculator:
    k = 0
    def __init__(self, k=40):
        self.k = k

    def calculateRankChange(self, p1Elo, p2Elo, winner):
        p1 = p(p1Elo, p2Elo)
        p2 = 1 - p1

        p1Win = 0
        p2Win = 0

        if (winner == 0):
            p1Win = 1
        else:
            p2Win = 1

        return [p1Elo + self.k * (p1Win - p1), p2Elo + self.k * (p2Win - p2)]

def p(a, b):
    return (1.0 / (1.0 + math.pow(10.0, (b - a) / 400)))

if __name__ == "__main__":
    calc = eloCalculator()
    print(calc.calculateRankChange(1500, 1500, 0))