from gekko import GEKKO
import math
m = GEKKO(remote=True)
print(math.cos(30*math.pi/180))
print(m.cos(30))

