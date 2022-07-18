import numpy as np
import pprint as pp
lst = [[],[]]
lst[0].append(0)
lst[1].append(1)
pp.pprint(int(np.array(lst[0])))

message = "You are so beautiful"
print(message.lower().split())
count = {}
# if count.get("I") is None:
#     count["I"] = 1
# else:
#     count["I"] += 1
count["I"] = 1
print(len(count))
lst = [[0]]*5
print(lst)