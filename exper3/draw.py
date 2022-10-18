import os

title = ''
Size = []
Gflops = []
with open("./log.05") as f:
    title = f.readline()
    print(f"Title:{title}\n")
    f.readline()
    for line in f:
        size = line[6:10]
        size = int(''.join(list(filter(str.isdigit, size))))
        Size.append(size)
        gflops = line[line.find("s:"):line.find("s:")+7]
        gflops = eval(''.join(list(filter(lambda ch: ch in '0123456789.', gflops))))
        Gflops.append(gflops)
print("Size:\n")
for s in Size:
    print(s)

print("Gflops:\n")
for g in Gflops:
    print(g)