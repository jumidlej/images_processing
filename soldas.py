arquivo = open("soldas.txt", "r")
novo = open("new.txt", "w")

for line in arquivo:
    line = line.split()
    line[0] = float(line[0])
    # line[0] -= 53 # x
    line[0] /= 302
    line[1] = float(line[1])
    # line[1] -= 56
    line[1] /= 652
    novo.write("{} {}\n".format(line[0], line[1]))

arquivo.close
novo.close