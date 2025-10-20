CC = mpic++

EXEF = exe

FILE = test
CLUSTER = cluster_check

SRC = $(FILE).cpp
EXE = $(FILE)
HOSTF = hosts

CFLAG = -g -Wall -std=c++11 -lm

NP = 8

ARGS = 0 1 1024 10 80 10

$(EXEF):
	mkdir $(EXEF)

compile: $(SRC) $(EXEF)
	$(CC) -o $(EXEF)/$(EXE) $(SRC) $(CFLAG)

run: $(EXEF)/$(EXE)
	mpirun --np $(NP) --hostfile ./$(HOSTF) ./$(EXEF)/$(EXE) $(ARGS)

check:
	$(CC) -o $(EXEF)/$(CLUSTER) $(CLUSTER).cpp $(CFLAG)

run_check:
	mpirun --np $(NP) --hostfile ./$(HOSTF) ./$(EXEF)/$(CLUSTER) $(ARGS)

clean:
	rm -rf $(EXEF)