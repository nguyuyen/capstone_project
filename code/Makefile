CC = mpic++

EXEF = exe

FILE = test

SRC = $(FILE).cpp
EXE = $(FILE)
HOSTF = hosts

CFLAG = -g -Wall -std=c++11 -lm

NP=8

ARGS="1000 10 80 10"

$(EXEF):
	mkdir $(EXEF)

compile: $(SRC) $(EXEF)
	$(CC) -o $(EXEF)/$(EXE) $(SRC) $(CFLAG)

run: $(EXEF)/$(EXE)
	mpirun --np $(NP) --hostfile ./$(HOSTF) ./$(EXEF)/$(EXE) $(ARGS)

clean:
	rm -rf $(EXEF)