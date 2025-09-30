CC = mpic++

EXEF = exe

FILE = test

SRC = $(FILE).cpp
EXE = $(FILE)
HOSTF = hosts

CFLAG = -g -Wall -std=c++11 -lm
DEBUG_FLAG = -DDEBUG -DWRITE_TO_FILE

NP=8

$(EXEF):
	mkdir $(EXEF)

compile: $(SRC) $(EXEF)
	$(CC) -o $(EXEF)/$(EXE) $(SRC) $(CFLAG)

run: $(EXEF)/$(EXE)
	mpirun --np $(NP) --hostfile ./$(HOSTF) ./$(EXEF)/$(EXE) $(NUMS)

clean:
	rm -rf $(EXEF)