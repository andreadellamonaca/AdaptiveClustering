OBJS	= main.o adaptive_clustering.o error.o
SOURCE	= main.cpp adaptive_clustering.cpp error.cpp
HEADER	= adaptive_clustering.h error.h
OUT	    = adaptiveClustering.out
CC	    = g++
FLAGS	= -g -c -Wall
LFLAGS	= -l armadillo

all: $(OBJS)
	$(CC) -g $(OBJS) -o $(OUT) $(LFLAGS)

main.o: main.cpp
	$(CC) $(FLAGS) main.cpp

adaptive_clustering.o: adaptive_clustering.cpp
	$(CC) $(FLAGS) adaptive_clustering.cpp

error.o: error.cpp
	$(CC) $(FLAGS) error.cpp


clean:
	rm -f $(OBJS) $(OUT)
