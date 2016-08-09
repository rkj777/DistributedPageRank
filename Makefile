all: main.c Lab4_IO.c Lab4_IO.h timer.h
	mpicc -g -Wall main.c Lab4_IO.c timer.h -o main
	
clean:
	rm main