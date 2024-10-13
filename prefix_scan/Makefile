main: CVHelper.o malloc2D.o ocl.o main.o
	@g++ -o main *.o -g -m64 -lOpenCL -lpthread -fopenmp `pkg-config opencv4 --cflags --libs`

CVHelper.o: CVHelper.cpp
	@g++ -c CVHelper.cpp `pkg-config opencv4 --cflags --libs`

malloc2D.o: malloc2D.cpp
	@g++ -c malloc2D.cpp `pkg-config opencv4 --cflags --libs`

ocl.o: ocl.cpp
	@g++ -c ocl.cpp `pkg-config opencv4 --cflags --libs`

main.o: main.cpp
	@g++ -c main.cpp `pkg-config opencv4 --cflags --libs`