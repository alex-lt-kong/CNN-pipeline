CXX=g++
CXXFLAGS=-O2 -Wall -pedantic -Wextra -Wno-unused-parameter
INC=-I/usr/local/include/opencv4/ -I/usr/local/include/torch
LDFLAGS=-L/usr/local/lib/torch/ -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -ltorch_cpu -lc10

main: predict.out

predict.out: predict.cpp
	$(CXX) $(INC) predict.cpp -o predict.out $(CXXFLAGS) $(LDFLAGS)

.PHONY: clean

clean:
	rm *.out