# 10.3 Variables Used by Implicit Rules
# https://www.gnu.org/software/make/manual/html_node/Implicit-Variables.html
CXX=g++
CXXFLAGS=-O2 -Wall -pedantic -Wextra -Wno-unused-parameter -std=c++17
INC=-I/usr/local/include/opencv4/ -isystem/usr/local/include/torch
LDFLAGS=-L/usr/local/lib/torch/ -lopencv_core -lopencv_imgproc \
				-lopencv_imgcodecs -lopencv_highgui -ltorch_cpu -lc10 -lspdlog

main: debug.out

debug.out: debug.cpp
	$(CXX) $(INC) debug.cpp -o debug.out $(CXXFLAGS) $(LDFLAGS)

.PHONY: clean

clean:
	rm *.out
