CC=gcc
CFLAGS=-c -o3 -m64 -I${MKLROOT}/include
LDFLAGS=-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -lmpi
OBJS=dsanls.o common.o load.o timer.o main.o
EXECUTABLE=disNMF


all: ${EXECUTABLE}
	rm -f *.o

${EXECUTABLE}: ${OBJS}
	${CC} ${LDFLAGS} ${OBJS} -o $@

${OBJS}: %.o: %.c
	${CC} ${CFLAGS} $< -o $@

clean:
	rm -f *.o
	rm -f ${EXECUTABLE}
