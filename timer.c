#include "timer.h"


#ifdef _WIN32   // Windows system specific
#include <windows.h>
#else          // Unix based system specific
#include <sys/time.h>
#endif


#ifdef _WIN32
LARGE_INTEGER frequency;                    // ticks per second
LARGE_INTEGER startCount;
LARGE_INTEGER endCount;
#else
struct timeval startCount;
struct timeval endCount;
#endif

int started = 0;


void timer_start() {
    started = 1; // reset stop flag
#ifdef _WIN32
    QueryPerformanceCounter(&startCount);
    QueryPerformanceFrequency(&frequency);
#else
    gettimeofday(&startCount, NULL);
#endif
}


double timer_stop() {
    if (!started)
        return 0.0;

#ifdef _WIN32
    QueryPerformanceCounter(&endCount);

    double startTimeInSec = (double)startCount.QuadPart / frequency.QuadPart;
    double endTimeInSec = (double)endCount.QuadPart / frequency.QuadPart;
#else
    gettimeofday(&endCount, NULL);

    double startTimeInSec = startCount.tv_sec + startCount.tv_usec / 1000000.0;
    double endTimeInSec = endCount.tv_sec + endCount.tv_usec / 1000000.0;
#endif

    return endTimeInSec - startTimeInSec;
}

