#ifndef NMF_MONITOR_H
#define NMF_MONITOR_H

#ifdef _WIN32   // Windows system specific
#include <windows.h>
#else          // Unix based system specific

#include <sys/time.h>

#endif

class Monitor {
public:
    Monitor() {
    }

    ~Monitor() {
    }

    inline void start() {
        stopped = 0; // reset stop flag
#ifdef _WIN32
        QueryPerformanceCounter(&startCount);
#else
        gettimeofday(&startCount, NULL);
#endif
    }

    inline void stop() {
        stopped = 1; // set timer stopped flag
#ifdef _WIN32
        QueryPerformanceCounter(&endCount);
#else
        gettimeofday(&endCount, NULL);
#endif
    }

    /**
    *  compute elapsed time in micro-second resolution.
    *  other getElapsedTime will call this first, then convert to correspond resolution.
    */
    inline double getElapsedTimeInMicroSec() {
#ifdef _WIN32
        if (!stopped)
            QueryPerformanceCounter(&endCount);

        startTimeInMicroSec = startCount.QuadPart * (1000000.0 / frequency.QuadPart);
        endTimeInMicroSec = endCount.QuadPart * (1000000.0 / frequency.QuadPart);
#else
        if (!stopped)
            gettimeofday(&endCount, NULL);

        startTimeInMicroSec = (startCount.tv_sec * 1000000.0) + startCount.tv_usec;
        endTimeInMicroSec = (endCount.tv_sec * 1000000.0) + endCount.tv_usec;
#endif

        return endTimeInMicroSec - startTimeInMicroSec;
    }

    inline double getElapsedTimeInMilliSec() {
        return this->getElapsedTimeInMicroSec() * 0.001;
    }

    inline double getElapsedTime() {
        return this->getElapsedTimeInSec();
    }

    inline double getElapsedTimeInSec() {
        return this->getElapsedTimeInMicroSec() * 0.000001;
    }

private:
    double startTimeInMicroSec;                 // starting time in micro-second
    double endTimeInMicroSec;                   // ending time in micro-second
    int stopped;                             // stop flag
#ifdef _WIN32
    LARGE_INTEGER frequency;                    // ticks per second
    LARGE_INTEGER startCount;
    LARGE_INTEGER endCount;
#else
    timeval startCount;
    timeval endCount;
#endif
};
#endif //NMF_MONITOR_H
