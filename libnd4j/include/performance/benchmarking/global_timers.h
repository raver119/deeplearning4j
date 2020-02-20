
// Yves Quemener on 2020-02-20

#ifndef LIBND4J_GLOBAL_TIMERS_H
#define LIBND4J_GLOBAL_TIMERS_H

#include <chrono>
#include <dll.h>
#include <helpers/logger.h>


namespace nd4j {
class ND4J_EXPORT GlobalTimers {
private:
    static GlobalTimers* _instance;

public:
    std::chrono::high_resolution_clock::time_point *timers;

    GlobalTimers();
    static GlobalTimers* getInstance();
};

}

#endif
