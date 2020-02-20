// Yves Quemener on 2020-02-20

#include "../global_timers.h"

namespace nd4j {

GlobalTimers::GlobalTimers()
{
    timers = new std::chrono::high_resolution_clock::time_point[200];
}

GlobalTimers* GlobalTimers::getInstance(){
    if (_instance == 0)
        _instance = new GlobalTimers();

    return _instance;
}

GlobalTimers* GlobalTimers::_instance = 0;

}
