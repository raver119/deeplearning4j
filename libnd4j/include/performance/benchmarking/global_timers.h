
// Yves Quemener on 2020-02-20

#ifndef LIBND4J_GLOBAL_TIMERS_H
#define LIBND4J_GLOBAL_TIMERS_H

#include <chrono>
#include <helpers/logger.h>
#include <string>
#include <map>


namespace sd {
class GlobalTimers {
private:
    static GlobalTimers* _instance;
    const int num_timers = 20000;

public:
    std::chrono::high_resolution_clock::time_point *timers;
    int *line_numbers;
    int *labels;
    int current_label_ind;
    int next_label_ind;
    std::map<int, std::string> ind_to_labels;
    std::map<std::string, int> labels_to_ind;
    int next_timer_ind;

    GlobalTimers();
    static GlobalTimers* getInstance();
    void stopWatch(const int line_no=-1, int label_id=-1, int timer_id=-1);
    void reset(int reset_to=0);
    void displayTimers();
};

#define FIRST_TIMER(x, y) GlobalTimers* _global_timers = GlobalTimers::getInstance(); _global_timers->stopWatch(x,y);
#define GLOBAL_TIMER(x, y) _global_timers->stopWatch(x,y);

}

#endif
