
// Yves Quemener on 2020-02-20

#ifndef LIBND4J_GLOBAL_TIMERS_H
#define LIBND4J_GLOBAL_TIMERS_H

#include <chrono>
#include <helpers/logger.h>
#include <string>
#include <map>


namespace nd4j {
class GlobalTimers {
private:
    static nd4j::GlobalTimers* _instance;
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

}

#endif
