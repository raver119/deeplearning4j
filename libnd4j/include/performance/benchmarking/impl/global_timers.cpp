// Yves Quemener on 2020-02-20

#include "../global_timers.h"

namespace sd {

GlobalTimers::GlobalTimers()
{
    timers = new std::chrono::high_resolution_clock::time_point[num_timers];
    line_numbers = new int[num_timers];
    labels = new int[num_timers];
    auto now=std::chrono::high_resolution_clock::now();
    for(int i=0;i<num_timers;i++)
    {
        timers[i] = now;
        labels[i] = 0;
        line_numbers[i] =0;
    }
    next_timer_ind=0;
    current_label_ind=0;
    next_label_ind=1;
    labels_to_ind["N/A"]=0;
    ind_to_labels[0]=std::string("N/A");
}

GlobalTimers* GlobalTimers::getInstance(){
    if (_instance == 0)
        _instance = new GlobalTimers();

    return _instance;
}

void GlobalTimers::stopWatch(const int line_no, const int label_id, int timer_id){
    if(timer_id==-1)
    {
        timer_id = next_timer_ind;
        next_timer_ind=(next_timer_ind+1)%num_timers;
    }
    timers[timer_id] = std::chrono::high_resolution_clock::now();
    line_numbers[timer_id] = line_no;
    labels[timer_id] = label_id;
}

void GlobalTimers::reset(int reset_to){
//    nd4j_printf("Reset\n", 0);
    auto now=std::chrono::high_resolution_clock::now();
    for(int i=0;i<num_timers;i++)
    {
        timers[i] = now;
        labels[i] = 0;
        line_numbers[i] =0;
    }
    next_timer_ind = reset_to;
    current_label_ind=0;
    labels_to_ind["N/A"]=0;
    ind_to_labels[0]=std::string("N/A");
}

void GlobalTimers::displayTimers(){
    for(int i=1;i<num_timers;i++){
        if(line_numbers[i-1]==0) continue;
        if(line_numbers[i]==0) continue;
        auto dt1 = std::chrono::duration_cast<std::chrono::nanoseconds> ((timers[i] - timers[i-1])).count();
        auto dt2 = std::chrono::duration_cast<std::chrono::microseconds> ((timers[i] - timers[i-1])).count();
        auto dt3 = std::chrono::duration_cast<std::chrono::milliseconds> ((timers[i] - timers[i-1])).count();
        nd4j_printf("%ld ns\t%ld us\t%ld ms\t: %d:%d\n", dt1, dt2, dt3, labels[i], line_numbers[i]);
    }
}

/*void GlobalTimers::setLabel(std::string lbl){
//    nd4j_printf("SetLabel %s\n", lbl.c_str());
    if(!labels_to_ind.count(lbl)>0){
        labels_to_ind[lbl] = next_label_ind;
        ind_to_labels[next_label_ind] = lbl;
        next_label_ind++;
    }
    current_label_ind = labels_to_ind[lbl];
//    nd4j_printf("currentlabel %d\n", current_label_ind);
}

void GlobalTimers::setLabelNum(int lbl){
    current_label_ind = lbl;
}*/

GlobalTimers* GlobalTimers::_instance = 0;

}
