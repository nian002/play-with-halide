/**
 * Copyright 2019 Xiaobin Wei <xiaobin.wee@gmail.com>
 */

#ifndef UTILS_H_
#define UTILS_H_


#include <cstddef>
#include <string>


/**
 * Scoped timer class
 * 
 * Automatically print time elapsed to stdout
 */
class ScopeTimer {
 public:
    explicit ScopeTimer(const char *name);
    ~ScopeTimer();

   /**
    * Restart another timer
    * 
    * This will reset the name and start point
    * and make it a not finished timer
    */
    void reset(const char *name);

   /**
    * Finish the timer and print the result
    * 
    * After a timer is finished, DO NOT use it again
    */
    void finish();


 private:
    std::string mName;
    bool mFinished;
};

#endif   // UTILS_H_
