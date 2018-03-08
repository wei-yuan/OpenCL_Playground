#ifndef _MAIN_H__
#define _MAIN_H__

#include <iostream>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>

#include <CL/cl.h>

using namespace std;

double get_event_exec_time(cl_event event);

cl_program load_program(cl_context context, cl_device_id device, const char *filename);

bool get_cl_context(cl_context *context, cl_device_id **devices, int num_platform);

static inline void check_err(size_t err_num, const char *statement)
{
    if (err_num)
    {
        cout << statement << endl;
        exit(err_num);
    }
}

#endif