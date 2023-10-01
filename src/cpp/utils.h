#ifndef UTILS_H
#define UTILS_H

#include <signal.h>

extern volatile sig_atomic_t e_flag;

static void signal_handler(int signum);

void install_signal_handler();

#endif /* UTILS_H */