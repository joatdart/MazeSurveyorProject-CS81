#!/usr/bin/env python

# PD controller
class PD():
    def __init__(self, kp, kd):
        self._p = kp # proportional gain
        self._d = kd # derivative gain
        
    ''' 
    Compute new actuation
    If there is no prev error, curr_err = prev_err and d_t is zero so that the d term will not be accounted for.
    '''
    def step(self, prev_err, curr_err, dt):
        u = self._p * curr_err # p term

        if dt != 0: # dt is zero if there is no prev error
            u += self._d * (curr_err - prev_err) / dt # add d term
        return u