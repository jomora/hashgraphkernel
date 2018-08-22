import time
import datetime

def format_time(millis):
    return datetime.datetime.fromtimestamp(millis).strftime('%Y-%m-%d %H:%M:%S.%f')

def time_it(f,*args,**kwargs):
    start = time.time()
    print("Started function %s at %s" % (f.__name__,format_time(start)))
    res = f(*args,**kwargs)
    end = time.time()
    print("Ended function %s at %s" % (f.__name__,format_time(end)))
    print("Duration in [s]:\t%s\n" % str(end - start))
    return res
