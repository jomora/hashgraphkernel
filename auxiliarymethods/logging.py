import time
import datetime

def logNow(): return "[" + datetime.datetime.now().replace(microsecond=0).isoformat() + "]"
    
def format_time(millis):
    return datetime.datetime.fromtimestamp(millis).strftime('%Y-%m-%d %H:%M:%S.%f')

def time_it(f,*args,**kwargs):
    start = time.time()
    print(logNow() + " [HGK] Started function %s at %s" % (f.__name__,format_time(start)))
    res = f(*args,**kwargs)
    end = time.time()
    print(logNow() + " [HGK] Ended function %s at %s" % (f.__name__,format_time(end)))
    print(logNow() + " [HGK] Duration in [s]:\t%s" % str(end - start))
    return res
