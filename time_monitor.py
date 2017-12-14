from threading import Thread
import time
class TimeMonitor(Thread):
    
    def run(self):
        start = time.time()
        while(True):
            time.sleep(5)
            curr = time.time()
            print ("5s passed")
            print ("diff is now: " + str(curr-start))
        
