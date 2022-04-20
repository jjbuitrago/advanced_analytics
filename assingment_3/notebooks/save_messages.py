import threading

# Helper thread to avoid the Spark StreamingContext from blocking Jupyter
        
class StreamingThread(threading.Thread):
    def __init__(self, ssc):
        super().__init__()
        self.ssc = ssc
    def run(self):
        self.ssc.start()
        self.ssc.awaitTermination()
    def stop(self):
        print('----- Stopping... this may take a few seconds -----')
        self.ssc.stop(stopSparkContext=False, stopGraceFully=True)
    

#Initializing PySpark
from pyspark import SparkContext, SparkConf

# #Spark Config
conf = SparkConf().setAppName("sample_app")
sc = SparkContext(conf=conf)

from pyspark.streaming import StreamingContext
ssc = StreamingContext(sc, 10)

lines = ssc.socketTextStream("localhost", 8080)
lines.saveAsTextFiles("/home/juanjosebuitrago/Documents/KU Leuven/Advanced Analytics/assingment_3/stream/")

ssc_t = StreamingThread(ssc)
ssc_t.start()

a = input("Press any key to stop ")

ssc_t.stop()