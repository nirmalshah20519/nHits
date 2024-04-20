from neuralforecast.core import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.utils import AirPassengersDF
import time

nf = NeuralForecast(
    models = [NHITS(input_size=24, h=12, max_steps=100)],
    freq = 'M'
)
start = time.time()
nf.fit(df=AirPassengersDF)
print("Training Completed in :", time.time()-start , " s")
y=nf.predict()
print(y)