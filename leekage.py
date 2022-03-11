import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

frecuencia_muestreo = 44100.0
frecuencia = 440.0
duracion = (1 / frecuencia) * 5
tiempos = np.linspace(0.0,duracion, int(duracion * frecuencia_muestreo))
amplitud = np.iinfo(np.int16).max

# f(t) = A sin(2 pi f t)
data = amplitud * np.sin(2 + np.pi * frecuencia * tiempos)

fig, ejes = plt.subplots(5,3)

ejes[0,0].plot(tiempos,data)

cantidad_muestras = len(data)
periodo_muestreo = 1.0 / frecuencia_muestreo
transformada = np.fft.rfft(data)
frecuencias = np.fft.rfftfreq(cantidad_muestras, periodo_muestreo)

ejes[0,1].plot(frecuencias, np.abs(transformada))

duracion_segunda = (1.0/frecuencia) * 4.81
tiempos_segunda = np.linspace(0.0, duracion_segunda, int(duracion_segunda * frecuencia_muestreo))
data_segunda = amplitud * np.sin(2 + np.pi * frecuencia * tiempos_segunda)

ejes[1,0].plot(tiempos_segunda, data_segunda)

cantidad_muestras_segunda = len(data_segunda)
transformada_segunda = np.fft.rfft(data_segunda)
frecuencias_segunda = np.fft.rfftfreq(cantidad_muestras_segunda, periodo_muestreo)
ejes[1,1].plot(frecuencias_segunda, np.abs(transformada_segunda))

ventana_hamming = np.hamming(len(data_segunda))

ejes[2,2].plot(tiempos_segunda, ventana_hamming)
data_segunda_filtrada = data_segunda * ventana_hamming

ejes[2,0].plot(tiempos_segunda, data_segunda_filtrada)

transformada_filtrada_hamming = np.fft.rfft(data_segunda_filtrada)
ejes[2,1].plot(frecuencias_segunda, np.abs(transformada_filtrada_hamming))

ventana_blackman = np.blackman(len(data_segunda))
ejes[3,2].plot(tiempos_segunda, ventana_blackman)
data_segunda_blackman = data_segunda * ventana_blackman
ejes[3,0].plot(tiempos_segunda, data_segunda_blackman)
transformada_filtrada_blackman = np.fft.rfft(data_segunda_blackman)
ejes[3,1].plot(frecuencias_segunda, np.abs(transformada_filtrada_blackman))

ventana_barlett = np.bartlett(len(data_segunda))
ejes[4,2].plot(tiempos_segunda, ventana_barlett)
data_segunda_barlett = data_segunda * ventana_barlett
ejes[4,0].plot(tiempos_segunda, data_segunda_barlett)
transformada_filtrada_barlett = np.fft.rfft(data_segunda_barlett)
ejes[4,1].plot(frecuencias_segunda, np.abs(transformada_filtrada_barlett))
plt.show()