import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

frecuencia_muestreo = 44100
frecuencia = 500
tiempos = np.linspace(0.0,1.0,frecuencia_muestreo)
amplitud = np.iinfo(np.int16).max

ciclos = frecuencia * tiempos

fracciones, enteros = np.modf(ciclos)
data = fracciones

data = fracciones - 0.5

data = np.abs(data)

data = data - data.mean()

alto, bajo = abs(max(data)), abs(min(data))
data = amplitud * data / max(alto, bajo)

plt.figure()
plt.plot(tiempos,data)
plt.show()

write("triangular.wav", frecuencia_muestreo, data.astype(np.int16))

cantidad_muestras = len(data)
periodo_muestreo = 1.0 / frecuencia_muestreo
transformada = np.fft.rfft(data)
frecuencias = np.fft.rfftfreq(cantidad_muestras, periodo_muestreo)

plt.figure()
plt.plot(frecuencias, np.abs(transformada))
#plt.show()


#1 obtener en Hz las frecuencias de  los armónicos de la señal

print(frecuencias[transformada > 100000])

#2 aplicar filtro pasabajas que solo deje pasar la frecuencia fundamental y aplicar transformada inversa
#graficar y crear archivo wav 
pasa_bajas = transformada.copy()
pasa_bajas[frecuencias > frecuencia] *= 0

plt.plot(frecuencias, np.abs(pasa_bajas), label = "Espectro filtrado, pasa bajas")
plt.legend()
plt.show()

pasa_bajas_data = np.fft.irfft(pasa_bajas)

plt.plot(tiempos,pasa_bajas_data, label="Audio con pasa bajas")
plt.legend()
plt.show()

write("triangular_bajas.wav", frecuencia_muestreo, pasa_bajas_data.astype(np.int16))
