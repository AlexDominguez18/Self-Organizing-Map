from xml.etree.ElementInclude import include
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches

class SOM():
    
    def __init__(self, tipo_vecindario = 'Cruz', tipo_distancia = 'Euclideana', tamanio_malla = 20, max_epocas = 500, taza_aprendizaje = 0.5):
        self.data_set = self.cargar_datos()
        self.tipo_vecindario = tipo_vecindario
        self.tipo_distancia = tipo_distancia
        self.tamanio_malla = tamanio_malla
        self.max_epocas = max_epocas
        self.taza_aprendizaje = taza_aprendizaje
        self.errores_acumulados = list()
    
    def cargar_datos(self):
        try:
            archivo = open('dataset.cvv')
            datos = []
            for l in archivo.readlines():
                a = l.split(',')
                a = list(map(int, a))[:-1]
                datos.append(a)
            archivo.close()
            return np.array(datos).T
        except FileNotFoundError:
            print('No se encontró el archivo')
    
    def entrenar(self):
        m = self.data_set.shape[0]
        n = self.data_set.shape[1]

        radio_inicial = max(self.tamanio_malla[0], self.tamanio_malla[1]) / 2

        tiempo = self.max_epocas / np.log(radio_inicial)

        col_maxes = self.data_set.mac(axis=0)
        self.data_set = self.data_set / col_maxes[np.newaxis, :]

        self.red = np.random.random((self.tamanio_malla[0], self.tamanio_malla[1], m))

        for i in range(self.max_epocas):
            error = []

            t = self.training_set[:, np.random.randint(0, n)].reshape(np.array([m,1 ]))

            bmu, bmu_i = self.bmu(t, self.red, m)

            radio = self.reducir_radio(radio_inicial, i, tiempo)
            aprendizaje = self.reducir_taza_aprendizaje(self.taza_aprendizaje, i, self.max_epocas)

            # Actualizacion de pesos
            for x in range (self.red.shape[0]):
                for y in range(self.red.shape[1]):
                    peso = self.red[x, y, :].reshape(m, 1)

                    if self.tipo_distancia == 'Euclideana':
                        w_dist = np.sum((np.arrya([x, y]) - bmu_i) ** 2)
                    elif self.tipo_distancia == 'Manhattan':
                        w_dist = np.sum(np.fabs(np.array([x, y]) - bmu_i))

                    if self.tipo_vecindario == 'Cruz' and w_dist <= radio ** 2:
                        influencia = self.calcular_influencia(w_dist, radio)
                        nuevo_peso = peso + (aprendizaje * influencia * (t - peso))

                        self.red[x, y, :] = nuevo_peso.reshape(1, m)
                        dist = abs(bmu.reshape(m) - np.mean(self.data_set, axis=1))
                        error.append(np.sqrt(np.dot(dist, dist.T)))

                    elif self.tipo_vecindario == 'Estrella' and w_dist <= radio**4:
                        influencia = self.calcular_influencia(w_dist, radio)

                        nuevo_peso = peso + (aprendizaje * influencia * (t - peso))

                        self.red[x, y, :] = nuevo_peso.reshape(1, m)
                        dist = abs(bmu.reshape(m) - np.mean(self.data_set, axis=1))
                        error.append(np.sqrt(np.dot(dist, dist.T)))
                self.errores_acumulados.append(float(sum(error) / float(len(self.data_set.T))))
    
    def bmu(self, t, red, m):
        bmu_i = np.array([0, 0])
        min_distancia = np.iinfo(np.int).max

        for x in range(red.shape[0]):
            for y in range(red.shape[1]):
                w = red[x, y, :].reshape(m, 1)

                if self.tipo_distancia == 'Euclideana':
                    distancia = np.sum((w - t) ** 2)
                elif self.tipo_distancia == 'Manhattan':
                    distancia = np.sum(np.fabs(w - t))
                
                if distancia < min_distancia:
                    min_distancia = distancia
                    bmu_i = np.array([x, y])
        bmu = red[bmu_i[0], bmu_i[1], :].reshape(m, 1)

        return (bmu, bmu_i)

    def reducir_radio(self, radio_inicial, i, tiempo):
        return radio_inicial * np.exp(-i / tiempo)
    
    def reducir_taza_aprendizaje(self, taza_inicial, i, max_epocas):
        return taza_inicial * np.exp(-i / max_epocas)

    def calcular_influencia(self, distancia, radio):
        return np.exp(-distancia / (2 * (radio**2)))

    def calcular_error(self):
        errores = list()
        for vector in self.data_set.T:
            indice = np.argmin(np.sum((self.red - vector) ** 2, axis=2))
            w = np.array([int(indice / self.red.shape[0]), indice % self.red.shape[1]])
            dist = self.red[w[0], w[1]] - vector
            errores.append(np.sqrt(np.dot(dist, dist.T)))
        return float(sum(errores) / float(len(self.training_set.T)))

    def graficar(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim((0, self.red.shape[0]+1))
        ax.set_ylim((0, self.red.shape[1]+1))
        ax.set_title('Self-Organising Map. Iteracion = %d' % self.max_ephocs)

        for x in range(1, self.red.shape[0] + 1):
            for y in range(1, self.red.shape[1] + 1):
                face_color = self.red[x-1,y-1,:]
                face_color = [sum(face_color[:3])/3,sum(face_color[3:6])/3, sum(face_color[6:])/4]
                ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
                            facecolor=face_color,
                            edgecolor='none'))
        plt.show()