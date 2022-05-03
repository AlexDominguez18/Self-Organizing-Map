from cProfile import label
from xml.etree.ElementInclude import include
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from matplotlib.widgets import TextBox
import matplotlib

class SOM():
    textos=[]
    def __init__(self, tipo_vecindario = 'Cruz', tipo_distancia = 'Euclideana', max_epocas = 500, taza_aprendizaje = 0.5, filename='dataset.csv',tamanio_malla = [20,20],ventana=None):
        
        self.tipo_vecindario = tipo_vecindario
        self.tipo_distancia = tipo_distancia
        self.tamanio_malla = np.array(tamanio_malla)
        self.max_epocas = max_epocas
        self.taza_aprendizaje = taza_aprendizaje
        self.filename=filename
        self.ventana=ventana
        self.errores_acumulados = list()
        self.data_set = self.cargar_datos()
        
    
    def cargar_datos(self):
        try:
            archivo = open(self.filename,'r')
            datos = []
            for l in archivo.readlines():                 
                b=l.strip('\n')        
                a = b.split(',')            
                a = list(map(int,a))                
                datos.append(a)
            archivo.close()        
            data=np.array(datos).T   
            if self.ventana is not None:
                self.ventana.text_box_dimensiones.set_val("Dim: "+str(data.shape))
                #self.ventana.text_box_dimensiones.set_text("Dim: "+str(data.shape))         
            return data
        except FileNotFoundError:
            print('No se encontró el archivo')
        
    
    def entrenar(self):
        m = self.data_set.shape[0]
        n = self.data_set.shape[1]

        radio_inicial = max(self.tamanio_malla[0], self.tamanio_malla[1]) / 2

        tiempo = self.max_epocas / np.log(radio_inicial)

        col_maxes = self.data_set.max(axis=0)
        self.data_set = self.data_set / col_maxes[np.newaxis, :]
        #print("dataset")
        #print(self.data_set)
        #print("tamaño de malla")
        #print(self.tamanio_malla)
        self.red = np.random.random((self.tamanio_malla[0], self.tamanio_malla[1], m))
        #print("la net es esta: --------------------------------------")
        #print(self.red)

        for i in range(self.max_epocas):
            

            t = self.data_set[:, np.random.randint(0, n)].reshape(np.array([m,1 ]))
            
            bmu, bmu_i = self.encontrar_bmu(t, self.red, m)

            radio = self.reducir_radio(radio_inicial, i, tiempo)
            aprendizaje = self.reducir_taza_aprendizaje(self.taza_aprendizaje, i, self.max_epocas)

            # Actualizacion de pesos
            for x in range (self.red.shape[0]):
                for y in range(self.red.shape[1]):
                    peso = self.red[x, y, :].reshape(m, 1)

                    if self.tipo_distancia == 'Euclideana':
                        w_dist = np.sum((np.array([x, y]) - bmu_i) ** 2)
                    elif self.tipo_distancia == 'Manhattan':
                        w_dist = np.sum(np.fabs(np.array([x, y]) - bmu_i))

                    if self.tipo_vecindario == 'Cruz' and w_dist <= radio ** 2:
                        influencia = self.calcular_influencia(w_dist, radio)
                        nuevo_peso = peso + (aprendizaje * influencia * (t - peso))

                        self.red[x, y, :] = nuevo_peso.reshape(1, m)
                        dist = abs(bmu.reshape(m) - np.mean(self.data_set, axis=1))
                        

                    elif self.tipo_vecindario == 'Estrella' and w_dist <= radio**4:
                        influencia = self.calcular_influencia(w_dist, radio)

                        nuevo_peso = peso + (aprendizaje * influencia * (t - peso))

                        self.red[x, y, :] = nuevo_peso.reshape(1, m)
                        dist = abs(bmu.reshape(m) - np.mean(self.data_set, axis=1))
                        
                
    
    def encontrar_bmu(self, t, red, m):
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

    def calcular_pertenencia_de_dataset(self):
        
        m = self.data_set.shape[0]
        n = self.data_set.shape[1]
        self.text_map=np.zeros((self.red.shape[0],self.red.shape[1]), dtype=int)
        
        for i in range(n):
            x=self.data_set[:, i].reshape(np.array([m,1 ]))
            bmu, bmu_i = self.encontrar_bmu(x, self.red, m)
            print("El dato numero:"+str(i)+"pertenece a la neurona:"+str(bmu_i))            
            self.text_map[bmu_i[0]][bmu_i[1]]+=1
        for x in range(1, self.red.shape[0] + 1):
                for y in range(1, self.red.shape[1] + 1):
                    self.ventana.grafica.text(x,y,str(self.text_map[x-1][y-1]),fontsize=8)
        


    def graficar(self):
        col_aux=[]
        x_aux=[]
        y_aux=[]
        if self.ventana is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
            ax.set_xlim((0, self.red.shape[0]+1))
            ax.set_ylim((0, self.red.shape[1]+1))
            ax.set_title('Self-Organising Map. Iteracion = %d' % self.max_epocas)

            for x in range(1, self.red.shape[0] + 1):
                for y in range(1, self.red.shape[1] + 1):
                    face_color = self.red[x-1,y-1,:]
                    face_color = [sum(face_color[:3])/3,sum(face_color[3:6])/3, sum(face_color[6:])/4]
                    ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
                                facecolor=face_color,
                                edgecolor='none'))
        else:
            self.ventana.grafica.set_aspect(1)
            self.ventana.grafica.set_xlim((0, self.red.shape[0]+1))
            self.ventana.grafica.set_ylim((0, self.red.shape[1]+1))
            self.ventana.grafica.set_title('Self-Organising Map. Iteracion = %d' % self.max_epocas)
            cmap = matplotlib.cm.get_cmap('viridis')
            for x in range(1, self.red.shape[0] + 1):
                for y in range(1, self.red.shape[1] + 1):
                    face_color = self.red[x-1,y-1,:]
                    promedio=sum(face_color[:])/len(face_color)
                    
                    x_aux.append(x)
                    y_aux.append(y)
                    col_aux.append(promedio*100)
                    
                    #self.ventana.grafica.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
                     #           facecolor=rgba,
                      #          edgecolor='none'))
                    #self.ventana.grafica.text(x,y,str(x)+','+str(y),fontsize=8)
        #self.ventana.plot_colorbar(x_aux,y_aux,col_aux)
        
        
        s = ((self.ventana.grafica.get_window_extent().width  / (self.red.shape[0]+1.) * 72./self.ventana.fig.dpi) ** 2)
        
        self.ventana.grafica.scatter(x=x_aux, y=y_aux, c=col_aux, cmap="viridis",marker='.',s=s)
        self.calcular_pertenencia_de_dataset()

        plt.show()
            