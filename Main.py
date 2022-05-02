from SOM import SOM
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button

class Ventana():

    def __init__(self):
        #Configuracion inicial de la interfaz grafica.
        mpl.rcParams['toolbar'] = 'None'
        self.fig, (self.grafica) = plt.subplots(1)
        self.fig.canvas.manager.set_window_title('Mapas Auto-Organizados - SOM(Self-Organizing  Maps)')
        self.fig.set_size_inches(9, 7, forward=True)
        plt.subplots_adjust(bottom=0.220, top=0.920)
        self.grafica.set_xlim(-5.0,5.0)
        self.grafica.set_ylim(-5.0,5.0)
        self.fig.suptitle('Mapas auto-organizados - SOM')
        

        # Acomodo de los botones y cajas de texto (x,y,heigh,width)
        coordenadas_rejillas_neuronas = plt.axes([0.390, 0.10, 0.07, 0.03])
        coordenadas_vecindad = plt.axes([0.210, 0.05, 0.07, 0.03])
        coordenadas_epocas = plt.axes([0.390, 0.05, 0.07, 0.03])
        cordenadas_abrir_archivo = plt.axes([0.550, 0.10, 0.13, 0.03])
        coordenadas_boton_SOM = plt.axes([0.550, 0.05, 0.13, 0.03])
        
        
        self.text_box_rejillas_neuronas = TextBox(coordenadas_rejillas_neuronas, "Tamaño de la rejilla de neuronas: ")
        self.text_box_vecindad = TextBox(coordenadas_vecindad, "Vecindad: ")
        self.text_box_epocas = TextBox(coordenadas_epocas, "Épocas: ")
        boton_abrir_archivo = Button(cordenadas_abrir_archivo, "Abrir archivo")
        boton_SOM = Button(coordenadas_boton_SOM, "SOM")
        
        self.text_box_rejillas_neuronas.on_submit(self.fijar_rejilla_neuronas)
        self.text_box_vecindad.on_submit(self.validar_vecindad)
        self.text_box_epocas.on_submit(self.validar_epocas)
        boton_abrir_archivo.on_clicked(self.abrir_archivo)
        boton_SOM.on_clicked(self.inicializar_SOM)
        plt.show()


    def abrir_archivo(self,event):
        pass


    def fijar_rejilla_neuronas(self,event):
        pass


    def validar_vecindad(self,event):
        pass


    def validar_epocas(self,event):
        pass


    def inicializar_SOM(self,event):
        som = SOM(max_epocas=5000)
        som.entrenar()
        som.graficar()


if __name__ == "__main__":
    Ventana()