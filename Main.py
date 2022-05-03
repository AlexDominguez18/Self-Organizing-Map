
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, RadioButtons
import matplotlib as mpl
from SOM import SOM

from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

class Ventana:
    dataset  = []
    som=None
    epoca_actual=0
    epocas_maximas=0
    
    #rango=0.1
    texto_de_epoca = None
    #texto_de_convergencia= None
    #error_minimo=0.1
    #neuronas_capa=None
    neuronas_rejijlla=None
    #rango_inicializado=False
    pesos_inicializados=False
    #som_entrenado=False
    #lineas=[]
    puntos_rejilla=[]
    termino=False
    #errores=[]
    #errores_quick=[]
    puntos_barrido=[]
    #clase=0
    #colores = ['b', 'r', 'g', 'm', 'c', 'y']
    #marcadores = ['x', '.', '^', 's']
    #comobinacion_marcadores=[color + marker for marker, color in zip(marcadores, colores)]
    #marcadores_de_linea=[c + '-' for c in reversed(colores)]
    #tipo_gradiente=0
    #quick_entrenado=False
    #entrenando_quick =False
    som_entrenado=False
    tipo_rejilla=0
    archivo_seleccionado=None


    def __init__(self):
        #Configuracion inicial de la interfaz grafica.
       
        mpl.rcParams['toolbar'] = 'None'
        self.fig,self.grafica = plt.subplots()
        self.fig.canvas.manager.set_window_title('Self Organinzed Maps - SOM')
        self.fig.set_size_inches(9, 7, forward=True)
        plt.subplots_adjust(bottom=0.220, top=0.920)
        self.grafica.set_xlim(-5.0,5.0)
        self.grafica.set_ylim(-5.0,5.0)
        self.fig.suptitle("Algoritmo SOM")
        
        # Acomodo de los botones y cajas de texto
        cordenadas_archivo = plt.axes([0.210, 0.10, 0.125, 0.03])
        self.coordenadas_dimensiones = plt.axes([0.210, 0.07, 0,0])
        coordenadas_epcoas = plt.axes([0.520, 0.10, 0.07, 0.03])        
        coordenadas_rejilla=plt.axes([0.520, 0.05, 0.07, 0.03])
        coordenadas_vecinidad = plt.axes([0.770, 0.04, 0.13, 0.1])
        coordenadas_pesos = plt.axes([0.600, 0.05, 0.125, 0.03])

        #coordenadas_clase = plt.axes([0.510, 0.10, 0.125, 0.03])
        #coordenadas_quick = plt.axes([0.650, 0.05, 0.1, 0.03])
        coordenadas_entrenar_SOM= plt.axes([0.600, 0.10, 0.125, 0.03])
        
        
        self.boton_archivo =Button(cordenadas_archivo, "Archivo")
        #self.texto_dimensiones = self.grafica.text(
        #            -5,
        #            -7.3,
        #            "Dimensiones: ",
        #            fontsize=10
        #        ) 
        self.text_box_epocas = TextBox(coordenadas_epcoas, "Épocas maximas: ")
        self.text_box_rejilla = TextBox(coordenadas_rejilla, "tamaño de la rejilla (,): ")
        self.boton_vecindad = RadioButtons(coordenadas_vecinidad, ('X','*'), activecolor='blue')
        boton_pesos = Button(coordenadas_pesos, "Inicializar pesos")
        boton_entrenar_SOM = Button(coordenadas_entrenar_SOM, "SOM")
        self.text_box_dimensiones=TextBox(self.coordenadas_dimensiones,"")
        self.text_box_dimensiones.set_val("Dim:")
        
        
        

        self.boton_archivo.on_clicked(self.seleccionar_archivo)
        self.text_box_epocas.on_submit(self.validar_epocas)
        self.text_box_rejilla.on_submit(self.validar_rejilla)
        self.boton_vecindad.on_clicked(self.indice)
        boton_pesos.on_clicked(self.inicializar_pesos)
        boton_entrenar_SOM.on_clicked(self.entrenar_som)

        """
        self.text_box_rango.on_submit(self.validar_rango)
        self.text_box_error_minimo_deseado.on_submit(self.validar_error_minimo_deseado)
        
        
        self.boton_clase.on_clicked(self.cambio_clase)
        boton_quick.on_clicked(self.entrenar_quick)
        boton_entrenar_mlp.on_clicked(self.entrenar_mlp)
        self.fig.canvas.mpl_connect('button_press_event', self.__onclick)
        """
        
        
        plt.show()

    def indice(self,label):
        
        if(label=='X'):
            self.tipo_rejilla=0
        else:
            self.tipo_rejilla=1
    def seleccionar_archivo(self,event):
        Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
        filename = askopenfilename(title='Seleccione archivo de dataset')
        self.archivo_seleccionado=filename
        print("archivo selecciondao")
        print(self.archivo_seleccionado)
    def validar_rejilla(self,expression):
        value = 0
        try:
            value = eval(expression)
        except (SyntaxError, NameError):
            if expression:
                value = [10,10]
                self.text_box_rejilla.set_val(value)

        if type(value) != tuple:
            value = [10,10]
        self.neuronas_rejijlla = [x for x in value]
    
    def inicializar_pesos(self, event):
        if self.epocas_maximas>0 and self.archivo_seleccionado is not None and  not self.som_entrenado and self.neuronas_rejijlla is not None:
            self.som=SOM(tipo_vecindario=self.get_tipo_vecindario(),
            tamanio_malla=self.neuronas_rejijlla, 
            max_epocas=self.epocas_maximas,
            filename=self.archivo_seleccionado,
            ventana=self
            )           
            self.pesos_inicializados = True
            print("Se inicializaron los pesos")

    def get_tipo_vecindario(self):
        if self.tipo_rejilla==0:
            return 'Cruz'
        return 'Estrella'

    def validar_epocas(self, expression):
        try:
            self.epocas_maximas =int(expression)
        except ValueError:
            self.epocas_maximas = 50
        finally:
            self.text_box_epocas.set_val(self.epocas_maximas)

    def entrenar_som(self, event):
        max_epochs_initialized = self.epocas_maximas != 0
        
        
        if not self.som_entrenado and self.pesos_inicializados and max_epochs_initialized:
            self.som.entrenar()
            self.som_entrenado=True
            print("termino de entrenar")
            self.som.graficar()
    def plot_colorbar(self,x_x,y_y,c_c):
        self.grafica.scatter(x=x_x, y=y_y, c=c_c, cmap="viridis",marker='s',s=120)
        plt.show()
    
if __name__ == '__main__':
    Ventana()

    #20 = 40
    #10