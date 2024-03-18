import customtkinter as ctk
from tkinter import filedialog
from analysis import Analysis
import os
import sys

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Analizador de encuestas")
        self.geometry("800x400")
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=1)
        
        # Configuración del título
        titulo_fuente = ("Arial", 14, "bold")  # Define la fuente, tamaño y estilo
        
        # Multiple option frame
        self.multiple_option_frame = ctk.CTkFrame(self)
        self.multiple_option_frame.grid(row=1, column=0, pady=5, columnspan=3, sticky="ewns")
        self.multiple_option_label = ctk.CTkLabel(self.multiple_option_frame,
                                    text="ANÁLISIS ESTADÍSTICO",
                                    anchor="e",
                                    justify="left",
                                    font=titulo_fuente)
        self.multiple_option_label.grid(row=0,column=0,padx=(10,0), sticky="w")
        
        # Open option frame
        self.open_option_frame = ctk.CTkFrame(self)
        self.open_option_frame.grid(row=2, column=0, pady=5, columnspan=3, sticky="ewns")
        self.open_option_label = ctk.CTkLabel(self.open_option_frame,
                                    text="ANÁLISIS SENTIMENTAL",
                                    anchor="e",
                                    justify="left",
                                    font=titulo_fuente)
        self.open_option_label.grid(row=0,column=0,padx=(10,0),  sticky="w")
        
        # Temporal frame
        self.temporal_option_frame = ctk.CTkFrame(self)
        self.temporal_option_frame.grid(row=3, column=0, pady=5, columnspan=3, sticky="ewns")
        self.temporal_option_label = ctk.CTkLabel(self.temporal_option_frame,
                                    text="ANÁLISIS TEMPORAL",
                                    anchor="e",
                                    justify="left",
                                    font=titulo_fuente)
        self.temporal_option_label.grid(row=0,column=0,padx=(10,0), sticky="w")
        
        self.begin()
        
    def begin(self):
        self.filename = ctk.CTkLabel(self,
                                    text="Archivo:",
                                    anchor="e",
                                    justify="left",
                                    wraplength=150)
        self.filename.grid(row=0,column=0,
                           padx=(10,0), pady=20, sticky="w")
        
        
        # --- B O T O N E S ---
        self.searchfileButton = ctk.CTkButton(self,
                                              text="Elegir Archivo",
                                              command=self.search_file)
        self.searchfileButton.grid(row=0, column=1,padx=30, sticky="ew")
        
        self.loadfileButton = ctk.CTkButton(self,
                                              text="Cargar",
                                              command=self.load_file)
        self.loadfileButton.grid(row=0, column=2,padx=30, sticky="ew")
        self.loadfileButton.configure(state="disabled")
        
        # --- Option Menu ---
        self.opmenu = ctk.CTkOptionMenu(self.multiple_option_frame,
                                        values=["Sin opción múltiple"],
                                        dynamic_resizing=False,
                                        state="disabled")
        self.opmenu.grid(row=1,column=0,padx=30)
        self.opmenuButton = ctk.CTkButton(self.multiple_option_frame,
                                          text="Analizar",
                                          command=self.analizer_opmul,)
        self.opmenuButton.grid(row=1, column=1,padx=30,pady=30)
        self.opmenuButton.configure(state="disabled")
        self.opmenu2 = ctk.CTkOptionMenu(self.multiple_option_frame,
                                        values=["Sin opción múltiple"],
                                        dynamic_resizing=False,
                                        state="disabled")
        self.opmenu2.grid(row=1,column=2,padx=30,pady=30)
        self.opcorrButton = ctk.CTkButton(self.multiple_option_frame,
                                          text="Correlación",
                                          command=self.analizer_corr,)
        self.opcorrButton.grid(row=1, column=3,padx=30,pady=30)
        self.opcorrButton.configure(state="disabled")
        
        self.aboption = ctk.CTkOptionMenu(self.open_option_frame,
                                        values=["Nube de palabras", "Sentimientos"],
                                        dynamic_resizing=False,
                                        state="disabled")
        self.aboption.grid(row=1,column=0,padx=30,pady=30)
        self.abmenu = ctk.CTkOptionMenu(self.open_option_frame,
                                        values=["Sin respuesta abierta"],
                                        dynamic_resizing=False,
                                        state="disabled")
        self.abmenu.grid(row=1,column=1,padx=30,pady=30)
        self.abmenuButton = ctk.CTkButton(self.open_option_frame,
                                          text="Analizar",
                                          command=self.analizer_opab,)
        self.abmenuButton.grid(row=1, column=2,padx=30,pady=30)
        self.abmenuButton.configure(state="disabled")
        
        self.tempoption = ctk.CTkOptionMenu(self.temporal_option_frame,
                                          values=["Respuestas por hora", "Respuestas por día", "Emociones por hora"],
                                          dynamic_resizing=False,
                                          state="disabled",
                                          command=self.check_temp_value)
        self.tempoption.grid(row=1,column=0,padx=30,pady=30)
        self.tempmenu = ctk.CTkOptionMenu(self.temporal_option_frame,
                                          values=["Sin respuestas abiertas"],
                                          dynamic_resizing=False,
                                          state="disabled")
        self.tempmenu.grid(row=1,column=1,padx=30,pady=30)
        self.tempoptionButton = ctk.CTkButton(self.temporal_option_frame,
                                          text="Analizar",
                                          command=self.analizer_temp,)
        self.tempoptionButton.grid(row=1, column=2,padx=30,pady=30)
        self.tempoptionButton.configure(state="disabled")
    
    def analizer_corr(self):
        self.analizer.get_correlation(self.opmenu.get(), self.opmenu2.get())
    
    def analizer_opmul(self):
        print(self.opmenu.get())
        self.analizer.opmul_graph(self.opmenu.get())
        
    def analizer_opab(self):
        if self.aboption.get() == "Nube de palabras":
            self.analizer.wordcloud(self.abmenu.get())
        elif self.aboption.get() == "Sentimientos":
            self.analizer.get_emotions(self.abmenu.get())
        #self.analizer.opmul_graph(self.abmenu.get())
    
    def analizer_temp(self):
        if self.tempoption.get() == "Respuestas por hora":
            self.analizer.per_hour()
        elif self.tempoption.get() == "Respuestas por día":
            self.analizer.per_day()
        elif self.tempoption.get() == "Emociones por hora":
            self.analizer.emotions_per_hour(column=self.tempmenu.get())
    
    def check_temp_value(self, choice):
        if choice == "Emociones por hora":
            self.tempmenu.configure(state="normal")
        else:
            self.tempmenu.configure(state="disabled")
        
    def load_columns(self):
        if self.analizer.op_cols:
            self.opmenu.configure(values=self.analizer.op_cols, state="normal")
            self.opmenu.set(self.analizer.op_cols[0])
            self.opmenu2.configure(values=self.analizer.op_cols, state="normal")
            self.opmenu2.set(self.analizer.op_cols[0])
            self.opmenuButton.configure(state="normal")
            self.opcorrButton.configure(state="normal")
        else:
            self.opmenu.configure(values=self.analizer.op_cols, state="disabled")
            self.opmenuButton.configure(state="disabled")
            
        if self.analizer.ab_cols:
            self.abmenu.configure(values=self.analizer.ab_cols, state="normal")
            self.aboption.configure(state="normal")
            self.abmenuButton.configure(state="normal")
            self.abmenu.set(self.analizer.ab_cols[0])
        else:
            self.abmenu.configure(values=self.analizer.ab_cols, state="disabled")
            self.abmenuButton.configure(state="disabled")
            
        if self.analizer.tem_cols:
            self.tempoption.configure(state="normal")
            self.tempoptionButton.configure(state="normal")
            if self.analizer.ab_cols:
                self.tempmenu.configure(values=self.analizer.ab_cols)
                self.tempmenu.set(self.analizer.ab_cols[0])
        else:
            self.tempoption.configure(state="disabled")
            self.abmenu.configure(values=self.analizer.ab_cols, state="disabled")
            self.abmenuButton.configure(state="disabled")
            
            
    def search_file(self):
        filepath = filedialog.askopenfilename(
            title= "Selecciona un archivo csv",
            filetypes=[("Archivos csv", "*.csv*")]
        )
        
        if not filepath:
            print("No se selecionó ningún archivo")
            return
        
        self.filepath = filepath
        name = self.filepath.split("/")[-1]
        self.filename.configure(text=f"Archivo: {name}")
        self.loadfileButton.configure(state="normal")
        
    def load_file(self):
        self.analizer = Analysis(self.filepath)
        if self.analizer.load_file():
            self.load_columns()
            print("Archivo cargado exitosamente")
        else:
            print("No se pudo cargar")
        
if __name__ == "__main__":
    app = App()
    ctk.set_appearance_mode("light")
    app.mainloop()