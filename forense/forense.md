 # metadatos
	 exiftool 

  analisis de imagenes
	 sherloq

  forense movil
	 openbackupextractor
	 iLEAPP ios logs eventos

   forense windows
	   PyShadow ver copias en shadow, borrarlas crearlas etc
	   LogonTracer
	   RegRippy

   forense red
	   wireshark

  reverse engineering 
	  ghidra

desde ek ftk imager
files importantes de system32

config/default
config/SAM
config/SECURITY
config/software
config/system

tomar nota de zona horaria

https://github.com/warewolf/regripper/tree/master/plugins link de plugins y de la herramienta regripper

abrimos regripper: rip.exe -r "direccion del archivo que queremos analizar" -p timezone 

ese plugin extrae la zona horaria del archivo

hacemos una captura y la pegamos en obsidian

ultimo apagado de SO cambiamos el plugin: shutdown

arquitectura de sistema: processor_architecture

nombre de pc: compname

si ejecutamos el plugin winver a el archivo software, vemos la version del SO

con el ftk imager buscamos NTUSER.DAT en la carpeta de usuario que sea dentro de documents and settings dentro de root

lo analizamos con regripper con el plugin recentdocs para ver los documentos abiertos por el usuario

buscamos el historial de navegación en ese mismo archivo  con el plugin typedurls

buscamos los correos electronicos que no han sido leidos con el plugin unreadmail

con el plugin printers vemos las cosas que se han imprimido

ejecutamos regripper con el archivo system para ver el timezone para ver la imagen forense en autopsy

una vez todo configurado y la imagen abierta en autopsy, ejecutamos los modulos para ver la actividad mas reciente

en SAM se guardan todos los usuarios



