@echo -off

set ALLBIOSROM  NHHJTFB26.711
set BIOSEXE  TFB26711.efi
                                   
if '%1' == '' then     									
goto flash  												
endif   													
set ALLBIOSROM %1     										
:flash  													
# Check the file exist or not.  							
if not exist %ALLBIOSROM% then   							
 echo The %ALLBIOSROM% doesn't exist.    					
 goto end   												
endif   													
fpt -f %ALLBIOSROM% -a 0 -l 0x500000 -y
%BIOSEXE% 		
:end    													
@echo -on
