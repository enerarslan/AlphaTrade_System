@echo -off
set BIOSNAME NH5XHH
set BIOSVER  11
set BIOSEXE  %BIOSNAME%%BIOSVER%.efi
#Tool Name  OPTION ME(TXE) VERSION     [OPTION2]-C CHECK -Y ALWAYS MESET		    	    
CkMEver.efi -mecmp 0015.0000.0030.1716 -Y      									
if %Lasterror% == 0 then      												
 goto flash       		    												
endif     																	
if %Lasterror% == 1 then      												
 goto meset       		    												
endif 																		
if %Lasterror% == 2 then      												
 goto meset       		    												
endif 																		
if %Lasterror% == 3 then      												
 goto lerror       		    												
endif 																		
:flash  			    														
%BIOSEXE%       						
 goto end       		    													
:meset    																	
 MeSet.efi -t5  			    											    
 goto end     																
:lerror   																	
@echo Tool Report Get Me Version Error										
 goto end     																
:end    			    														
@echo -on         															
