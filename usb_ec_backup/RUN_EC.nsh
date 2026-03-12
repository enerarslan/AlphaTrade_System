@echo -off

if exist fs0:\EC\EcFlash.nsh then
  fs0:
  cd EC
  EcFlash.nsh
  goto end
endif

if exist fs1:\EC\EcFlash.nsh then
  fs1:
  cd EC
  EcFlash.nsh
  goto end
endif

if exist fs2:\EC\EcFlash.nsh then
  fs2:
  cd EC
  EcFlash.nsh
  goto end
endif

if exist fs3:\EC\EcFlash.nsh then
  fs3:
  cd EC
  EcFlash.nsh
  goto end
endif

echo EC flash script not found on fs0/fs1/fs2/fs3.

:end
@echo -on
