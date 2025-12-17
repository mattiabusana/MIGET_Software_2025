      PROGRAM SHORT
C 
C        This is a data entry program for the steady state multiple inert
C        gas elimination technique.  It creates a raw data file & a file
C        for VQDIST, the program which estimates ventilation and blood  
C        flow distributions.  Entry of data is format-free.
C        Measured VO2 and VCO2 are entered as well.
C        Data files are named from the keyboard.
C
C        Updated by PDW  on DECEMBER 14, 1990
C 
      CHARACTER IQ,IADDON,IOK,IFLAG,NAMRAW                         
      DIMENSION S(10),PC(10),PA(10),PE(10),PV(10),R(10),E(10)           
      DIMENSION WT(10),ERRA(10),ERRE(10),ERRV(10),ERRPC(10),RMV(10) 
      DIMENSION EMV(10),VARQT(10),QTCALC(10),PAC(10),PVC(10),PEC(10)    
      DIMENSION NAMRAW(60),GA(10),GE(10),GV(10),PCC(10)
      DIMENSION HUMSLO(6),DOGSLO(6),HORSLO(6),PCFACT(6),PCBODY(6)
      DATA GA/10*1.0/,GE/10*1.0/,GV/10*1.0/
      DATA HUMSLO/2950.,1374.,2025.,3016.,4066.,805./
      DATA DOGSLO/2263.,2396.,2909.,4435.,3281.,1391./
      DATA HORSLO/1251.,2466.,3262.,2696.,3877.,2412./
      FILE3='FILE3.txt'
      FILE7='FILE7.txt'
      FILE9='FILE9.txt'
      WRITE(*,1)
  1   FORMAT(' WRITE OUTPUT TO DISK (0) OR TO PRINTER (1) ?')
      READ(*,*)IOUT
      IF(IOUT.EQ.0) WRITE(*,2)
  2   FORMAT(' ENTER  DISK:NAME.EXT  OF OUTPUT DATA FILE'/)
      IF(IOUT.EQ.0) OPEN(3,FILE=FILE3,STATUS='REPLACE')
      IF(IOUT.EQ.1) OPEN(3,FILE='PRN')
      WRITE(*,3)
  3   FORMAT(' ARE DATA FOR MAN(1), DOG(2), OR HORSE(3) ?')
      READ(*,*)ISPEC
      WRITE(*,4)
  4   FORMAT(' WANT TO CORRECT FOR BODY/BATH TEMP DIFF ? 1=YES, 0=NO')
      READ(*,*)IBATH
      WRITE(*,1234)
 1234 FORMAT(' WANT TO CORRECT FOR EXPIRED ACETONE LOSS ? (YES=1/NO=0)')
      READ(*,*)IACET
      VO2=300.0                                                          
      VCO2=240.0                                    
      TOL=99000.0                                                       
      TEMP=37.0                                        
      PICO2=0.0                                                        
      Y1=30.0                                         
      Y2=45.0                                                            
      X3=40.0                                           
      X4=55.0                                                          
      SO2=0.003                                            
      APCO2=30.0                                                        
      BPCO2=60.0                                     
      LCASE=1                                                          
  10  WRITE(*,20)                                          
  20  FORMAT(' THIS VERSION OF SHORT WILL WEIGHT EITHER RETENTIONS OR    
     + EXCRETIONS',/,
     +' ENTER 0 FOR R (USUAL WAY) OR 1 FOR E, BUT NOT <CR>')      
      READ(*,*) IRORE
C************
C     ASK WHETHER ARTERIAL OR PERIPHERAL VENOUS BLOOD IS TO BE USED
C************
      WRITE(*,25)
  25  FORMAT(' USING ARTERIAL (1) OR PERIPHERAL VENOUS (2) BLOOD ?'/)
      READ(*,*)IPAPV
      WRITE (*,30)                                                  
  30  FORMAT(' ARE DATA FROM HP (0) OR FROM BECKMAN (1) GC ?',/,
     1'  <CR> NOT ALLOWED')  
      READ(*,*)IGC
      IF(IRORE.NE.0 .AND. IRORE.NE.1) GO TO 10
  40  FORMAT(' ENTER LABEL FOR DATA SET, UP TO 60 CHRS')
  50  FORMAT(60A1)                                     
      WRITE (*,60)                                                     
  60  FORMAT(' ARE RAW DATA ALREADY ON FILE ? (Y/N)')   
      READ (*,70) IQ                                                   
  70  FORMAT(A1)
      IF(IQ.EQ.'N') WRITE(*,40)
      IF(IQ.EQ.'N') READ(*,50)(NAMRAW(I),I=1,60)                                   
      IF(IQ.NE.'Y') GO TO 130                                            
      LCASE=3
      WRITE(*,80)
  80  FORMAT(' ENTER    DISK:FILE.EXT    FOR OLD RAW DATA FILE'/)
      OPEN(8,FILE=' ')
      READ(8,50)(NAMRAW(I),I=1,60)
      WRITE (*,90) (NAMRAW(I),I=1,60)                  
  90  FORMAT(' DO YOU WANT TO ADD ON TO FILE: ',/,60A1'?? (Y/N)')           
      READ (*,70) IADDON                           
      READ(8,190) NRUNS,PBSEA,ELECT,TBATH,RER,SO2
C**************          
C     IF(IADDON.NE.'Y') GO TO 150
      IF(IADDON.NE.'Y') GO TO 130                 
C**************
      LCASE=2                                                           
 100  WRITE (*,110) NRUNS                           
 110  FORMAT(I4' DATA SETS ARE ON FILE (IF INCORRECT EDIT # IN FILE|)'  
     +/' ENTER # OF SETS TO ADD')   
      READ (*,190) NADDON
      NTOT=NRUNS+NADDON                               
      WRITE (*,120) NRUNS,NADDON,NTOT                                   
 120  FORMAT(I3' SETS IN OLD FILE +'I3' NEW SETS ='I3' TOTAL, OK? 
     + (Y/N)')                                                          
      READ (*,70) IOK                               
      IF(IOK.EQ.'N') GO TO 100                                          
 130  CONTINUE
      IF(LCASE.EQ.3) NTOT=NRUNS
      WRITE(*,140)
 140  FORMAT(' ENTER     DISK:FILE.EXT     FOR NEW RAW DATA FILE'/)
      OPEN(7,FILE=FILE7,STATUS='REPLACE')
      WRITE(7,50)(NAMRAW(I),I=1,60)
 150  CONTINUE
      WRITE(*,160)
 160  FORMAT(' ENTER     DISK:FILE.EXT     FOR VQDIST INPUT FILE'/)
      OPEN(9,FILE=FILE9,STATUS='REPLACE')                              
      WRITE(3,170)(NAMRAW(I),I=1,60)                
 170  FORMAT(' LABEL FOR DATA SET  =  ',60A1/)
      WRITE(9,50)(NAMRAW(I),I=1,60)
      IF(IQ.EQ.'Y') GO TO 200                     
      WRITE (*,180)                                                    
 180  FORMAT(' ENTER # OF DATA SETS, PB SEA LEVEL,',/,
     1' BLOOD GAS ELECTRODE TEMP, & INERT GAS BATH TEMP ',/,
     2' FRACTIONAL CV OF GASES OTHER THAN SF6',/,
     3' AND O2 SOLUBILITY, ML/100 ML/TORR'/)
C
C     RER, THE (FRACTIONAL) MEASURED CV FOR GASES OTHER THAN SF6,
C     IS NORMALLY 0.03, AND IS TWICE THIS FOR SF6.
C     HOWEVER, YOU CAN ENTER ANY VALUE THAT YOU WISH
C
      READ(*,*)NRUNS,PBSEA,ELECT,TBATH,RER,SO2
      WRITE(7,190) NRUNS,PBSEA,ELECT,TBATH,RER,SO2          
 190  FORMAT(I4,F7.1,2F6.1,F8.2,F8.4)
C************
C 200  IF(LCASE.EQ.2) GO TO 220
 200  IF(LCASE.EQ.2.OR.LCASE.EQ.3) GO TO 220 
C************                                         
      WRITE(9,190)NRUNS,PBSEA,ELECT,TBATH,RER,SO2
      WRITE(3,210)NRUNS,PBSEA,ELECT,TBATH,RER,SO2
 210  FORMAT(' NUMBER OF RUNS =',I5,';    SEA LEVEL PB =',F7.1,/,
     1' ELECTRODE TEMP =',F5.1,';   H2O BATH TEMP =',F7.1,/,
     2' FRACTIONAL MEASUREMENT CV (EXCEPT SF6) =',F8.2,/,
     3' AND O2 SOLUBILITY, ML/100 ML/Torr      = ',F9.4/)
      GO TO 230                                                         
 220  WRITE(7,190)NTOT,PBSEA,ELECT,TBATH,RER,SO2
      WRITE(9,190)NTOT,PBSEA,ELECT,TBATH,RER,SO2                                
      WRITE(3,210)NTOT,PBSEA,ELECT,TBATH,RER,SO2                  
 230  IF(LCASE.EQ.1) GO TO 270                                          
      READ(8,240) NGASES,NVAQS,ZZ,VQLO,VQHI       
      READ(8,250) (PC(I),I=1,NGASES)                                
 240  FORMAT(2I4,1X,F5.1,2X,F5.3,2X,F5.1)
 250  FORMAT(1X,6(1PE12.3))          
 260  FORMAT(1X,1PE12.3,5(1PE12.3))                                     
C*************
      IF(LCASE-2) 310,310,310
C      IF(LCASE-2) 310,310,320                     
C*************
 270  WRITE (*,280)
 280  FORMAT(' ENTER NGASES,NVAQS,Z,VQLO & VQHI')
      READ (*,*) NGASES,NVAQS,ZZ,VQLO,VQHI
      WRITE (*,240) NGASES,NVAQS,ZZ,VQLO,VQHI                          
      WRITE (*,290) NGASES                                
 290  FORMAT(' ENTER',I3,'   PARTITION COEFFICIENTS, BATH TEMP')    
      READ (*,*) (PC(I),I=1,NGASES)
      WRITE (*,250) (PC(I),I=1,NGASES)             
      WRITE (*,300)                                                    
 300  FORMAT(' OK OR NOT ? (Y/N)')                       
      READ (*,70) IOK                                                  
      IF(IOK.EQ.'N') GO TO 270                          
 310  WRITE(7,240) NGASES,NVAQS,ZZ,VQLO,VQHI                            
      WRITE(7,250) (PC(I),I=1,NGASES)                    
 320  WRITE(3,330)NGASES,NVAQS,ZZ,VQLO,VQHI                             
 330  FORMAT(' # OF GASES=',I2,';   # OF COMPTS=',I3,';   Z=',F7.2, 
     1';  VQLO=',F6.3,';  VQHI=',F6.1)                                  
      WRITE(3,340)(PC(I),I=1,NGASES)                                           
 340  FORMAT(' PARTITION COEFFICIENTS, BATH TEMP =',
     1F7.5,2F7.3,F6.2,F6.1,F8.1)
      WRITE(3,350)                               
 350  FORMAT(80('-'))                                                  
      INDEX=0                                             
      LOGAS=1                                                           
      IF(PC(1).LT.0.01) LOGAS=2                     
      L1=1                                                             
      IAGAIN=0                                       
 360  DO 1350 LM=L1,NRUNS                                               
 370  CONTINUE           
      DO 380 I=1,NGASES                                                 
      E(I)=0.0                                         
      R(I)=0.0                                                          
      EMV(I)=0.0                                      
      RMV(I)=0.0                                                        
      WT(I)=0.0                                       
      QTCALC(I)=0.0                                                     
 380  CONTINUE                                         
      INDEX=INDEX+1
C************
C      IF(IQ.EQ.'Y') GO TO 610
      IF(IQ.EQ.'Y'.AND.IAGAIN.EQ.0) GO TO 610                            
C************
      IF(IAGAIN.EQ.1) GO TO 420
 390  IF(IPAPV.EQ.1) WRITE (*,400) LM                                    
 400  FORMAT(//9X'SET #'I3//3X'ENTER ARTERIAL PEAKS (in mm)')
      IF(IPAPV.EQ.2) WRITE (*,401) LM                                    
 401  FORMAT(//9X'SET #'I3//3X'ENTER PERIPHERAL VENOUS PEAKS (in mm)')
      READ (*,*) (PA(I),I=1,NGASES)
      IF(IGC.EQ.1.AND.IPAPV.EQ.1) WRITE (*,410) NGASES
 410  FORMAT(' ENTER',I3'  ARTERIAL SF6 DILUTION/FID GAINS (decimal)') 
      IF(IGC.EQ.1.AND.IPAPV.EQ.2) WRITE (*,411) NGASES
 411  FORMAT(' ENTER',I3'  PERIPHERAL VENOUS SF6 DILUTION/FID GAINS') 
      IF(IGC.EQ.1) READ (*,*) (GA(I),I=1,NGASES)
 420  WRITE (*,1200)(PA(I),I=1,NGASES)                       
      WRITE (*,1210)(GA(I),I=1,NGASES)                                  
      WRITE (*,300)                                               
      READ (*,70) IOK                                                  
      IF(IOK.EQ.'N') GO TO 390                                   
      IF(IAGAIN.EQ.1) GO TO 460                                        
 430  WRITE (*,440)                                                           
 440  FORMAT(3X,'ENTER EXPIRED PEAKS (decimal in mm)')                 
      READ (*,*) (PE(I),I=1,NGASES)
      IF(IGC.EQ.1) WRITE (*,450) NGASES                               
 450  FORMAT(' ENTER',I3'  EXPIRED SF6 DILUTION/FID GAINS (decimal)')
      IF(IGC.EQ.1) READ(*,*)(GE(I),I=1,NGASES)  
 460  WRITE (*,1200)(PE(I),I=1,NGASES)                                 
      WRITE (*,1210)(GE(I),I=1,NGASES)                             
      WRITE (*,300)                                         
      READ (*,70) IOK                                                 
      IF(IOK.EQ.'N') GO TO 430                                                
      IF(IAGAIN.EQ.1) GO TO 500                                       
 470  WRITE (*,480)                                                          
 480  FORMAT(3X,'ENTER VENOUS PEAKS (decimal in mm)')
      READ(*,*)(PV(I),I=1,NGASES)                  
      IF(IGC.EQ.1) WRITE (*,490) NGASES                               
 490  FORMAT(' ENTER',I3'  VENOUS SF6 DILUTION/FID GAINS (decimal)')
      IF(IGC.EQ.1) READ(*,*)(GV(I),I=1,NGASES) 
 500  WRITE (*,1200) (PV(I),I=1,NGASES)              
      WRITE (*,1210) (GV(I),I=1,NGASES)                                
      WRITE (*,300)                                                           
      READ (*,70) IOK                                                 
      IF(IOK.EQ.'N') GO TO 470                                                 
      IF(IAGAIN.EQ.1) GO TO 530                                       
 510  WRITE (*,520)                                                           
 520  FORMAT(' ENTER VE,QT,PB,TEMPB,TEMPR,VGA,VBA,VHA,VGV,VBV,VHV
     1 (FREE FORMAT)')
      READ(*,*)VE,QT,PB,TEMPB,TEMPR,VGA,VBA,VHA,VGV,VBV,VHV
 530  WRITE (*,535) VE,QT,PB,TEMPB,TEMPR,VGA,VBA,VHA,VGV,VBV,VHV
 535  FORMAT(F9.2,F6.2,2X,F5.1,2F6.1,6F7.2)
      WRITE (*,300)                                                           
      READ (*,70) IOK                                                 
      IF(IOK.EQ.'N') GO TO 510                                                 
      VEO=VE                                                           
      IF(IAGAIN.EQ.1) GO TO 600                                               
 540  IF(PV(1).EQ.0.0) GO TO 560                                       
      WRITE (*,550)                                                           
 550  FORMAT(' ENTER HB,HCRIT,PVO2,PVCO2,PHV,FIO2,FICO2,P50,PMAO2,PMACO2
     1,PHA,VO2 & VCO2',/,' FREE FORMAT')                            
      GO TO 590                                                        
 560  CONTINUE                                                                 
      WRITE (*,570)                                                   
 570  FORMAT(' ENTER HB,HCRIT,VO2,VCO2,TOL,FIO2,FICO2,P50,PMAO2,PMACO2,
     1PHA, FREE FORMAT')                                           
      WRITE (*,580)                                                           
 580  FORMAT(' TOL=20.0 will iterate mixed venous gases to match       
     + VO2 & VCO2'/' TOL=99000. will use Fick calculated mixed                
     + venous tensions'//)                                             
 590  CONTINUE
      READ(*,*)HB,HCRIT,PVO2,PVCO2,PHV,FIO2,FICO2,P50,PMAO2,PMACO2,PHA,
     1VO2DUM,VCO2DM
 600  WRITE (*,1230)HB,HCRIT,PVO2,PVCO2,PHV,FIO2,FICO2,P50,PMAO2,PMACO2,
     1PHA,VO2DUM,VCO2DM   
      WRITE (*,300)                                                    
      READ (*,70) IOK                                                         
      IF(IOK.EQ.'N') GO TO 540                                          
      GO TO 620                                                                
 610  CONTINUE                                                         
      READ(8,1200) (PA(I),I=1,NGASES)                                           
      READ(8,1210) (GA(I),I=1,NGASES)                                   
      READ(8,1200) (PE(I),I=1,NGASES)                                           
      READ(8,1210) (GE(I),I=1,NGASES)                                   
      READ(8,1200) (PV(I),I=1,NGASES)                                           
      READ(8,1210) (GV(I),I=1,NGASES)                                   
      READ(8,1220) VEO,QT,PB,TEMPB,TEMPR,VGA,VBA,VHA,VGV,VBV,VHV                
      READ(8,1230)HB,HCRIT,PVO2,PVCO2,PHV,FIO2,FICO2,P50,PMAO2,PMACO2,
     1PHA,VO2DUM,VCO2DM
 620  CONTINUE                                                                 
C                                                                      
C     ABILITY TO COPE WITH 1 FEWER GAS ON ANY RUN                              
C                                                                       
      NGSS=NGASES                                                              
      IGAS=0                                                            
      DO 630 I=1,NGASES                                                        
      PCC(I)=PC(I)                                                      
      IF(PA(I).GT.0.0) GO TO 630                                       
      IF(PE(I).GT.0.0) GO TO 630                                               
      IF(PV(I).GT.0.0) GO TO 630                                       
      IGAS=I                                                                   
 630  CONTINUE                                       
      IF(IGAS.GT.0) NGSS=NGASES-1                                              
      DP50=P50-26.8                                                    
      REXP=9.0578-2290.5/(273.15+TEMPR)                                        
      BEXP=9.0578-2290.5/(273.15+TEMPB)
      CEXP=9.0578-2290.5/(273.15+TBATH)
      SVPB=10.0**BEXP
      SVPBTH=10.0**CEXP
      IF(TEMPR.LT.TEMPB) SVPR=10.0**REXP
      IF(TEMPR.GE.TEMPB) SVPR=SVPB
      EXPFAC=(PB-SVPB)/(PBSEA-SVPB)                                            
      PIO2=(PB-SVPB)*FIO2
      PICO2=(PB-SVPB)*FICO2                                              
      X5=PB-SVPB-PIO2                                                          
      TEMP=TEMPB                                                      
      VC=0.01*HCRIT                                                            
      TBODY=TEMPB-(ELECT-37.0)                                         
      IF(PHV.GT.6.5.AND.PHV.LT.8.0) CALL TEMPCR(PHV,PVCO2,PVO2,VC,             
     &TBODY,PHVBT,PVCOBT,PVO2BT,BX)                                    
      IF(PHA.GT.6.5.AND.PHA.LT.8.0) CALL TEMPCR(PHA,PMACO2,PMAO2,VC,           
     &TBODY,PHABT,PACOBT,PAO2BT,BX)                                    
      FACT=(PBSEA-SVPB)/100.0                                                  
      WRITE (*,640) INDEX                                             
 640  FORMAT(30X,'  RUN NUMBER  ',I3,/)                                        
      IF(TEMPB.GT.TEMPR) CORR=(273.0+TEMPB)*(PB-SVPR)/                 
     & ((273.0+TEMPR)*(PB-SVPB))
      IF(TBATH.GT.TEMPR) COR1=(273.0+TBATH)*(PB-SVPR)/
     & ((273.0+TEMPR)*(PB-SVPBTH))  
      IF(TEMPB.LE.TEMPR) CORR=(273.0+TEMPB)/(273.0+TEMPR)
      IF(TBATH.LE.TEMPR) COR1=(273.0+TBATH)/(273.0+TEMPR)
      VE=VEO*CORR
C***********
      VESTAR=VEO*COR1
C***********
      IF(PA(1).GT.0.0.AND.IPAPV.EQ.1) 
     1ERRA(1)=(2.0*RER)*(2.0*RER) + 0.212*0.212/(PA(1)*PA(1))
      IF(PA(1).GT.0.0.AND.IPAPV.EQ.2) 
     1ERRA(1)=(4.0*RER)*(4.0*RER) + 0.212*0.212/(PA(1)*PA(1))
      IF(PE(1).GT.0.0) ERRE(1)=(2.0*RER)*(2.0*RER) + 
     10.212*0.212/(PE(1)*PE(1))             
      IF(PV(1).GT.0.0) ERRV(1)=(2.0*RER)*(2.0*RER) + 
     10.212*0.212/(PV(1)*PV(1))    
      ERRPC(1) = (4.0*RER)*(4.0*RER)/2.0                               
      DO 650 I=LOGAS,NGASES                                            
      IF(IGAS.EQ.I) GO TO 650      
      IF(PA(I).GT.0.0.AND.IPAPV.EQ.1) 
     1ERRA(I)=RER*RER + 0.212*0.212/(PA(I)*PA(I))
      IF(PA(I).GT.0.0.AND.IPAPV.EQ.2) 
     1ERRA(I)=(2.0*RER)*(2.0*RER) + 0.212*0.212/(PA(I)*PA(I))
      IF(PE(I).GT.0.0) ERRE(I)=RER*RER + 0.212*0.212/(PE(I)*PE(I))
      IF(PV(I).GT.0.0) ERRV(I)=RER*RER + 0.212*0.212/(PV(I)*PV(I))
C             
C      PA(I) ETC MUST BE IN MM OF CHART PAPER, NOT INCHES|||||         
C      AND MUST BE PRIOR TO ANY GAIN OR DILUTION MULTIPLICATION   
C                                                            
      ERRPC(I) = (2.0*RER)*(2.0*RER)/2.0                                                 
 650  CONTINUE                                                         
      DO 670 I=1,NGASES
C
C     CORRECTIONS WHEN BODY AND INERT GAS BATH TEMP ARE DIFFERENT
C
      IF(IBATH.EQ.0) PCFACT(I)=1.0
      IF(IBATH.EQ.0) PCBODY(I)=PC(I)
      IF(IBATH.EQ.0) GO TO 659
      TT1=1.0/(TEMPB+273.0)
      TT2=1.0/(TBATH+273.0)
      IF(ISPEC.EQ.1) SLP=HUMSLO(I)
      IF(ISPEC.EQ.2) SLP=DOGSLO(I)
      IF(ISPEC.EQ.3) SLP=HORSLO(I)
      PCFACT(I)=EXP(SLP*(TT1-TT2))
      PCBODY(I)=PC(I)*PCFACT(I)
 659  CONTINUE 
      S(I)=PCBODY(I)/FACT                                                  
      PEC(I)=GE(I)*PE(I)*EXPFAC                                                
      IF(PA(I).EQ.0.0) GO TO 660
      IF(IPAPV.EQ.1) PAPV=1.0
      IF(IPAPV.EQ.2) PAPV=0.95
      PAC(I) = GA(I)*PA(I)*(1.0 + VHA/VBA + VGA/(VBA*PC(I)))/
     1(PCFACT(I)*PAPV)
 660  IF(PV(I).GT.0.0)
     1PVC(I) = GV(I)*PV(I)*(1.0 + VHV/VBV + VGV/(VBV*PC(I)))/PCFACT(I)
C
C     CORRECT FOR EXPIRED ACETONE LOSS BASED ON ETHER EXP/ART RATIO:
C     ACETONE "BOHR" VDVT MUST BE NO GREATER THAN THAT OF ETHER, THUS:
C
C     (Pa-PE)/Pa (acetone)  =  (Pa-PE)/Pa (ether)
C
C     OR
C    
C      PE (acetone)  =  Pa (acetone) * PE (ether) / Pa (ether)     
C
C     HENCE, REPLACE MEASURED PE (acetone) BY THE ABOVE IF THE ABOVE
C     IS THE LARGER NUMBER
C
      IF(IACET.EQ.0) GO TO 670
      IF(I.LT.NGASES) GO TO 670
      IF(PAC(NGASES-1).EQ.0.0.OR.PAC(NGASES).EQ.0.0) GO TO 670
      IF(PEC(NGASES-1).EQ.0.0.OR.PEC(NGASES).EQ.0.0) GO TO 670
      ETHER=PEC(NGASES-1)/PAC(NGASES-1)
 669  FORMAT(/' CORRECTING FOR EXPIRED ACETONE LOSS BY ',F7.3,/)
      IF(PV(NGASES).GT.0.0) ACEXP=0.5*ETHER*(PAC(NGASES)+PVC(NGASES))
      IF(PV(NGASES).EQ.0.0) ACEXP=    ETHER*PAC(NGASES)
      IF(PEC(NGASES).GE.ACEXP) GO TO 670
      ACEFAC= ACEXP/PEC(NGASES)
      WRITE(*,669)ACEFAC
      WRITE(3,669)ACEFAC
      PEC(NGASES)=ACEXP
 670  CONTINUE                                  
      IF(PA(1).EQ.0.0) GO TO 690                                               
      IF(PE(1).EQ.0.0) GO TO 710                                       
      IF(PV(1).EQ.0.0) GO TO 730                                               
      WRITE (*,680)                                                   
 680  FORMAT(/' PA, PE, AND PV ALL MEASURED IN THIS SET'/)                     
      GO TO 750                                                        
 690  WRITE (*,700)                                                           
 700  FORMAT(/' ONLY PE AND PV HAVE BEEN MEASURED - PA IS DERIVED'/)  
      GO TO 750                                                                
 710  WRITE (*,720)                                                   
 720  FORMAT(/' ONLY PA AND PV HAVE BEEN MEASURED - PE IS DERIVED'/)           
      GO TO 750                                                        
 730  IF(IPAPV.EQ.1) WRITE(*,740)              
 740  FORMAT(/' ONLY PA AND PE HAVE BEEN MEASURED - PV IS DERIVED'/)
      IF(IPAPV.EQ.2) WRITE(*,741)
 741  FORMAT(/' PERIPHERAL VENOUS AND PE ARE MEASURED - PV IS DERIVED'/
     1' ARTERIAL = PERIPHERAL VENOUS/0.95, AND COEFF VAR ARE DOUBLED'/)   
 750  CONTINUE                                                                 
      IF(NGASES.EQ.6) WRITE (*,760)                                   
 760  FORMAT(19X,'SF6',6X,'ETHANE',5X,'CYCLO',4X,'ENFLURANE',4X,               
     1'ETHER',5X,'ACETONE')                                            
      WRITE (*,770) (S(I),I=1,NGASES)                                         
      WRITE (*,780) (PC(I),I=1,NGASES)
      WRITE (*,781) (PCBODY(I),I=1,NGASES)                                
 770  FORMAT(' SOLUBILITY ', 8F11.5)                                         
 780  FORMAT(' PC, BATH T ', 8F11.5)
 781  FORMAT(' PC, BODY T ', 8F11.5)
      IF(PA(1).EQ.0.0) GO TO 940                                               
      IF(PE(1).EQ.0.0) GO TO 980                                        
      IF(PV(1).EQ.0.0) GO TO 1010
      VO2=VO2DUM
      VCO2=VCO2DM                                      
      SUMTOP=0.0                                                        
      SUMBOT=0.0                                                               
      DO 790, I=1,NGASES                                                
      IF(I.EQ.IGAS) GO TO 790                                                  
      R(I) = PAC(I)/PVC(I)                                              
      E(I) = PEC(I)/PVC(I)                                                     
      QTCALC(I) = VESTAR*E(I)/(PC(I)*(1.0-R(I)))                            
      V = ERRE(I) + ERRPC(I) + ERRV(I)/((1.0-R(I))**2) +                       
     1ERRA(I)*R(I)*R(I)/((1.0-R(I))**2)                                
      VARQT(I) = QTCALC(I)*QTCALC(I)*V                                         
      IF(QTCALC(I).LE.0.0) VARQT(I)=10.0**6                             
      IF (PC(I).GT.50.0) VARQT(I)=10.0**6                                      
      SUMTOP=SUMTOP+QTCALC(I)/VARQT(I)                                 
      SUMBOT=SUMBOT+1.0/VARQT(I)                                               
 790  CONTINUE                                                         
      QT=SUMTOP/SUMBOT                                                         
      DO 810, I=1,NGASES                                               
      IF(I.EQ.IGAS) GO TO 810                                                  
      IF(IRORE.NE.0) GO TO 800                                          
      X=R(I)*R(I)*(ERRA(I)+ERRV(I))                                            
      Y=((VE*E(I)/(QT*PCBODY(I)))**2)*(ERRE(I) + ERRPC(I) + ERRV(I))        
      Z=VE*E(I)*R(I)*ERRV(I)/(QT*PCBODY(I))                                       
      TOPT=(Y+Z)/(X + Y + 2.0*Z)                                               
      RMV(I)=TOPT*R(I) + (1.0-TOPT)*(1.0-VE*E(I)/(QT*PCBODY(I)))            
      EMV(I)=QT*PCBODY(I)*(1.0-RMV(I))/VE                    
      VRCE=Y - 2.0*TOPT*(Y+Z) + TOPT*TOPT*(X+Y+2.0*Z)                   
      WT(I) = 1.0/SQRT(VRCE)                                                   
      GO TO 810                                                         
 800  AA=QT*PCBODY(I)/VE                                                           
      ASQ=AA*AA                                                         
      RSQ=R(I)*R(I)                                                           
      ESQ=E(I)*E(I)                                                            
      BB=ESQ*ERRPC(I)+ASQ*RSQ*ERRA(I)                                   
      CC=ASQ*ERRV(I)                                                           
      TNUM=BB+CC*R(I)                                                   
      DENOM=BB+ESQ*ERRE(I)+CC                                                  
      TOPT=TNUM/DENOM                                                   
      VRCE=TOPT*TOPT*DENOM-2.0*TOPT*TNUM+BB+CC*RSQ                             
      EMV(I)=TOPT*E(I)+(1.0-TOPT)*AA*(1.0-R(I))                        
      RMV(I)=1.0-EMV(I)/AA                                                     
      WT(I)=1.0/SQRT(VRCE)                                              
 810  CONTINUE
      WRITE (*,819)(PCFACT(I),I=1,NGASES)
 819  FORMAT(' BODY/BATH PC',F10.5,7F11.5)
      IF(IRORE.NE.0) GO TO 870
      WRITE (*,820)(R(I),I=1,NGASES)                                   
 820  FORMAT(' MEASURED R ',8F11.5)                                   
      WRITE (*,830)(RMV(I),I=1,NGASES)                                 
 830  FORMAT(' MIN VAR R  ',8F11.5)                                    
      WRITE (*,840)(E(I),I=1,NGASES)                                   
 840  FORMAT(' MEASURED E ',8F11.5)                                     
      WRITE (*,850)(EMV(I),I=1,NGASES)                                 
 850  FORMAT(' ASSOCIATED E ',F9.5,7F11.5)                                  
      WRITE (*,860)(WT(I),I=1,NGASES)                                 
 860  FORMAT(' WEIGHTS (R)',8F11.1)                                  
      GO TO 910                                                        
 870  WRITE (*,840) (E(I),I=1,NGASES)                                 
      WRITE (*,880) (EMV(I),I=1,NGASES)                                
 880  FORMAT(' MIN VAR E  ',8F11.5)                                  
      WRITE (*,820) (R(I),I=1,NGASES)                                 
      WRITE (*,890) (RMV(I),I=1,NGASES)                                
 890  FORMAT(' ASSOCIATED R ',F9.5,7F11.5)                                   
      WRITE (*,900) (WT(I),I=1,NGASES)                                
 900  FORMAT(' WEIGHTS (E)',8F11.1)                                  
 910  WRITE (*,920) (QTCALC(I),I=1,NGASES)                            
 920  FORMAT(' PREDICTED QT ',F9.2,7F11.2)                                  
      WRITE (*,930) VE,QT                                             
 930  FORMAT(/' MINUTE VENTILATION, BTPS ='F8.2,'   MEAN FICK CARDIAC 
     1OUTPUT =',F7.2)                         
      GO TO 1050                                                        
 940  CONTINUE                                                         
      DO 960, I=1,NGASES                                               
      IF(I.EQ.IGAS) GO TO 960                                      
      PAC(I) = PVC(I) - VE*PEC(I)/(QT*PCBODY(I))                            
      R(I) = PAC(I)/PVC(I)                           
      RMV(I) = R(I)                                              
      E(I) = PEC(I)/PVC(I)                                              
      EMV(I) = E(I)                                                    
      IF(IRORE.NE.0) GO TO 950                                          
      V = ERRE(I) + ERRV(I) + ERRPC(I)                                 
      V = V*((VE*E(I)/(QT*PCBODY(I)))**2)                                   
      WT(I) = SQRT(1.0/V)                                               
      GO TO 960                                                         
 950  V=E(I)*E(I)*(ERRE(I)+ERRV(I))                                    
      WT(I)=SQRT(1.0/V)                                                 
 960  CONTINUE
      WRITE (*,819) (PCFACT(I),I=1,NGASES)
      WRITE (*,820) (R(I),I=1,NGASES)                                 
      WRITE (*,840) (E(I),I=1,NGASES)                                  
      IF(IRORE.EQ.0) WRITE (*,860) (WT(I),I=1,NGASES)                 
      IF(IRORE.NE.0) WRITE (*,900) (WT(I),I=1,NGASES)                 
      WRITE (*,970) VE,QT                                              
 970  FORMAT(/' MEASURED MINUTE VENTILATION =',F9.2,
     1'   AND MEASURED CARDIAC OUTPUT =',F8.2)                     
      GO TO 1050                                                       
 980  CONTINUE                                                          
      DO 1000, I=1,NGASES                                                
      IF(I.EQ.IGAS) GO TO 1000                                          
      R(I) = PAC(I)/PVC(I)                                             
      RMV(I) = R(I)                                                     
      PEC(I) = QT*PCBODY(I)*(PVC(I)-PAC(I))/VE                             
      E(I)=PEC(I)/PVC(I)                                               
      EMV(I)=E(I)                                                      
      IF(IRORE.NE.0) GO TO 990                                          
      V=R(I)*R(I)*(ERRA(I)+ERRV(I))                                     
      WT(I) = SQRT(1.0/V)                                              
      GO TO 1000                                      
 990  ASQ=(QT*PCBODY(I)/VE)**2                                              
      V=ASQ*(ERRPC(I)+R(I)*R(I)*(ERRA(I)+ERRV(I)+ERRPC(I)))             
      WT(I)=SQRT(1.0/V)                                                
1000  CONTINUE
      WRITE (*,819) (PCFACT(I),I=1,NGASES)
      WRITE (*,820) (R(I),I=1,NGASES)                                 
      WRITE (*,840) (E(I),I=1,NGASES)                                  
      IF(IRORE.EQ.0) WRITE (*,860) (WT(I),I=1,NGASES)                 
      IF(IRORE.NE.0) WRITE (*,900) (WT(I),I=1,NGASES)                  
      WRITE (*,970) VE,QT                                              
      GO TO 1050                                                        
1010  CONTINUE                                                          
      TOL=PHV                                                          
      VO2=PVO2                                                          
      VCO2=PVCO2                                                       
      IF(TOL.GT.50.0) GO TO 1020                                       
      TOL=20.0                                                          
      PVO2=40.0                                                         
      PVCO2=45.0                                                        
1020  CONTINUE                                        
      DO 1040 I=1,NGASES                                                
      IF(I.EQ.IGAS) GO TO 1040                                         
      PVC(I) = PAC(I)+ VE*PEC(I)/(QT*PCBODY(I))                             
      R(I)=PAC(I)/PVC(I)                                                
      RMV(I)=R(I)                                                       
      E(I)=PEC(I)/PVC(I)                                               
      EMV(I)=E(I)                                                       
      IF(IRORE.NE.0) GO TO 1030                                         
      X=QT*PCBODY(I)*PAC(I)                            
      Y=VE*PEC(I)                                                       
      W = X*Y/((X+Y)**2)                                               
      V=ERRA(I) + ERRE(I) + ERRPC(I)                                    
      V = V*W*W                                                         
      WT(I) = SQRT(1.0/V)                                               
      GO TO 1040                                    
1030  ESQ=E(I)*E(I)                                                     
      RSQ=R(I)*R(I)                                                     
      V=ESQ*ERRPC(I)+ESQ*RSQ*(ERRA(I)+ERRPC(I)+ERRE(I))                 
      WT(I)=SQRT(1.0/V)                                                
1040  CONTINUE
      WRITE (*,819) (PCFACT(I),I=1,NGASES)
      WRITE (*,820) (R(I),I=1,NGASES)                                 
      WRITE (*,840) (E(I),I=1,NGASES)                                  
      IF(IRORE.EQ.0) WRITE (*,860) (WT(I),I=1,NGASES)                  
      IF(IRORE.NE.0) WRITE (*,900) (WT(I),I=1,NGASES)              
      WRITE (*,970) VE,QT                                              
1050  CONTINUE  
C
C      NOW WRITE DATA TO DISK FOR THE LONG PROGRAM
C
C*************
C      IF(IQ.EQ.'Y') GO TO 1070                                          
C*************
      WRITE (*,1060)                                                   
1060  FORMAT(' IS THIS SET OK OR NOT ? (Y/N)')                          
      READ (*,70) IFLAG                                                
      IF(IFLAG.NE.'N') GO TO 1070                                       
      INDEX=INDEX-1                                                     
      IAGAIN=1
      VE=VEO                                                         
      GO TO 370                                                        
1070  CONTINUE                                                          
      IAGAIN=0                                                          
      WRITE(3,640)INDEX
      WRITE(3,1075)
1075  FORMAT(22X,'GAS 1',5X,'GAS 2',5X,'GAS 3',5X,'GAS 4',5X,'GAS 5',
     15X,'GAS 6'/)
      WRITE(3,1080)(PA(I),I=1,NGASES)                                    
1080  FORMAT(' ARTERIAL PEAKS  ',10F10.1)                               
      WRITE(3,1090)(GA(I),I=1,NGASES)                                    
1090  FORMAT(' AND GAIN FACTORS',10F10.1)                               
      WRITE(3,1100)(PE(I),I=1,NGASES)                                    
1100  FORMAT(' EXPIRED PEAKS   ',10F10.1)                               
      WRITE(3,1090)(GE(I),I=1,NGASES)                                    
      WRITE(3,1110)(PV(I),I=1,NGASES)                                    
1110  FORMAT(' VENOUS PEAKS    ',10F10.1)                               
      WRITE(3,1090)(GV(I),I=1,NGASES)
      WRITE(3,1119)(PCFACT(I),I=1,NGASES)
1119  FORMAT(' BODY/BATH PC  ',10F10.4)
      WRITE(3,1120)VE,QT,PB,SVPR,SVPB,TEMPB,TEMPR,VGA,VBA,VHA,VGV,VBV,
     1VHV
1120  FORMAT(
     1' VE  =',F7.2,'; QT   =',F7.2,'; PB   =',F5.1,/,
     2' SVPR=',F7.1,'; SVPB =',F7.1,'; TMPB =',F5.1,'; TMPR =',F6.1,';'/
     3' VGa =',F7.2,'; VBa  =',F7.2,'; VHa  =',F5.2,'; VGv  =',F6.2,      
     4';  VBv =',F5.2,'; VHv=',F4.2)                   
      IF(PV(1).EQ.0.0) GO TO 1140                                       
      WRITE(3,1130)HB,HCRIT,PVO2BT,PVCOBT,PHVBT,P50,PIO2,PICO2,PAO2BT,
     1PACOBT,PHABT,VO2,VCO2,VCO2/VO2                               
1130  FORMAT(
     1' HB  =',F7.1,'; HCT  =',F7.1,';',13X,' PvO2 =',F6.1,'; PvCO2=',
     2F5.1,'; PHv=',F4.2,/,
     3' P50 =',F7.1,'; PIO2 =',F7.1,'; PICO2=',F5.1,'; PaO2 =',F6.1,
     4'; PaCO2=',F5.1,'; PHa=',F4.2,/,
     5' VO2 =',F7.1,'; VCO2 =',F7.1,';   R  =',F5.2,';')
      GO TO 1160                                                        
1140  WRITE(3,1150)HB,HCRIT,PHV,P50,PIO2,PICO2,PAO2BT,PACOBT,PHABT,
     1PVO2,PVCO2,PVCO2/PVO2
1150  FORMAT(
     1' HB  =',F7.1,'; HCT  =',F6.1,'; TOLERANCE =',F7.1,/,
     2' P50 =',F7.1,'; PIO2 =',F6.1,'; PICO2 =',F6.1,'; PaO2 =',F6.1,
     3'; PaCO2=',F6.1,'; PHa =',F5.2,/,
     4' VO2 =',F7.1,'; VCO2 =',F6.1,';   R  =',F6.2)
1160  WRITE(3,1170)EXPFAC                                               
1170  FORMAT(' HYPO/HYPERBARIC CORRECTION FACTOR FOR EXPIRED GAS'
     1' VALUES=',F6.3)     
      IPF=0                                                             
      IF(PV(1).EQ.0.0 .OR.PA(1).EQ.0.0 .OR.PE(1).EQ.0.0) IPF=1          
      IF(IPF.EQ.0) WRITE(3,680)                                         
      IF(PV(1).EQ.0.0.AND.IPAPV.EQ.1) WRITE(3,740)
      IF(PV(1).EQ.0.0.AND.IPAPV.EQ.2) WRITE(3,741)         
      IF(PA(1).EQ.0.0) WRITE(3,700)                                     
      IF(PE(1).EQ.0.0) WRITE(3,720)                                     
      WRITE(3,760)                                                      
      WRITE(3,770)(S(I),I=1,NGASES)                                     
      WRITE(3,780)(PC(I),I=1,NGASES)
      WRITE(3,781)(PCBODY(I),I=1,NGASES)
      WRITE(3,819)(PCFACT(I),I=1,NGASES)                                    
      IF(IRORE.NE.0) GO TO 1180                                         
      WRITE(3,820)(R(I),I=1,NGASES)                                     
      IF(IPF.EQ.0) WRITE(3,830)(RMV(I),I=1,NGASES)                      
      WRITE(3,840)(E(I),I=1,NGASES)                                     
      IF(IPF.EQ.0) WRITE(3,850)(EMV(I),I=1,NGASES)                      
      WRITE(3,860)(WT(I),I=1,NGASES)                                    
      GO TO 1190                                                        
1180  WRITE(3,840)(E(I),I=1,NGASES)                                     
      IF(IPF.EQ.0) WRITE(3,880)(EMV(I),I=1,NGASES)                      
      WRITE(3,820)(R(I),I=1,NGASES)                                     
      IF(IPF.EQ.0) WRITE(3,890)(RMV(I),I=1,NGASES)                      
      WRITE(3,900)(WT(I),I=1,NGASES)                                    
1190  IF(IPF.EQ.0) WRITE(3,920)(QTCALC(I),I=1,NGASES)                   
      IF(IPF.EQ.0) WRITE(3,930)VE,QT                                    
      IF(IPF.EQ.1) WRITE(3,970)VE,QT                                    
C************
C      IF(LCASE.EQ.3) GO TO 1240                                         
C************
      WRITE(7,1200) (PA(I),I=1,NGASES)                                   
      WRITE(7,1210) (GA(I),I=1,NGASES)                                   
      WRITE(7,1200) (PE(I),I=1,NGASES)                                   
      WRITE(7,1210) (GE(I),I=1,NGASES)                                   
      WRITE(7,1200) (PV(I),I=1,NGASES)                                   
      WRITE(7,1210) (GV(I),I=1,NGASES)                                   
      WRITE(7,1220) VEO,QT,PB,TEMPB,TEMPR,VGA,VBA,VHA,VGV,VBV,VHV        
      IF(PV(1).GT.0.0) WRITE(7,1230) HB,HCRIT,PVO2,PVCO2,PHV,FIO2,       
     &FICO2,P50,PMAO2,PMACO2,PHA,VO2,VCO2                                          
      IF(PV(1).EQ.0.0) WRITE(7,1230) HB,HCRIT,VO2,VCO2,TOL,FIO2,FICO2,         
     & P50,PMAO2,PMACO2,PHA                                             
1200  FORMAT(6(1X,F6.1))
1210  FORMAT(1X,6(F5.0,2X))
1220  FORMAT(F7.2,F6.2,2X,F5.1,2F6.1,6F7.2)
1230  FORMAT(2(2X,F4.1),2F7.1,F9.2,1X,2F6.4,F6.1,/,2F7.1,F6.2,2F8.1)
1240  CONTINUE 
      PICO2=(PB-SVPB)*FICO2
      WRITE(9,1250) VO2,VCO2,PIO2,PB,TEMP,HB,HCRIT,PICO2,BX,DP50         
1250  FORMAT(F7.1,F8.1,F8.2,F7.1,3F6.1,F7.2,2F6.2)                      
      IF(PV(1).GT.0.0) GO TO 1280                                       
      WRITE(9,1260) PVO2,PVCO2,Y1,Y2,X3,X4,X5,SO2                        
      WRITE(9,1270) PHABT,PHV,APCO2,BPCO2,PAO2BT,PACOBT                 
1260  FORMAT(F7.1,6F7.1,F9.5)                                           
1270  FORMAT(F9.2,F9.2,2F6.1,2F7.1)                                     
      GO TO 1290                                       
1280  WRITE(9,1260) PVO2BT,PVCOBT,Y1,Y2,X3,X4,X5,SO2                     
      WRITE(9,1270) PHABT,PHVBT,APCO2,BPCO2,PAO2BT,PACOBT               
1290  WRITE(9,1300) NGSS,NVAQS,ZZ,VQLO,VQHI,VE,SVPB,QT,TOL              
1300  FORMAT(I3,I4,F6.1,F7.3,F7.1,2F8.2,F6.2,F8.0)                      
      IF(IGAS.EQ.0) GO TO 1320                                          
      DO 1310 I=IGAS,NGSS                                                
      PCC(I)=PCBODY(I+1)                                                    
      RMV(I)=RMV(I+1)                                                   
      EMV(I)=EMV(I+1)                                                   
      WT(I)=WT(I+1)                                                     
1310  CONTINUE                                                          
1320  CONTINUE
      IF(IGAS.NE.0) WRITE(9,260) (PCC(I),I=1,NGSS)
      IF(IGAS.EQ.0) WRITE(9,260) (PCBODY(I),I=1,NGSS)              
      WRITE(9,1330) (RMV(I),I=1,NGSS)                                   
      WRITE(9,1330) (EMV(I),I=1,NGSS)                                   
      WRITE(9,1340) (WT(I),I=1,NGSS)                                    
1330  FORMAT(F12.6,7F12.6)                                              
1340  FORMAT(F11.2,7(F11.2,1X))                                         
      WRITE(3,350)                                                      
1350  CONTINUE                                                          
      IF(LCASE.NE.1) CLOSE(8)                                           
      IF(LCASE.NE.2) GO TO 1360                                         
      L1=NRUNS+1                                                        
      NRUNS=NTOT                                                        
      IQ='N'                                                            
      LCASE=1                                                           
      GO TO 360                                                         
1360  CLOSE(9)                                                          
      IF(LCASE.NE.3) CLOSE(7)                                           
      STOP                                                              
      END                                                               
      SUBROUTINE TEMPCR(PHP,PCO2,PO2,VC,T,PHPT,PCO2T,PO2T,BEB)          
C      CORRECTS BLOOD GAS & PH VALUES TO BODY TEMP & CALCS BASE EXCESS  
      REAL N,N1,MIN,MAX                                                 
      A=15.                                                             
      B=2045.                                                           
      C=-2000.                                                          
      D=2400.                                                           
      E=31100.                                                          
      F=2.4E+6                                                          
      PP=75.0                                                           
      N=-PO2*10.**(.48*(PHP-7.4))                                       
      SO2=(N**4+A*N**3+B*N**2+C*N)/(N**4+A*N**3+D*N**2+E*N+F)           
      DSO2BB=-VC*(1-SO2)*(8.5+PCO2*(.01-.05526*10.**(PHP-7.4)))         
      RH=.627-.440*(PHP-7.4)                                            
      PK=6.099-.04167*(PHP-7.4)                                         
      HBO2NH=20.1*VC*.2658*PCO2/(.8238*PCO2+10.**(8.-PHP)/RH*           
     &(8.611+10.**(8.-PHP)/RH))                                         
      PB=(1.-VC)*PP*(.2413+.104*(PHP-7.4))                              
      PHC=PHP+ALOG10(RH)                                                
      HB=20.1*VC*(10.625*PHC-.5*PHC**2-48.46)                           
      HCO3=.0306*PCO2*10.**(PHP-PK)*(1.+VC*(.7676*RH-1.))               
      BBB=HCO3+HB+PB+HBO2NH+DSO2BB                                      
      BEB=BBB-24.48-31.395*VC-.2413*(1.-VC)*PP                          
      N1=-PO2*10.**(.48*(PHP-7.4)-.0013*BEB)                            
      PCO2T=PCO2*10.**(.019*(T-37.))                                    
      PHPT=PHP-(T-37.)*(.0147+.00654*(PHP-7.4))                         
      SO237=(N1**4+A*N1**3+B*N1**2+C*N1)/(N1**4+A*N1**3+D*N1**2+        
     &E*N1+F)                                                           
      SM=-PO2*10.**(.48*(PHPT-7.4)-.0013*BEB-.024*(T-37.))              
      SP=1.+ALOG10(T/37.)+.00012*(T-37.)**2                             
      ST=342.18*VC/(PO2*(.02114+.00516*VC))                             
      SA=A/SM+SP*(ST*(1.-SO237)-1.)                                     
      SB=(D/SM+A*SP*(ST*(1.-SO237)-1.))/SM                              
      SC=(E/SM+SP*(B*ST-D*(1.+ST*SO237)))/SM**2                         
      SD=(F/SM+SP*(C*ST-E*(1.+ST*SO237)))/SM**3                         
      SE=-F*SP*(1.+ST*SO237)/SM**4                                      
      MIN=0.                                                            
      MAX=1.                                                            
      CALCX=T-37.                                                       
      IF(CALCX)20,10,70                                                 
10    X=1.                                                              
      GO TO 120                                                         
20    DO 50, J=1,24                                                     
      XT=(MAX+MIN)/2.                                                   
      Y=XT**5+SA*XT**4+SB*XT**3+SC*XT**2+SD*XT+SE                       
      IF(Y) 40,60,30                                                    
30    MAX=XT                                                            
      GO TO 50                                                          
40    MIN=XT                                                            
50    CONTINUE                                                          
60    X=XT                                                              
      GO TO 120                                                         
70    DO 100, K=1,24                                                    
      RXT=(MAX+MIN)/2.                                                  
      Y=1.+SA*RXT+SB*RXT**2+SC*RXT**3+SD*RXT**4+SE*RXT**5               
      IF(Y)80,110,90                                                    
80    MAX=RXT                                                           
      GO TO 100                                                         
90    MIN=RXT                                                           
100   CONTINUE                                                          
110   X=1./RXT                                                          
120   PO2T=X*PO2                                                        
      SL=SM*X                                                           
      SO2T=(SL**4+A*SL**3+B*SL**2+C*SL)/(SL**4+A*SL**3+D*               
     &SL**2+E*SL+F)                                                     
      RETURN                                                            
      END                                                               
