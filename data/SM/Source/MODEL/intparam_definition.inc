ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

C     Parameters that should not be recomputed event by event.
C     
      IF(READLHA) THEN

        G = 2 * DSQRT(AS*PI)  ! for the first init

        MDL_CHW = 0.000000D+00

        MDL_DMH2 = 0.000000D+00

        MDL_DGF = 0.000000D+00

        MDL_DGW = 0.000000D+00

        MDL_CH = 0.000000D+00

        MDL_DG1 = 0.000000D+00

        MDL_CHB = 0.000000D+00

        MDL_COMPLEXI = DCMPLX(0.000000D+00,1.000000D+00)

        MDL_MWSM = MDL_MW

        MDL_MW__EXP__2 = MDL_MW**2

        MDL_MZ__EXP__2 = MDL_MZ**2

        MDL_SQRT__2 = SQRT(DCMPLX(2.000000D+00))

        MDL_NB__2__EXP__0_25 = 2.000000D+00**2.500000D-01

        MDL_MH__EXP__2 = MDL_MH**2

        MDL_STH2 = 1.000000D+00-MDL_MW__EXP__2/MDL_MZ__EXP__2

        MDL_NB__10__EXP___M_40 = 1.000000D+01**-40

        MDL_MZ1 = MDL_MZ

        MDL_MH1 = MDL_MH

        MDL_MT1 = MDL_MT

        MDL_WZ1 = MDL_WZ

        MDL_WW1 = MDL_WW

        MDL_WH1 = MDL_WH

        MDL_WT1 = MDL_WT

        MDL_CTH = SQRT(DCMPLX(1.000000D+00-MDL_STH2))

        MDL_MW1 = MDL_MWSM

        MDL_SQRT__STH2 = SQRT(DCMPLX(MDL_STH2))

        MDL_STH = MDL_SQRT__STH2

        MDL_LAMBDASMEFT__EXP__2 = MDL_LAMBDASMEFT**2

        MDL_MT__EXP__2 = MDL_MT**2

        MDL_MH__EXP__6 = MDL_MH**6

        MDL_MWSM__EXP__6 = MDL_MWSM**6

        MDL_MH__EXP__4 = MDL_MH**4

        MDL_MWSM__EXP__4 = MDL_MWSM**4

        MDL_MWSM__EXP__2 = MDL_MWSM**2

        MDL_MZ__EXP__4 = MDL_MZ**4

        MDL_MZ__EXP__6 = MDL_MZ**6

        MDL_CTH__EXP__2 = MDL_CTH**2

        MDL_STH__EXP__2 = MDL_STH**2

        MDL_MB__EXP__2 = MDL_MB**2

        MDL_MZ__EXP__3 = MDL_MZ**3

        MDL_STH__EXP__4 = MDL_STH**4

        MDL_STH__EXP__6 = MDL_STH**6

        MDL_STH__EXP__3 = MDL_STH**3

        MDL_STH__EXP__5 = MDL_STH**5

        MDL_CTH__EXP__3 = MDL_CTH**3

        MDL_AEW = (MDL_GF*MDL_MW__EXP__2*(1.000000D+00-MDL_MW__EXP__2
     $   /MDL_MZ__EXP__2)*MDL_SQRT__2)/PI

        MDL_SQRT__GF = SQRT(DCMPLX(MDL_GF))

        MDL_VEVHAT = 1.000000D+00/(MDL_NB__2__EXP__0_25*MDL_SQRT__GF)

        MDL_LAM = (MDL_GF*MDL_MH__EXP__2)/MDL_SQRT__2

        MDL_SQRT__AEW = SQRT(DCMPLX(MDL_AEW))

        MDL_EE = 2.000000D+00*MDL_SQRT__AEW*SQRT(DCMPLX(PI))

        MDL_YB = (MDL_YMB*MDL_SQRT__2)/MDL_VEVHAT

        MDL_YT = (MDL_YMT*MDL_SQRT__2)/MDL_VEVHAT

        MDL_VEVHAT__EXP__2 = MDL_VEVHAT**2

        MDL_VEVT = (1.000000D+00+MDL_DGF/2.000000D+00)*MDL_VEVHAT

        MDL_G1 = MDL_EE/MDL_CTH

        MDL_GW = MDL_EE/MDL_STH

        MDL_EE__EXP__2 = MDL_EE**2

        MDL_GHAA = (MDL_EE__EXP__2*(-1.750000D+00+(4.000000D+00
     $   *(3.333333D-01+(7.000000D+00*MDL_MH__EXP__2)/(3.600000D+02
     $   *MDL_MT__EXP__2)))/3.000000D+00-(2.900000D+01*MDL_MH__EXP__6)
     $   /(1.680000D+04*MDL_MWSM__EXP__6)-(1.900000D+01*MDL_MH__EXP__4)
     $   /(1.680000D+03*MDL_MWSM__EXP__4)-(1.100000D+01*MDL_MH__EXP__2)
     $   /(1.200000D+02*MDL_MWSM__EXP__2)))/(8.000000D+00*PI**2)

        MDL_GHZA = (MDL_EE__EXP__2*(((4.583333D-01+(2.900000D+01
     $   *MDL_MH__EXP__6)/(1.008000D+05*MDL_MWSM__EXP__6)+(1.900000D
     $   +01*MDL_MH__EXP__4)/(1.008000D+04*MDL_MWSM__EXP__4)
     $   +(1.100000D+01*MDL_MH__EXP__2)/(7.200000D+02*MDL_MWSM__EXP__2)
     $   +(MDL_MH__EXP__4*MDL_MZ__EXP__2)/(2.100000D+03
     $   *MDL_MWSM__EXP__6)+(MDL_MH__EXP__2*MDL_MZ__EXP__2)/(2.800000D
     $   +02*MDL_MWSM__EXP__4)+(7.000000D+00*MDL_MZ__EXP__2)
     $   /(1.800000D+02*MDL_MWSM__EXP__2)+(6.700000D+01*MDL_MH__EXP__2
     $   *MDL_MZ__EXP__4)/(1.008000D+05*MDL_MWSM__EXP__6)+(5.300000D
     $   +01*MDL_MZ__EXP__4)/(1.008000D+04*MDL_MWSM__EXP__4)
     $   +(4.300000D+01*MDL_MZ__EXP__6)/(5.040000D+04*MDL_MWSM__EXP__6)
     $   -(3.100000D+01*MDL_CTH__EXP__2)/(2.400000D+01*MDL_STH__EXP__2)
     $   -(2.900000D+01*MDL_CTH__EXP__2*MDL_MH__EXP__6)/(2.016000D+04
     $   *MDL_MWSM__EXP__6*MDL_STH__EXP__2)-(1.900000D+01
     $   *MDL_CTH__EXP__2*MDL_MH__EXP__4)/(2.016000D+03
     $   *MDL_MWSM__EXP__4*MDL_STH__EXP__2)-(1.100000D+01
     $   *MDL_CTH__EXP__2*MDL_MH__EXP__2)/(1.440000D+02
     $   *MDL_MWSM__EXP__2*MDL_STH__EXP__2)-(MDL_CTH__EXP__2
     $   *MDL_MH__EXP__4*MDL_MZ__EXP__2)/(5.600000D+02
     $   *MDL_MWSM__EXP__6*MDL_STH__EXP__2)-(3.100000D+01
     $   *MDL_CTH__EXP__2*MDL_MH__EXP__2*MDL_MZ__EXP__2)/(2.520000D+03
     $   *MDL_MWSM__EXP__4*MDL_STH__EXP__2)-(MDL_CTH__EXP__2
     $   *MDL_MZ__EXP__2)/(9.000000D+00*MDL_MWSM__EXP__2
     $   *MDL_STH__EXP__2)-(4.300000D+01*MDL_CTH__EXP__2
     $   *MDL_MH__EXP__2*MDL_MZ__EXP__4)/(2.016000D+04
     $   *MDL_MWSM__EXP__6*MDL_STH__EXP__2)-(1.700000D+01
     $   *MDL_CTH__EXP__2*MDL_MZ__EXP__4)/(1.120000D+03
     $   *MDL_MWSM__EXP__4*MDL_STH__EXP__2)-(5.000000D+00
     $   *MDL_CTH__EXP__2*MDL_MZ__EXP__6)/(2.016000D+03
     $   *MDL_MWSM__EXP__6*MDL_STH__EXP__2))*MDL_STH)/MDL_CTH
     $   +((3.333333D-01+(7.000000D+00*MDL_MH__EXP__2)/(3.600000D+02
     $   *MDL_MT__EXP__2)+(1.100000D+01*MDL_MZ__EXP__2)/(3.600000D+02
     $   *MDL_MT__EXP__2))*(5.000000D-01-(4.000000D+00*MDL_STH__EXP__2)
     $   /3.000000D+00))/(MDL_CTH*MDL_STH)))/(4.000000D+00*PI**2)

        MDL_BARLAM = (1.000000D+00-MDL_DGF-MDL_DMH2)*MDL_LAM

        MDL_GWSH = (MDL_EE*(1.000000D+00+MDL_DGW-(MDL_CHW
     $   *MDL_VEVHAT__EXP__2)/MDL_LAMBDASMEFT__EXP__2))/MDL_STH

        MDL_VEV = (1.000000D+00-(3.000000D+00*MDL_CH
     $   *MDL_VEVHAT__EXP__2)/(8.000000D+00*MDL_LAM
     $   *MDL_LAMBDASMEFT__EXP__2))*MDL_VEVT

        MDL_G1SH = (MDL_EE*(1.000000D+00+MDL_DG1-(MDL_CHB
     $   *MDL_VEVHAT__EXP__2)/MDL_LAMBDASMEFT__EXP__2))/MDL_CTH

        MDL_EE__EXP__3 = MDL_EE**3

        MDL_VEVHAT__EXP__3 = MDL_VEVHAT**3

      ENDIF
C     
C     Parameters that should be recomputed at an event by even basis.
C     
      AS = G**2/4/PI

      MDL_SQRT__AS = SQRT(DCMPLX(AS))

      MDL_GHGG2 = (-7.000000D+00*AS)/(7.200000D+02*PI)

      MDL_GHGG4 = AS/(3.600000D+02*PI)

      MDL_GHGG5 = AS/(2.000000D+01*PI)

      MDL_G__EXP__2 = G**2

      MDL_GHGG1 = MDL_G__EXP__2/(4.800000D+01*PI**2)

      MDL_GHGG3 = (AS*G)/(6.000000D+01*PI)

      MDL_G__EXP__3 = G**3

C     
C     Parameters that should be updated for the loops.
C     
C     
C     Definition of the EW coupling used in the write out of aqed
C     
      GAL(1) = 2.378414230005442133435D0*MDL_MW*DSQRT(1D0-MDL_MW**2
     $ /MDL_MZ**2)*DSQRT(MDL_GF)
      GAL(2) = 1D0

