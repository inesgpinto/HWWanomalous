C     This File is Automatically generated by ALOHA 
C     The process calculated in this file is: 
C     Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) -
C      2*Metric(1,2)*Metric(3,4)
C     
      SUBROUTINE VVVVS6P1N_1(V2, V3, V4, S5, COUP,V1)
      IMPLICIT NONE
      COMPLEX*16 CI
      PARAMETER (CI=(0D0,1D0))
      COMPLEX*16 COUP
      COMPLEX*16 S5(*)
      COMPLEX*16 TMP1
      COMPLEX*16 TMP2
      COMPLEX*16 TMP4
      COMPLEX*16 V1(6)
      COMPLEX*16 V2(*)
      COMPLEX*16 V3(*)
      COMPLEX*16 V4(*)
      TMP1 = (V3(3)*V2(3)-V3(4)*V2(4)-V3(5)*V2(5)-V3(6)*V2(6))
      TMP2 = (V4(3)*V2(3)-V4(4)*V2(4)-V4(5)*V2(5)-V4(6)*V2(6))
      TMP4 = (V4(3)*V3(3)-V4(4)*V3(4)-V4(5)*V3(5)-V4(6)*V3(6))
      V1(3)= COUP*(-S5(3))*(+CI*(V4(3)*TMP1+V3(3)*TMP2)-2D0 * CI*(V2(3)
     $ *TMP4))
      V1(4)= COUP*S5(3)*(+CI*(V4(4)*TMP1+V3(4)*TMP2)-2D0 * CI*(V2(4)
     $ *TMP4))
      V1(5)= COUP*S5(3)*(+CI*(V4(5)*TMP1+V3(5)*TMP2)-2D0 * CI*(V2(5)
     $ *TMP4))
      V1(6)= COUP*S5(3)*(+CI*(V4(6)*TMP1+V3(6)*TMP2)-2D0 * CI*(V2(6)
     $ *TMP4))
      END


