C     This File is Automatically generated by ALOHA 
C     The process calculated in this file is: 
C     Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) -
C      2*Metric(1,2)*Metric(3,4)
C     
      SUBROUTINE VVVV8P1N_1(V2, V3, V4, COUP,V1)
      IMPLICIT NONE
      COMPLEX*16 CI
      PARAMETER (CI=(0D0,1D0))
      COMPLEX*16 COUP
      COMPLEX*16 TMP15
      COMPLEX*16 TMP16
      COMPLEX*16 TMP8
      COMPLEX*16 V1(6)
      COMPLEX*16 V2(*)
      COMPLEX*16 V3(*)
      COMPLEX*16 V4(*)
      TMP15 = (V2(3)*V4(3)-V2(4)*V4(4)-V2(5)*V4(5)-V2(6)*V4(6))
      TMP16 = (V3(3)*V4(3)-V3(4)*V4(4)-V3(5)*V4(5)-V3(6)*V4(6))
      TMP8 = (V2(3)*V3(3)-V2(4)*V3(4)-V2(5)*V3(5)-V2(6)*V3(6))
      V1(3)= COUP*(-1D0)*(+CI*(TMP8*V4(3)+V3(3)*TMP15)-2D0 * CI*(V2(3)
     $ *TMP16))
      V1(4)= COUP*(+CI*(TMP8*V4(4)+V3(4)*TMP15)-2D0 * CI*(V2(4)*TMP16))
      V1(5)= COUP*(+CI*(TMP8*V4(5)+V3(5)*TMP15)-2D0 * CI*(V2(5)*TMP16))
      V1(6)= COUP*(+CI*(TMP8*V4(6)+V3(6)*TMP15)-2D0 * CI*(V2(6)*TMP16))
      END


