C     This File is Automatically generated by ALOHA 
C     The process calculated in this file is: 
C     Metric(1,4)*Metric(2,3) - (Metric(1,3)*Metric(2,4))/2. -
C      (Metric(1,2)*Metric(3,4))/2.
C     
      SUBROUTINE VVVV11P1N_4(V1, V2, V3, COUP,V4)
      IMPLICIT NONE
      COMPLEX*16 CI
      PARAMETER (CI=(0D0,1D0))
      COMPLEX*16 COUP
      COMPLEX*16 TMP2
      COMPLEX*16 TMP6
      COMPLEX*16 TMP8
      COMPLEX*16 V1(*)
      COMPLEX*16 V2(*)
      COMPLEX*16 V3(*)
      COMPLEX*16 V4(6)
      TMP2 = (V2(3)*V1(3)-V2(4)*V1(4)-V2(5)*V1(5)-V2(6)*V1(6))
      TMP6 = (V1(3)*V3(3)-V1(4)*V3(4)-V1(5)*V3(5)-V1(6)*V3(6))
      TMP8 = (V2(3)*V3(3)-V2(4)*V3(4)-V2(5)*V3(5)-V2(6)*V3(6))
      V4(3)= COUP*1D0/2D0*(-2D0 * CI*(V1(3)*TMP8)+CI*(V2(3)*TMP6+TMP2
     $ *V3(3)))
      V4(4)= COUP*(-1D0/2D0)*(-2D0 * CI*(V1(4)*TMP8)+CI*(V2(4)*TMP6
     $ +TMP2*V3(4)))
      V4(5)= COUP*(-1D0/2D0)*(-2D0 * CI*(V1(5)*TMP8)+CI*(V2(5)*TMP6
     $ +TMP2*V3(5)))
      V4(6)= COUP*(-1D0/2D0)*(-2D0 * CI*(V1(6)*TMP8)+CI*(V2(6)*TMP6
     $ +TMP2*V3(6)))
      END


