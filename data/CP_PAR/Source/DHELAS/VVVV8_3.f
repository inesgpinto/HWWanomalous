C     This File is Automatically generated by ALOHA 
C     The process calculated in this file is: 
C     Metric(1,4)*Metric(2,3) + Metric(1,3)*Metric(2,4) -
C      2*Metric(1,2)*Metric(3,4)
C     
      SUBROUTINE VVVV8_3(V1, V2, V4, COUP, M3, W3,V3)
      IMPLICIT NONE
      COMPLEX*16 CI
      PARAMETER (CI=(0D0,1D0))
      COMPLEX*16 COUP
      REAL*8 M3
      REAL*8 OM3
      REAL*8 P3(0:3)
      COMPLEX*16 TMP0
      COMPLEX*16 TMP2
      COMPLEX*16 TMP5
      COMPLEX*16 TMP6
      COMPLEX*16 TMP7
      COMPLEX*16 TMP8
      COMPLEX*16 V1(*)
      COMPLEX*16 V2(*)
      COMPLEX*16 V3(6)
      COMPLEX*16 V4(*)
      REAL*8 W3
      COMPLEX*16 DENOM
      OM3 = 0D0
      IF (M3.NE.0D0) OM3=1D0/M3**2
      V3(1) = +V1(1)+V2(1)+V4(1)
      V3(2) = +V1(2)+V2(2)+V4(2)
      P3(0) = -DBLE(V3(1))
      P3(1) = -DBLE(V3(2))
      P3(2) = -DIMAG(V3(2))
      P3(3) = -DIMAG(V3(1))
      TMP0 = (V4(3)*V1(3)-V4(4)*V1(4)-V4(5)*V1(5)-V4(6)*V1(6))
      TMP2 = (V4(3)*V2(3)-V4(4)*V2(4)-V4(5)*V2(5)-V4(6)*V2(6))
      TMP5 = (V1(3)*V2(3)-V1(4)*V2(4)-V1(5)*V2(5)-V1(6)*V2(6))
      TMP6 = (V2(3)*P3(0)-V2(4)*P3(1)-V2(5)*P3(2)-V2(6)*P3(3))
      TMP7 = (V1(3)*P3(0)-V1(4)*P3(1)-V1(5)*P3(2)-V1(6)*P3(3))
      TMP8 = (V4(3)*P3(0)-V4(4)*P3(1)-V4(5)*P3(2)-V4(6)*P3(3))
      DENOM = COUP/(P3(0)**2-P3(1)**2-P3(2)**2-P3(3)**2 - M3 * (M3 -CI
     $ * W3))
      V3(3)= DENOM*(OM3*P3(0)*(+CI*(TMP0*TMP6+TMP2*TMP7)-2D0 * CI
     $ *(TMP5*TMP8))+(-CI*(V2(3)*TMP0+V1(3)*TMP2)+2D0 * CI*(V4(3)*TMP5)
     $ ))
      V3(4)= DENOM*(OM3*P3(1)*(+CI*(TMP0*TMP6+TMP2*TMP7)-2D0 * CI
     $ *(TMP5*TMP8))+(-CI*(V2(4)*TMP0+V1(4)*TMP2)+2D0 * CI*(V4(4)*TMP5)
     $ ))
      V3(5)= DENOM*(OM3*P3(2)*(+CI*(TMP0*TMP6+TMP2*TMP7)-2D0 * CI
     $ *(TMP5*TMP8))+(-CI*(V2(5)*TMP0+V1(5)*TMP2)+2D0 * CI*(V4(5)*TMP5)
     $ ))
      V3(6)= DENOM*(OM3*P3(3)*(+CI*(TMP0*TMP6+TMP2*TMP7)-2D0 * CI
     $ *(TMP5*TMP8))+(-CI*(V2(6)*TMP0+V1(6)*TMP2)+2D0 * CI*(V4(6)*TMP5)
     $ ))
      END


