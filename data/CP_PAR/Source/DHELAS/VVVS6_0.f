C     This File is Automatically generated by ALOHA 
C     The process calculated in this file is: 
C     P(3,1)*Metric(1,2) - P(3,2)*Metric(1,2) - P(2,1)*Metric(1,3) +
C      P(2,3)*Metric(1,3) + P(1,2)*Metric(2,3) - P(1,3)*Metric(2,3)
C     
      SUBROUTINE VVVS6_0(V1, V2, V3, S4, COUP,VERTEX)
      IMPLICIT NONE
      COMPLEX*16 CI
      PARAMETER (CI=(0D0,1D0))
      COMPLEX*16 COUP
      REAL*8 P1(0:3)
      REAL*8 P2(0:3)
      REAL*8 P3(0:3)
      COMPLEX*16 S4(*)
      COMPLEX*16 TMP1
      COMPLEX*16 TMP12
      COMPLEX*16 TMP13
      COMPLEX*16 TMP15
      COMPLEX*16 TMP16
      COMPLEX*16 TMP3
      COMPLEX*16 TMP5
      COMPLEX*16 TMP6
      COMPLEX*16 TMP7
      COMPLEX*16 V1(*)
      COMPLEX*16 V2(*)
      COMPLEX*16 V3(*)
      COMPLEX*16 VERTEX
      P1(0) = DBLE(V1(1))
      P1(1) = DBLE(V1(2))
      P1(2) = DIMAG(V1(2))
      P1(3) = DIMAG(V1(1))
      P2(0) = DBLE(V2(1))
      P2(1) = DBLE(V2(2))
      P2(2) = DIMAG(V2(2))
      P2(3) = DIMAG(V2(1))
      P3(0) = DBLE(V3(1))
      P3(1) = DBLE(V3(2))
      P3(2) = DIMAG(V3(2))
      P3(3) = DIMAG(V3(1))
      TMP1 = (V3(3)*V2(3)-V3(4)*V2(4)-V3(5)*V2(5)-V3(6)*V2(6))
      TMP12 = (V2(3)*P1(0)-V2(4)*P1(1)-V2(5)*P1(2)-V2(6)*P1(3))
      TMP13 = (V1(3)*P2(0)-V1(4)*P2(1)-V1(5)*P2(2)-V1(6)*P2(3))
      TMP15 = (V3(3)*P1(0)-V3(4)*P1(1)-V3(5)*P1(2)-V3(6)*P1(3))
      TMP16 = (V3(3)*P2(0)-V3(4)*P2(1)-V3(5)*P2(2)-V3(6)*P2(3))
      TMP3 = (V1(3)*V3(3)-V1(4)*V3(4)-V1(5)*V3(5)-V1(6)*V3(6))
      TMP5 = (V1(3)*V2(3)-V1(4)*V2(4)-V1(5)*V2(5)-V1(6)*V2(6))
      TMP6 = (V2(3)*P3(0)-V2(4)*P3(1)-V2(5)*P3(2)-V2(6)*P3(3))
      TMP7 = (V1(3)*P3(0)-V1(4)*P3(1)-V1(5)*P3(2)-V1(6)*P3(3))
      VERTEX = COUP*S4(3)*(TMP1*(-CI*(TMP13)+CI*(TMP7))+(TMP3*(+CI
     $ *(TMP12)-CI*(TMP6))+TMP5*(-CI*(TMP15)+CI*(TMP16))))
      END


