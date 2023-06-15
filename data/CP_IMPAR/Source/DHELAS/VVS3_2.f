C     This File is Automatically generated by ALOHA 
C     The process calculated in this file is: 
C     Metric(1,2)
C     
      SUBROUTINE VVS3_2(V1, S3, COUP, M2, W2,V2)
      IMPLICIT NONE
      COMPLEX*16 CI
      PARAMETER (CI=(0D0,1D0))
      COMPLEX*16 COUP
      REAL*8 M2
      REAL*8 OM2
      REAL*8 P2(0:3)
      COMPLEX*16 S3(*)
      COMPLEX*16 TMP1
      COMPLEX*16 V1(*)
      COMPLEX*16 V2(6)
      REAL*8 W2
      COMPLEX*16 DENOM
      OM2 = 0D0
      IF (M2.NE.0D0) OM2=1D0/M2**2
      V2(1) = +V1(1)+S3(1)
      V2(2) = +V1(2)+S3(2)
      P2(0) = -DBLE(V2(1))
      P2(1) = -DBLE(V2(2))
      P2(2) = -DIMAG(V2(2))
      P2(3) = -DIMAG(V2(1))
      TMP1 = (V1(3)*P2(0)-V1(4)*P2(1)-V1(5)*P2(2)-V1(6)*P2(3))
      DENOM = COUP/(P2(0)**2-P2(1)**2-P2(2)**2-P2(3)**2 - M2 * (M2 -CI
     $ * W2))
      V2(3)= DENOM*S3(3)*(-CI*(V1(3))+CI*(P2(0)*TMP1*OM2))
      V2(4)= DENOM*S3(3)*(-CI*(V1(4))+CI*(P2(1)*TMP1*OM2))
      V2(5)= DENOM*S3(3)*(-CI*(V1(5))+CI*(P2(2)*TMP1*OM2))
      V2(6)= DENOM*S3(3)*(-CI*(V1(6))+CI*(P2(3)*TMP1*OM2))
      END


